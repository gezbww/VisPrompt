#The core code will be made publicly available upon acceptance of the paper.


import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from dassl.utils import (
    MetricMeter, AverageMeter
)
import datetime
import time
import copy

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
      
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'VisPrompt',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        eot_indices = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection

        return x

class GeneralizedCrossEntropy(nn.Module):

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q
        return torch.mean(loss)


class SymmetricCrossEntropy(nn.Module):

    def __init__(self, alpha=0.5, beta=1.0, num_classes=None, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        if self.num_classes is None:
            C = inputs.shape[1]
        else:
            C = self.num_classes


        loss_ce = self.ce(inputs, targets)


        prob = F.softmax(inputs, dim=1)  # (B, C)

        with torch.no_grad():
            targets_onehot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)

            targets_smooth = targets_onehot * (1.0 - self.eps) + self.eps / C


        rce = - (prob * torch.log(targets_smooth)).sum(dim=1).mean()

        loss = self.alpha * loss_ce + self.beta * rce
        return loss




class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts_per_image = self.prompt_learner(image_features)

        B = image_features.shape[0]

        if prompts_per_image.dim() == 4:
            B, n_cls, L, dim = prompts_per_image.shape
            prompts_reshaped = prompts_per_image.view(B * n_cls, L, dim)
            tokenized = self.tokenized_prompts.unsqueeze(0).expand(B, -1, -1).contiguous()
            tokenized_reshaped = tokenized.view(B * n_cls, -1)
            text_features = self.text_encoder(prompts_reshaped, tokenized_reshaped)
            text_features = text_features.view(B, n_cls, -1)
        else:
            text_features = self.text_encoder(prompts_per_image, self.tokenized_prompts)
            text_features = text_features.unsqueeze(0).expand(B, -1, -1)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.einsum("bd,bkd->bk", image_features, text_features)

        return logits

