import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import normalize
import math
import glob, os, shutil
from torch import nn
import ot
from torch.cuda.amp import autocast

def curriculum_scheduler(t, T, begin=0, end=1, mode=None, func=None):

    pho = t/T
    if mode == 'linear':
        ratio = pho
    elif mode == 'exp':
        ratio = 1 - math.exp(-4*pho)
    elif mode == 'customize':
        ratio = func(t, T)
    budget = begin + ratio * (end-begin)
    return budget, pho

def output_selected_rate(conf_l_mask, conf_u_mask, lowconf_u_mask):
    selected_rate_conf_l = torch.sum(conf_l_mask)/conf_l_mask.size(0)
    selected_rate_conf_u = torch.sum(conf_u_mask)/conf_u_mask.size(0)
    selected_rate_lowconf_u = torch.sum(lowconf_u_mask)/lowconf_u_mask.size(0)
    return selected_rate_conf_l, selected_rate_conf_u, selected_rate_lowconf_u

def get_masks(argmax_plabels, noisy_labels, gt_labels, selected_mask):
    with torch.no_grad():
        equal_label_mask = torch.eq(noisy_labels, argmax_plabels)
        conf_l_mask = torch.logical_and(selected_mask, equal_label_mask)
        conf_u_mask = torch.logical_and(selected_mask, ~equal_label_mask)
        lowconf_u_mask = ~selected_mask
        return conf_l_mask, conf_u_mask, lowconf_u_mask

def curriculum_structure_aware_PL(features, P, top_percent, L=None,
                                    reg_feat=2., reg_lab=2., temp=1, device=None, version='fast', reg_e=0.01, reg_sparsity=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a = torch.ones((P.shape[0],), dtype=torch.float64).to(device) / P.shape[0]
    top_percent = min(torch.sum(a).item(), top_percent)
    b = torch.ones((P.shape[1],), dtype=torch.float64).to(device) / P.shape[1] * top_percent
    P = P.double()
    coupling = ot.sinkhorn(a, b, M=-P, reg = reg_e, numItermax=1000, stopThr=1e-6,)
    total = features.size(0)

    max_values, _ = torch.max(coupling, 1)
    topk_num = int(total * top_percent)
    _, topk_indices = torch.topk(max_values, topk_num)

    selected_mask = torch.zeros((total,), dtype=torch.bool).cuda()
    selected_mask[topk_indices] = True
    return coupling, selected_mask

def OT_PL(model, eval_loader, num_class, batch_size, feat_dim=512, budget=1., sup_label=None, 
          reg_feat=0.5, reg_lab=0.5, version='fast', Pmode='out', reg_e=0.01, reg_sparsity=None, load_all=False): # 默认关掉 load_all
    
    model.eval()
    

    total_samples = len(eval_loader.dataset)
    
    all_pseudo_labels = torch.zeros((total_samples, num_class), dtype=torch.float32).cuda()
    all_noisy_labels = torch.zeros((total_samples,), dtype=torch.int64).cuda()
    all_gt_labels = torch.zeros((total_samples,), dtype=torch.int64).cuda()
    all_selected_mask = torch.zeros((total_samples,), dtype=torch.bool).cuda()
    all_conf = torch.zeros((total_samples,), dtype=torch.float32).cuda()
    all_argmax_plabels = torch.zeros((total_samples,), dtype=torch.int64).cuda()



    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):

            inputs = batch["img"].cuda(non_blocking=True)
            labels = batch["label"].cuda(non_blocking=True)
            gt_labels = batch["gttarget"].cuda(non_blocking=True)
            index = batch["index"].cuda(non_blocking=True)


            with autocast():

                logits = model(inputs) 
                out = logits.softmax(dim=-1)


            if Pmode == 'out':
                P = out
            elif Pmode == 'logP':
                P = (out + 1e-8).log()
            elif Pmode == 'softmax':
                P = out
            

            if sup_label is not None:

                L = torch.eye(num_class, dtype=torch.float32)[sup_label[labels]].cuda()
            else:
                L = torch.eye(num_class, dtype=torch.float32)[labels].cuda()


            couplings, selected_mask = curriculum_structure_aware_PL(
                None,
                P,
                top_percent=budget,
                L=L,
                reg_feat=reg_feat,
                reg_lab=reg_lab,
                reg_e=reg_e
            )


            row_sum = torch.sum(couplings, 1).reshape((-1, 1))
            pseudo_labels = torch.div(couplings, row_sum + 1e-10)
            max_value, argmax_plabels = torch.max(couplings, axis=1)

            conf = max_value * inputs.size(0) 
            conf = torch.clip(conf, min=0, max=1.0)


            all_noisy_labels[index] = labels
            all_gt_labels[index] = gt_labels
            all_selected_mask[index] = selected_mask
            all_conf[index] = conf.float()
            all_pseudo_labels[index] = pseudo_labels.float()
            all_argmax_plabels[index] = argmax_plabels


            del inputs, logits, out, P, L, couplings, pseudo_labels
    

    return all_pseudo_labels, all_noisy_labels, all_gt_labels, all_selected_mask, all_conf, all_argmax_plabels

