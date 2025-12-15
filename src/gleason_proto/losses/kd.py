# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def kd_loss_binary(logit_mal: torch.Tensor, t_p: torch.Tensor, mask_low: float = 0.3, mask_high: float = 0.7):
    mask = (t_p < mask_low) | (t_p > mask_high)
    if mask.any():
        return F.binary_cross_entropy_with_logits(logit_mal[mask], t_p[mask])
    return logit_mal.sum() * 0
