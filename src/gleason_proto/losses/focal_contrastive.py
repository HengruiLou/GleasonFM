# -*- coding: utf-8 -*-
import torch

def focal_contrastive(sims, y, proto_labels, proto_w, tau: float, gamma: float):
    pos_mask = (proto_labels.unsqueeze(0) == y.unsqueeze(1)) & (proto_w.unsqueeze(0) > 0.8)
    neg_fill = torch.full_like(sims, -1e4)
    pos = torch.where(pos_mask, sims, neg_fill)
    s_pos, _ = pos.max(1)
    valid = s_pos > -1e3
    if not valid.any():
        return sims.sum() * 0

    exp = torch.exp(sims / tau)
    neg_mask = (proto_w.unsqueeze(0) > 0.2) & (~pos_mask)
    exp_neg = (exp * neg_mask * proto_w.unsqueeze(0)).sum(1)
    p = torch.exp(s_pos / tau) / (torch.exp(s_pos / tau) + exp_neg + 1e-12)
    w = (1 - p.detach()) ** gamma
    return (-(w * torch.log(p + 1e-12)))[valid].mean()
