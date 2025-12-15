# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_margin_vec(num_classes: int, margin_other: float, margin_45: float, device):
    # Default mapping like your original: classes 4/5 use margin_45
    mv = [margin_other] * num_classes
    if num_classes >= 6:
        mv[4] = margin_45
        mv[5] = margin_45
    return torch.tensor(mv, device=device)

class ProtoMetricNet(nn.Module):
    def __init__(self, centers, labels, weights, embed_dim: int, tau: float, margin_vec):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(768, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim, bias=False),
        ).to(centers.device)

        with torch.no_grad():
            proj_centers = F.normalize(self.projection(centers), 2, 1)

    def forward(self, x, y=None):
        B = sims.size(0)
        C = int(self.margin_vec.numel())
        cls_logits = sims.new_full((B, C), -1e9)

        for c in range(C):
            m = (self.proto_labels == c) & (self.proto_w > 0.2)
            if m.any():
                cls_logits[:, c] = sims[:, m].max(1).values

        logits = cls_logits / self.tau
        if y is not None:
            logits = logits.clone()
            logits[torch.arange(B, device=logits.device), y] -= self.margin_vec[y]
        return logits, sims
