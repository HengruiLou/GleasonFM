# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn

def init_ctran_patch_model(weight_path: str, unfreeze_ratio: float = 0.0):
    """Initialize CTransPath, remove head, optionally unfreeze last layers."""
    try:
        from models.ctran import ctranspath  # <-- adjust to your actual import
    except Exception as e:
        raise ImportError(
            "Failed to import CTransPath. Provide your implementation and adjust import "
            "`from models.ctran import ctranspath` in ctrans_backbone.py"
        ) from e

    m = ctranspath()
    m.head = nn.Identity()

    ck = torch.load(weight_path, map_location="cpu")
    state = ck.get("model", ck.get("state_dict", ck))
    m.load_state_dict(state, strict=True)

    for p in m.parameters():
        p.requires_grad = False

    if unfreeze_ratio and unfreeze_ratio > 0:
        params = list(m.parameters())
        n = max(1, int(len(params) * float(unfreeze_ratio)))
        for p in params[-n:]:
            p.requires_grad = True

    m.eval()
    return m

def load_ctran(weight_path: str, device):
    m = init_ctran_patch_model(weight_path, unfreeze_ratio=0.0)
    return m.to(device)
