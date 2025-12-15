# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import torch

def save_checkpoint(path: str | Path, metric_net, backbone, best_loss: float):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": metric_net.state_dict(),
        "proto_labels": metric_net.proto_labels.detach().cpu(),
        "proto_w": metric_net.proto_w.detach().cpu(),
        "tau": float(metric_net.tau),
        "margin_vec": metric_net.margin_vec.detach().cpu(),
        "backbone": backbone.state_dict(),
        "best_loss": float(best_loss),
    }, str(path))

def load_checkpoint(path: str | Path, device):
    ck = torch.load(path, map_location="cpu")
    centers = ck["state_dict"]["prototypes"].to(device)
    labels = ck["proto_labels"].to(device)
    weights = ck["proto_w"].to(device)
    tau = float(ck["tau"])
    margin_vec = ck["margin_vec"].to(device)
    backbone_sd = ck.get("backbone", None)
    embed_dim = centers.size(1)
    return {
        "centers": centers,
        "labels": labels,
        "weights": weights,
        "tau": tau,
        "margin_vec": margin_vec,
        "embed_dim": embed_dim,
        "backbone_sd": backbone_sd,
        "raw": ck,
    }
