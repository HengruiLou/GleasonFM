# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def update_proto_weights(
    metric_net,
    sims_full: torch.Tensor,
    proto_raw: torch.Tensor,
    noise_track: torch.Tensor,
    teacher,
    teacher_malign_prob_fn,
    cfg_proto: dict,
    device,
):
    """Implements your dynamic proto_w update logic."""
    # metric_net.proto_w is a buffer on device
    for idx in range(len(metric_net.prototypes)):
        if metric_net.proto_w[idx] > 0.8:
            continue

        N_samples = sims_full.size(0)
        k_top = min(256, N_samples)
        topk = sims_full[:, idx].topk(k_top).values.mean().item()

        c = int(metric_net.proto_labels[idx].item())
        mask_cls = (metric_net.proto_labels.cpu() == c) & (metric_net.proto_w.cpu() > 0.8)
        mean_valid = sims_full[:, mask_cls].max(1).values.mean().item() if mask_cls.any() else 0.0

        t_proto = float(teacher_malign_prob_fn(teacher, proto_raw[idx:idx+1].to(device)).item())

        if (topk - mean_valid) > cfg_proto["delta_pos"] and (
            (c <= 2 and t_proto < 0.3) or (c >= 3 and t_proto > 0.7)
        ):
            metric_net.proto_w[idx].add_(0.05).clamp_(0, 1)

        elif (topk - mean_valid) < -cfg_proto["delta_neg"]:
            metric_net.proto_w[idx].sub_(0.05).clamp_(0, 1)

        if metric_net.proto_w[idx] <= cfg_proto["min_weight"]:
            noise_track[idx] += 1
            if noise_track[idx] >= cfg_proto["noise_max"]:
                metric_net.proto_w[idx] = 0.05
        else:
            noise_track[idx] = 0

    return metric_net.proto_w, noise_track
