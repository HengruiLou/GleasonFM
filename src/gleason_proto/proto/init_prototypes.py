# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

def init_prototypes(features_frozen: torch.Tensor, labels_tensor: torch.Tensor, proto_cfg: dict, seed: int):
    """KMeans init in normalized frozen feature space."""
    X_np = normalize(features_frozen.numpy(), axis=1)
    n_cls = int(labels_tensor.max().item()) + 1

    proto_vecs, proto_labels, proto_w, noise_track = [], [], [], []

    for c in range(n_cls):
        idx = np.where(labels_tensor.numpy() == c)[0]
        Nc = len(idx)
        k = min(proto_cfg["k_max"], max(1, int(math.sqrt(Nc / 1000)) + 1))

        km = MiniBatchKMeans(k, random_state=seed).fit(X_np[idx])
        centers = km.cluster_centers_
        sizes = np.bincount(km.labels_, minlength=k)

        min_sz = max(proto_cfg["min_keep"], int(proto_cfg["small_ratio"] * Nc))
        trusted = sizes >= min_sz

        proto_vecs.append(centers)
        proto_labels.extend([c] * len(centers))
        proto_w.extend(np.where(trusted, 1.0, 0.5))
        noise_track.extend([0] * len(centers))

    proto_vecs = torch.tensor(np.concatenate(proto_vecs), dtype=torch.float)
    proto_labels = torch.tensor(proto_labels, dtype=torch.long)
    proto_w = torch.tensor(proto_w, dtype=torch.float)
    noise_track = torch.tensor(noise_track, dtype=torch.long)

    return proto_vecs, proto_labels, proto_w, noise_track, n_cls
