# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from gleason_proto.utils.seed import set_seed, set_env_threads
from gleason_proto.utils.config import get_device
from gleason_proto.data.dataset import GleasonDataset
from gleason_proto.data.transforms import default_patch_transforms
from gleason_proto.data.samplers import build_weighted_sampler

from gleason_proto.models.ctrans_backbone import init_ctran_patch_model
from gleason_proto.models.proto_metric import ProtoMetricNet, build_margin_vec

from gleason_proto.losses.focal_contrastive import focal_contrastive
from gleason_proto.losses.kd import kd_loss_binary
from gleason_proto.engine.checkpoint import save_checkpoint

def train(cfg: dict):
    set_env_threads()
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))

    # ---- data ----
    tfm = default_patch_transforms()
    train_csv = cfg["data"]["train_csv"]
    train_set = GleasonDataset(train_csv, tfm)

    # ---- models ----
    backbone = init_ctran_patch_model(
        cfg["paths"]["ctrans_weight"],
        unfreeze_ratio=float(cfg["model"].get("unfreeze_ratio", 0.0)),
    ).to(device)


    # ---- prototype init: frozen feature extraction ----
    feat_bs = int(cfg["train"].get("feature_extract_batch_size", 32))
    features_frozen, labels_tensor = extract_frozen_features(
        backbone, train_set, device,
        batch_size=feat_bs,
        num_workers=int(cfg["data"].get("num_workers", 4)),
    )
    proto_vecs, proto_labels, proto_w, noise_track, n_cls = init_prototypes(
        features_frozen, labels_tensor, cfg["proto"], seed=int(cfg.get("seed", 42))
    )

    proto_raw = proto_vecs.clone().to(device)   # in 768-space for teacher prior

    margin_vec = build_margin_vec(
        num_classes=n_cls,
        margin_other=float(cfg["model"]["margin_other"]),
        margin_45=float(cfg["model"]["margin_45"]),
        device=device
    )

    metric_net = ProtoMetricNet(
        proto_vecs.to(device), proto_labels.to(device), proto_w.to(device),
        embed_dim=int(cfg["model"]["embed_dim"]),
        tau=float(cfg["model"]["tau"]),
        margin_vec=margin_vec
    ).to(device)

    # ---- loader ----
    sampler = build_weighted_sampler(train_set.labels, n_cls=n_cls)
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=sampler,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
    )

    # ---- optim ----
    params = [
        {"params": metric_net.parameters(), "lr": float(cfg["optim"]["lr_proj_head"])},
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": float(cfg["optim"]["lr_backbone"])},
    ]
    optimizer = torch.optim.Adam(params)

    sch_cfg = cfg["optim"].get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(sch_cfg.get("patience", 10)),
        factor=float(sch_cfg.get("factor", 0.5)),
    )

    # ---- training loop ----
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / cfg["paths"]["ckpt_name"]

    best_loss = 1e30
    log_every = int(cfg["train"].get("log_every_steps", 5000))
    step = 0
    proto_running = 0.0
    kd_running = 0.0

    num_epochs = int(cfg["train"]["num_epochs"])
    lambda_kd = float(cfg["distill"]["lambda_kd"])
    gamma_focal = float(cfg["distill"]["gamma_focal"])
    mask_low = float(cfg["distill"].get("mask_low", 0.3))
    mask_high = float(cfg["distill"].get("mask_high", 0.7))
    tau = float(cfg["model"]["tau"])
    grad_clip = float(cfg["train"]["grad_clip"])

    print(f"[INFO] device={device}, n_cls={n_cls}, prototypes={len(proto_vecs)}")
    backbone.train()

    for ep in range(1, num_epochs + 1):
        tot = 0.0
        cnt = 0.0

        for imgs, yb in tqdm(train_loader, desc=f"Ep{ep}/{num_epochs}"):
            step += 1
            imgs = imgs.to(device)
            yb = torch.tensor(yb, dtype=torch.long, device=device) if not torch.is_tensor(yb) else yb.to(device)

            xb = backbone(imgs)
            logits, sims = metric_net(xb, yb)

            loss_proto = focal_contrastive(
                sims, yb,
                metric_net.proto_labels, metric_net.proto_w,
                tau=tau, gamma=gamma_focal
            )

            with torch.no_grad():
                t_p = teacher_malign_prob(teacher, xb)

            logit_mal = logits[:, 3:].sum(-1)
            loss_kd = kd_loss_binary(logit_mal, t_p, mask_low=mask_low, mask_high=mask_high)

            loss = loss_proto + lambda_kd * loss_kd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(metric_net.parameters(), grad_clip)
            optimizer.step()

            proto_running += float(loss_proto.item())
            kd_running += float(loss_kd.item())

            if step % log_every == 0:
                print(f"[ep{ep} step{step}] proto={proto_running/log_every:.4f} "
                      f"kd_raw={kd_running/log_every:.4f} kd_eff={(lambda_kd*kd_running/log_every):.4f}")
                proto_running = 0.0
                kd_running = 0.0

            tot += float(loss.item()) * imgs.size(0)
            cnt += imgs.size(0)

        epoch_loss = tot / max(cnt, 1.0)
        print(f"[Ep{ep}] avg_loss={epoch_loss:.4f} λ_kd={lambda_kd:.3f} lr_backbone={optimizer.param_groups[1]['lr']:.1e}")
        scheduler.step(epoch_loss)

        # ---- dynamic proto_w update ----
        sims_full = recompute_sims_full(backbone, metric_net, train_set, device)
        update_proto_weights(
            metric_net=metric_net,
            sims_full=sims_full,
            proto_raw=proto_raw,
            noise_track=noise_track,
            teacher=teacher,
            teacher_malign_prob_fn=teacher_malign_prob,
            cfg_proto=cfg["proto"],
            device=device,
        )
        backbone.train()
        metric_net.train()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(ckpt_path, metric_net, backbone, best_loss)
            print(f"   [*] New best {best_loss:.4f} -> {ckpt_path}")

    print(f"[DONE] best_loss={best_loss:.4f}")
