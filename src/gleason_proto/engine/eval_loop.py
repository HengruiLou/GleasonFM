# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from gleason_proto.utils.config import get_device
from gleason_proto.data.dataset import PatchDataset
from gleason_proto.data.transforms import default_patch_transforms
from gleason_proto.models.ctrans_backbone import load_ctran
from gleason_proto.models.proto_metric import ProtoMetricNet
from gleason_proto.engine.checkpoint import load_checkpoint

def evaluate(cfg: dict):
    device = get_device(cfg.get("device", "auto"))

    ckpt_path = cfg["paths"]["ckpt_path"]
    ck = load_checkpoint(ckpt_path, device)

    proto_model = ProtoMetricNet(
        centers=ck["centers"],
        labels=ck["labels"],
        weights=ck["weights"],
        embed_dim=int(ck["embed_dim"]),
        tau=float(ck["tau"]),
        margin_vec=ck["margin_vec"],
    ).to(device)
    proto_model.load_state_dict(ck["raw"]["state_dict"], strict=True)
    proto_model.eval()

    # Backbone: load pretrained then override with finetuned sd if exists
    backbone = load_ctran(cfg["paths"]["ctrans_pretrained"], device)
    if ck["backbone_sd"] is not None:
        backbone.load_state_dict(ck["backbone_sd"], strict=True)
    backbone.eval()

    tfm = default_patch_transforms()
    ds = PatchDataset(cfg["data"]["test_csv"], tfm)
    dl = DataLoader(
        ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
    )

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, lbls in dl:
            imgs = imgs.to(device)
            feats = backbone(imgs)
            logits, _ = proto_model(feats, y=None)
            pred = logits.argmax(1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend([int(x) for x in (lbls if isinstance(lbls, list) else lbls)])

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n", classification_report(y_true, y_pred, digits=4))
