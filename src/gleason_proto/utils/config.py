# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict
import yaml

def _expand_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = _expand_env(cfg)
    return cfg

def get_device(device_cfg: str):
    import torch
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)
