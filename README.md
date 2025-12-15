# GleasonFM

A refactored project scaffold for:
- CTransPath backbone feature extraction
- multi-prototype initialization via MiniBatchKMeans
- ProtoMetricNet with margin (ArcFace-style) and focal-contrastive loss
- Teacher distillation (patch malignancy probability)

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format (CSV)
CSV should have 2 columns:
- col0: image path
- col1: integer label 

## Configure paths via env vars (recommended)
```bash
export DATA_ROOT=/path/to/your/data
export WEIGHTS_DIR=/path/to/your/weights
```

## Train
```bash
python scripts/train.py --config configs/train.yaml
```

## Eval
```bash
python scripts/eval.py --config configs/eval.yaml
```

## Notes on external models
This repo expects you to provide:
- a CTransPath implementation importable as `models.ctran.ctranspath`

If your codebase uses different imports, edit:
- `src/gleason_proto/models/ctrans_backbone.py`

