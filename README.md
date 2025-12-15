# GleasonFM: A Prostate-Specific Pathology Foundation Model via Teacher-Guided Prototype Composition Learning
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![PyTorch](https://img.shields.io/badge/PyTorch-2.9-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10.18-brightgreen)
> Abstract: Prostate cancer is one of the most common malignancies in men, and clinical decision-making relies
heavily on Gleason score. However, substantial interobserver continues to hinder deployment of AI-assisted diagnostic systems. Recent years, pathology foundation models (PFMs) have demonstrated strong performance on general cancer-related tasks such as classification and localization. Nevertheless, these approaches typically treat prostate cancer as just another dataset for generic encoder and lack disease-specific modeling aligned with the structured Gleason grading system, limiting their practical utility in prostate pathology. To address this gap, we propose a Gleason-aware foundation model specifically tailored for prostate (GleasonFM). GleasonFM is built on a weakly supervised, teacher-guided dynamic multi-prototype contrastive learning framework. A multi-level graph teacher provides benign/malignant priors and multi-scale regional importance cues, guiding student network to learn interpretable Gleason class prototypes whose weights are dynamically adjusted to mitigate label noise and center-specific artifacts. At slide level, we introduce ProtoBoW representation that aggregates soft assignments between patches and prototypes into a compact fingerprint aligned with clinically meaningful primary and secondary Gleason patterns. This unified representation enables joint modeling of cancer detection, Grade Group ordinal grading, and lesion localization within a single semantic space. Since inference only requires computing similarities between patch features and small set of class-specific prototypes, GleasonFM achieves low memory and computational overhead and has already been deployed at clinical scale in partner hospital’s pathology department.
<p align="center"> 
<img src="main.jpg">
</p>


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

## Configure paths via env vars
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

