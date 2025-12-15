# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional determinism (may reduce speed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def set_env_threads(openblas: str = "16", omp: str = "16", mkl: str = "16") -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", openblas)
    os.environ.setdefault("OMP_NUM_THREADS", omp)
    os.environ.setdefault("MKL_NUM_THREADS", mkl)
