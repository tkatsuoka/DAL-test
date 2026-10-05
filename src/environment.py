import os
import random
import numpy as np
import torch


def set_seed(seed):
    # for torch
    torch.manual_seed(seed)
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_env(parallel=False):
    if parallel is False:
        os.environ['MAX_JOBS'] = '1'
    else:
        os.environ["JOBLIB_START_METHOD"] = "spawn"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
