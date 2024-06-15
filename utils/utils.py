import os
import sys
import random

import torch
import numpy as np

def ensure_pythonhashseed(seed: int = 0) -> None:
    # https://github.com/Lightning-AI/pytorch-lightning/issues/1939
    current_seed = os.environ.get("PYTHONHASHSEED")
    seed = str(seed)
    if current_seed is None or current_seed != seed:
        print(f'Setting PYTHONHASHSEED="{seed}"')
        os.environ["PYTHONHASHSEED"] = seed
        os.execl(sys.executable, sys.executable, *sys.argv)

def ensure_determinism(seed: int = 0) -> None:
    ensure_pythonhashseed(seed)
    torch.manual_seed(seed)
    np.random.seed(0)
    random.seed(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic
    #torch.use_deterministic_algorithms(True)
    #torch.utils.deterministic.fill_uninitialized_memory = True
    #os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
