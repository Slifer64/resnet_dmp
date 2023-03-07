from typing import Optional
import torch
import numpy as np
import random

def set_all_seeds(seed_: Optional[int] = 0) -> None:
    """ Sets all seeds """
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    torch.cuda.manual_seed(seed_)
    torch.cuda.manual_seed_all(seed_)
    torch.backends.cudnn.deterministic = True
