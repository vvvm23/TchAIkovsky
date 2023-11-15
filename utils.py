import random

import numpy as np
import torch


def seed_others(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
