"""
Utils
"""
import random
from copy import deepcopy
import torch
import numpy as np


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# def copy_base_model(model):
#     new_model = deepcopy(model)##
#     new_model.attention = model.attention
#     return new_model


def inf_loader(loader):
    while True:
        for data in loader:
            yield data
