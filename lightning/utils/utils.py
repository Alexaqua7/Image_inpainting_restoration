import torch
import numpy as np
import random
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id):
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)