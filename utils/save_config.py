import os
import random

import numpy as np
import torch
import json

def generate_config(dst, **kwargs):
    print(kwargs)
    CFG = kwargs

    # def seed_everything(seed):
        # random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True

    # seed_everything(CFG['SEED'])
    json.dump(CFG, open(dst, "w"))
    return CFG



def load_config(src):
    return json.load(open(src, 'r'))