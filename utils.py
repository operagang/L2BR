import random, os
from time import time
import torch
import numpy as np
from decoder import L2BR


def reset_seed(SEED=2024):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def load_model(args, device):
    model_loaded = L2BR(args, device)
    if torch.cuda.is_available():
        model_loaded.load_state_dict(torch.load(args.model_path, map_location={'cuda:0' : device ,'cuda:1': device , 'cuda:2' :device ,
                                                                   'cuda:3' : device ,'cuda:4': device , 'cuda:5' :device}, weights_only=True))
    else:
        model_loaded.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=True))
    return model_loaded

def create_log_directory(args):
    args.log_path
    os.makedirs(args.log_path, exist_ok=True)

def save_args_txt(args):
    fp = os.path.join(args.log_path, "log.txt")
    with open(fp, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
    return fp

def add_log_txt(args, epoch, score, t1):
    fp = os.path.join(args.log_path, "log.txt")
    t2 = time()
    with open(fp, 'a') as f:
        f.write('Epoch %d :  Validation Score: %1.3f, Elapsed time, %dmin%dsec\n' % (epoch, score, (t2 - t1) // 60, (t2 - t1) % 60))