
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def py2np(x):
    return x.detach().cpu().numpy()

def np2py(x):
    return torch.from_numpy(x).to(DEVICE)