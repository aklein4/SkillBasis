
import torch
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np2torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(DEVICE)
    return torch.tensor(x).to(DEVICE)

def torch2np(x):
    return x.cpu().detach().numpy()


# nvidia-ml-py3
import nvidia_smi
try:
    nvidia_smi.nvmlInit()
except:
    pass

def get_mem_use():
    # get the percentage of vram that is being used
    try:
        max_use = 0
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            use_perc = round(1 - info.free / info.total, 2)
            max_use = max(max_use, use_perc)
        return max_use
    except:
        return 0