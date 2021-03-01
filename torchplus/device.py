import torch
from pyctlib import vector
from pynvml import *


available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

def free_memory_amount(device_number):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(device_number)
    info = nvmlDeviceGetMemoryInfo(h)
    return info.free

def all_memory_amount(device_number):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(device_number)
    info = nvmlDeviceGetMemoryInfo(h)
    return info.total

available_gpus_memory = vector([free_memory_amount(i) for i in range(torch.cuda.device_count())])
all_gpus_memory = vector([all_memory_amount(i) for i in range(torch.cuda.device_count())])

warning_free_memory_threshold = 5

if torch.cuda.is_available():
    most_available_gpus = available_gpus_memory.max(with_index=True)[1]

    if available_gpus_memory[most_available_gpus] < warning_free_memory_threshold * 1.074e+9:
        print("Warning: the best gpu device is device {}".format(most_available_gpus))
        print("However, there are only {:.5} GB free memory memory in this GPU".format(available_gpus_memory[most_available_gpus] / 1.074e+9))
        tag = input("Do you want to proceed? [yes/no/y/n]:")
        if tag.lower() not in ["yes", "y"]:
            raise RuntimeError("There are no enough free memory left.")

    igpu = available_gpus_memory.index(max(available_gpus_memory))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)
    AutoDevice = torch.device(f"cuda:{igpu}")

else:
    AutoDevice = torch.device("cpu")

__all__ = ["available_gpus", "AutoDevice", "warning_free_memory_threshold", "available_gpus_memory", "all_gpus_memory"]
