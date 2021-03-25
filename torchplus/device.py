import torch
from pyctlib import vector, recursive_apply
from pynvml import *

available_gpu_ids = list(range(torch.cuda.device_count()))
available_gpus = [torch.cuda.device(i) for i in available_gpu_ids]

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

available_gpus_memory = vector([free_memory_amount(i) for i in available_gpu_ids])
all_gpus_memory = vector([all_memory_amount(i) for i in available_gpu_ids])

warning_free_memory_threshold = eval(os.environ.get('CUDA_RUN_MEMORY', '5'))

if torch.cuda.is_available():
    most_available_gpus = available_gpus_memory.max(with_index=True)[1]

    if available_gpus_memory[most_available_gpus] < warning_free_memory_threshold * 1.074e+9:
        print("Warning: the best gpu device is device {}".format(most_available_gpus))
        print("However, there are only {:.5} GB free memory memory in this GPU".format(available_gpus_memory[most_available_gpus] / 1.074e+9))
        tag = input("Do you want to proceed? [yes/no/y/n]:")
        if tag.lower() not in ["yes", "y"]:
            raise RuntimeError("There are no enough free memory left.")

    igpu = available_gpus_memory.index(max(available_gpus_memory))
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([f'{i}' for i in available_gpu_ids])
    AutoDevice = torch.device(f"cuda:{igpu}")
    print(f"Using GPU device {igpu}...")
else:
    AutoDevice = torch.device("cpu")

DeviceCPU = torch.device("cpu")

def recursive_autodevice(container):
    return recursive_apply(container, lambda x: x.to(AutoDevice))

__all__ = ["available_gpus", "AutoDevice", "warning_free_memory_threshold", "available_gpus_memory", "all_gpus_memory", "DeviceCPU", "recursive_autodevice"]
