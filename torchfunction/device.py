import torch
from pyctlib import vector, recursive_apply
from pynvml import *

available_gpu_ids = list(range(torch.cuda.device_count()))
available_gpus = [torch.cuda.device(i) for i in available_gpu_ids]

nvmlInit()

def free_memory_amount(device_number):
    h = nvmlDeviceGetHandleByIndex(device_number)
    info = nvmlDeviceGetMemoryInfo(h)
    return info.free

def all_memory_amount(device_number):
    h = nvmlDeviceGetHandleByIndex(device_number)
    info = nvmlDeviceGetMemoryInfo(h)
    return info.total

def device_name(device_number):
    h = nvmlDeviceGetHandleByIndex(device_number)
    name = nvmlDeviceGetName(h)
    return name

def power_usage(device_number):
    h = nvmlDeviceGetHandleByIndex(device_number)
    return nvmlDeviceGetPowerUsage(h) / nvmlDeviceGetPowerManagementLimit(h)

available_gpus_memory = vector([free_memory_amount(i) for i in available_gpu_ids])
all_gpus_memory = vector([all_memory_amount(i) for i in available_gpu_ids])
available_gpu_name = vector(available_gpu_ids).map(device_name)
gpu_power_usage = vector(available_gpu_ids).map(power_usage)

nvmlShutdown()

warning_free_memory_threshold = eval(os.environ.get('CUDA_RUN_MEMORY', '5'))

if torch.cuda.is_available():
    most_available_gpus = vector.map_from([available_gpus_memory, gpu_power_usage], lambda m, p: m * max(1 - p, 0)**0.5).max(with_index=True)[1]

    if available_gpus_memory[most_available_gpus] < warning_free_memory_threshold * 1.074e+9:
        print("Warning: the best gpu device is device {}".format(most_available_gpus))
        print("However, there are only {:.5} GB free memory memory in this GPU".format(available_gpus_memory[most_available_gpus] / 1.074e+9))
        tag = input("Do you want to proceed? [yes/no/y/n]:")
        if tag.lower() not in ["yes", "y"]:
            raise RuntimeError("There are no enough free memory left.")

    igpu = most_available_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([f'{i}' for i in available_gpu_ids])
    AutoDeviceId = igpu
    AutoDevice = torch.device(f"cuda:{igpu}")
    print(f"Using GPU device {igpu}...")
else:
    AutoDevice = torch.device("cpu")

DeviceCPU = torch.device("cpu")

def todevice(x, device="cuda"):
    if device == "cuda":
        return todevice(x, AutoDevice)
    elif device == "cpu":
        return todevice(x, DeviceCPU)
    else:
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, tuple):
            return tuple([todevice(t, device) for t in x])
        elif isinstance(x, vector):
            return x.map(lambda x: todevice(x, device))
        elif isinstance(x, list):
            return [todevice(t, device) for t in x]
    raise ValueError

def str_2_device(device):
    if device == "cuda":
        return AutoDevice
    elif device == "cpu":
        return DeviceCPU
    else:
        return device

def recursive_autodevice(container):
    return recursive_apply(container, lambda x: x.to(AutoDevice))

__all__ = ["available_gpus", "AutoDevice", "warning_free_memory_threshold", "available_gpus_memory", "all_gpus_memory", "DeviceCPU", "recursive_autodevice", "AutoDeviceId", "todevice", "str_2_device", "gpu_power_usage"]
