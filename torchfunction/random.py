import torch

def randn_seed(*shape, device=None, seed=0):

    pre_seed = torch.seed()
    torch.random.manual_seed(seed)
    ret = torch.randn(*shape).to(device)
    torch.random.manual_seed(pre_seed)
    return ret
