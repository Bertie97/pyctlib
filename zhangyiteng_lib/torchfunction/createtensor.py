import torch
import numpy as np

def truncated_normal(shape, mean=0.0, std=1.0, threshold=2):
    assert threshold > 1e-2
    with torch.no_grad():
        t = torch.zeros(shape)
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - threshold*std, t > mean + threshold*std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t
