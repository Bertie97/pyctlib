import math
import warnings
from typing import List

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from pyctlib import Logger
# logger = Logger(True)

class _enable_get_lr_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False

class ConstantLR(_LRScheduler):

    def __init__(self, optim, last_epoch=-1, verbose=False):
        super(ConstantLR, self).__init__(optim, last_epoch, verbose)
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        return self.base_lrs

    def _get_closed_form_lr(self):
        return self.base_lrs

class LinearWarmupLRWrapper(_LRScheduler):

    def __init__(self, lr_scheduler: _LRScheduler, warmup_epochs: int):

        last_epoch = lr_scheduler.last_epoch
        self.lr_scheduler = lr_scheduler
        self.warmup_epochs = warmup_epochs
        super(LinearWarmupLRWrapper, self).__init__(lr_scheduler.optimizer, lr_scheduler.last_epoch - 1, verbose=lr_scheduler.verbose)
        self.last_epoch = last_epoch
        temp = self
        while hasattr(temp, "lr_scheduler"):
            temp.lr_scheduler.last_epoch = last_epoch
            temp = temp.lr_scheduler

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        lr_value = self.lr_scheduler._get_closed_form_lr()
        if self.last_epoch < self.warmup_epochs:
            return [lr * self.last_epoch / self.warmup_epochs for lr in lr_value]
        else:
            return lr_value

    def step(self):
        temp = self
        while hasattr(temp, "lr_scheduler"):
            temp.lr_scheduler.last_epoch = self.last_epoch + 1
            temp = temp.lr_scheduler
        super(LinearWarmupLRWrapper, self).step()

class ExponentialDecayLRWrapper(_LRScheduler):

    def __init__(self, lr_scheduler: _LRScheduler, max_epochs: int, total_decay: float):

        assert total_decay <= 1
        last_epoch = lr_scheduler.last_epoch
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs
        self.total_decay = total_decay
        self.gamma = math.log(total_decay) / max_epochs
        super(ExponentialDecayLRWrapper, self).__init__(lr_scheduler.optimizer, lr_scheduler.last_epoch - 1, verbose=lr_scheduler.verbose)
        self.last_epoch = last_epoch
        temp = self
        while hasattr(temp, "lr_scheduler"):
            temp.lr_scheduler.last_epoch = last_epoch
            temp = temp.lr_scheduler

    def _get_closed_form_lr(self):
        lr_value = self.lr_scheduler._get_closed_form_lr()
        return [lr * math.exp(self.last_epoch * self.gamma) for lr in lr_value]

    def get_lr(self):
        return self._get_closed_form_lr()

    def step(self):
        temp = self
        while hasattr(temp, "lr_scheduler"):
            temp.lr_scheduler.last_epoch = self.last_epoch + 1
            temp = temp.lr_scheduler
        super(ExponentialDecayLRWrapper, self).step()

# if True:
#     from torch.optim.sgd import SGD
#     from pyctlib import vector
#     model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
#     optim = SGD(model, 0.1)

#     # scheduler_warmup is chained with schduler_steplr
#     # scheduler_steplr = LinearWarmupLRWrapper(CosineAnnealingLR(optim, T_max=20), 10)
#     scheduler_steplr = ExponentialDecayLRWrapper(LinearWarmupLRWrapper(CosineAnnealingLR(optim, T_max=100), 40), 1000, 0.01)
#     # scheduler_steplr = ExponentialDecayLRWrapper(ConstantLR(optim), 10, 0.1)
#     # scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)

#     # this zero gradient update is needed to avoid a warning message, issue #8.
#     optim.zero_grad()
#     optim.step()
#     l_v = vector()

#     for epoch in range(0, 1000):
#         print(epoch, optim.param_groups[0]['lr'])
#         l_v.append(optim.param_groups[0]['lr'])

#         optim.step()    # backward pass (update network)
#         # print(scheduler_steplr.last_epoch, "!")
#         scheduler_steplr.step()
#         # print(scheduler_steplr.last_epoch, "?")
