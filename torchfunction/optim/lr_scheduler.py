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

    """
    learning_rate = original_lr * exp(max_epochs * gamma)
    where gamma = log(total_decay) / max_epochs
    """

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

class PowerDecayLRWrapper(_LRScheduler):

    def __init__(self, lr_scheduler: _LRScheduler, max_epochs: int, total_decay: float):

        assert total_decay <= 1
        last_epoch = lr_scheduler.last_epoch
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs
        self.total_decay = total_decay
        self.gamma = math.log(total_decay) / math.log(max_epochs + 1)
        super(PowerDecayLRWrapper, self).__init__(lr_scheduler.optimizer, lr_scheduler.last_epoch - 1, verbose=lr_scheduler.verbose)
        self.last_epoch = last_epoch
        temp = self
        while hasattr(temp, "lr_scheduler"):
            temp.lr_scheduler.last_epoch = last_epoch
            temp = temp.lr_scheduler

    def _get_closed_form_lr(self):
        lr_value = self.lr_scheduler._get_closed_form_lr()
        return [lr * ((self.last_epoch + 1) ** self.gamma) for lr in lr_value]

    def get_lr(self):
        return self._get_closed_form_lr()

    def step(self):
        temp = self
        while hasattr(temp, "lr_scheduler"):
            temp.lr_scheduler.last_epoch = self.last_epoch + 1
            temp = temp.lr_scheduler
        super(PowerDecayLRWrapper, self).step()
