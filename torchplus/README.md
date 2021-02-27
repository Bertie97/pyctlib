# torchplus

## Introduction

[torchplus](https://github.com/Bertie97/pyctlib/tree/main/torchplus) is a package affiliated to project [`PyCTLib`](https://github.com/Bertie97/pyctlib). We encapsulated a new type on top of `pytorch` tensers, which we call it `torchplus.Tensor`. It has the same function as `torch.Tensor`, but it can automatically select the device it was on and provide batch or channel dimensions. Also, we try to provide more useful module for torch users to make deep learning to be implemented more easily. It relies `python v3.6+` with `torch v 1.7.0+`. ***Note that `torch v1.7.0` was released in 2020,*** *and it is necessary for this package as the inheritance behavior for this version is different from previous versions.* All original `torch` functions can be used for `torchplus` tensors. 

> Special features for `torchplus` are still under development. If unknown errors pop our, please use traditional `torch` code to bypass it and meanwhile it would be very kind of you to let us know if anything is needed: please contact us by [e-mail](https://github.com/Bertie97/pyctlib#Contact). 

```python
>>> import torchplus as tp
>>> import torch.nn as nn
>>> tp.set_autodevice(False)
>>> tp.manual_seed(0)
>>> t = tp.randn([3000], 400, requires_grad=True)
>>> LP = nn.Linear(400, 400)
>>> a = LP(t)
>>> a.sum().backward()
>>> print(t.grad)
Tensor([[-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702],
        [-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702],
        [-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702],
        ...,
        [-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702],
        [-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702],
        [-0.2986,  0.0267,  0.9059,  ...,  0.4563, -0.1291,  0.5702]], shape=torchplus.Size([3000], 400))
```

`torchplus` has all of following appealing features:

1. **Auto assign** the tensors to available `GPU` **device** by default. 
2. Use `[nbatch]` or `{nchannel}` to specify **the batch and channel dimensions**. i.e. `tp.rand([4], {2}, 20, 30)` returns a tensor of $20\times30$ matrices of channel 2 with batch size 4. One may also use `tensor.batch_dimension` to access to batch dimension, channel dimension can be operated likewise. 
3. Batch and channel dimension can help **auto matching the sizes** of two tensors in operations. For example, tensors of sizes `(3, [2], 4)` and `(3, 4)` can be automatically added together with axis of size 3 and 4 matched together. Some methods will also use this information. Sampling, for example, will take the batch dimension as priority.
4. The tensor object is **compatible with all `torch` functions**. 

## Installation

This package can be installed by `pip install torchplus` or moving the source code to the directory of python libraries (the source code can be downloaded on [github](https://github.com/Bertie97/pyctlib) or [PyPI](https://pypi.org/project/torchplus/)). 

```shell
pip install torchplus
```

## Usages

Not available yet, one may check the codes for usages.

## Acknowledgment

@Yiteng Zhang, Yuncheng Zhou: Developers