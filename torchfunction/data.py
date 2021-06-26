import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from .device import str_2_device, todevice
from pyctlib import vector, totuple, vhelp
from collections.abc import Iterable
from .inspect import get_shape

def dataloader_sampler(dataloader: data.DataLoader):
    return next(iter(dataloader))

class DiscountedRandomSampler(data.Sampler):
    r"""Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, discount):
        self.discount = discount
        self.data_source = data_source
        self.selected_indexs = torch.randperm(len(self.data_source)).tolist()[:len(self)]

    def __iter__(self):
        return iter(self.selected_indexs)

    def __len__(self):
        return int(len(self.data_source) * self.discount)

class DataLoader(data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, todevice="cpu"):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
        self.device = str_2_device(todevice)
        self.shuffle = shuffle

    def __iter__(self):
        for x in super(DataLoader, self).__iter__():
            yield todevice(x, self.device)

    def partial_dataset(self, discount=1.0, batch_size=None, shuffle=None):
        if isinstance(discount, int) and discount > 1:
            assert discount <= len(self.dataset)
            new_len = discount
        elif isinstance(discount, int) and (discount == 0 or discount == 1):
            new_len = int(len(self.dataset) * discount)
        elif isinstance(discount, float) and (0 <= discount <= 1):
            new_len = int(len(self.dataset) * discount)
        else:
            raise ValueError
        small_dataset = vector.range(len(self.dataset)).sample(new_len, replace=False).map(lambda index: self.dataset[index])
        if shuffle is None:
            shuffle = self.shuffle
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(small_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers, collate_fn=self.collate_fn, pin_memory=self.pin_memory, drop_last=self.drop_last, timeout=self.timeout, worker_init_fn=self.worker_init_fn, todevice=self.device)

    def sample(self):
        ret = self.collate_fn(vector.randint(len(self.dataset), size=(self.batch_size, )).map(lambda index: self.dataset[index]))
        return todevice(ret)

    def __len__(self):
        return len(self.dataset)

    def to(self, todevice):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collate_fn, pin_memory=self.pin_memory, drop_last=self.drop_last, timeout=self.timeout, worker_init_fn=self.worker_init_fn, todevice=todevice)

class RandomDataset:

    def __init__(self, length, shapes, dtypes=float):
        self.length = length
        self.shapes = shapes
        if isinstance(shapes, Iterable) and len(shapes) == 0:
            raise ValueError
        if isinstance(shapes, tuple):
            self.num_var = 1
            self.shapes = vector([shapes])
        elif isinstance(shapes, list):
            self.num_var = len(shapes)
            self.shapes = vector(shapes).map(lambda x: totuple(x))
        elif isinstance(shape, int):
            self.num_var = 1
            self.shapes = vector([(shapes, )])
        if isinstance(dtypes, type):
            self.dtypes = vector([dtypes for _ in range(self.num_var)])
        else:
            self.dtypes = vector(dtypes)
        self.init()

    def __len__(self):
        return self.length

    def init(self):
        def ranndom_sample(length, shape, dtype):
            shape = (length, *shape)
            if dtype == float:
                return torch.randn(shape)
            elif dtype == int:
                return torch.randint(2, shape)
            elif isinstance(dtype, tuple) and dtype[0] == int:
                if len(dtype) == 2:
                    return torch.randint(dtype[1], shape)
                elif len(dtype) == 3:
                    return torch.randint(*dtype[1:3], shape)
            raise ValueError
        self.data = vector.zip(self.shapes, self.dtypes).map(lambda shape, dtype: ranndom_sample(self.length, shape, dtype), split_tuple=True)

    def __getitem__(self, index):
        ret = self.data.map(lambda x: x[index])
        if self.num_var == 1:
            return ret[0]
        else:
            return tuple(ret)

    @property
    def shape(self):
        return get_shape(self.data)
