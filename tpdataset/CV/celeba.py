from tpdataset import RawDataSet, DataDownloader

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from pyctlib import vector, path, fuzzy_obj
import math
import torch
from ..download_googledrive import download_file_from_google_drive

from functools import partial
import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.utils import check_integrity, verify_str_arg

class CelebA:

    base_folder = "celeba"
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(self, root="", transform="default", download=False):

        self.root = root
        if isinstance(transform, str) and transform == "default":
            self.trans = transforms.ToTensor()
        elif isinstance(transform, str) and transform == "vae":
            """
            transforms from https://github.com/AntixK/PyTorch-VAE/blob/8700d245a9735640dda458db4cf40708caf2e77f/experiment.py#L14
            """
            SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
            self.trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(64),
                transforms.ToTensor(),
                SetRange])
        else:
            self.trans = transform

        if download:
            self.download()

        self.__train_set = datasets.CelebA(root=str(root), split="train", transform=self.trans, download=False)
        self.__test_set = datasets.CelebA(root=str(root), split="test", transform=self.trans, download=False)
        self.__valid_set = datasets.CelebA(root=str(root), split="valid", transform=self.trans, download=False)

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            if check_integrity(fpath, md5):
                print("file {} already downloaded and veriried".format(filename))
                continue

            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder, filename), md5)
            print("file {} has been successfully downloaded!".format(filename))

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    @property
    def train_set(self):
        if hasattr(self, "_CelebA__train_set_vector"):
            return self.__train_set_vector
        self.__train_set_vector = vector(self.__train_set, str_function=lambda x: "\n".join(["Dataset CelebA", "    Number of datapoints: {}".format(x.length), "    Split: Train"]))
        return self.__train_set_vector

    @property
    def valid_set(self):
        if hasattr(self, "_CelebA__valid_set_vector"):
            return self.__valid_set_vector
        self.__valid_set_vector = vector(self.__valid_set, str_function=lambda x: "\n".join(["Dataset CelebA", "    Number of datapoints: {}".format(x.length), "    Split: Validation"]))
        return self.__valid_set_vector

    @property
    def test_set(self):
        if hasattr(self, "_CelebA__test_set_vector"):
            return self.__test_set_vector
        self.__test_set_vector = vector(self.__test_set, str_function=lambda x: "\n".join(["Dataset CelebA", "    Number of datapoints: {}".format(x.length), "    Split: Test"]))
        return self.__test_set_vector

    def train_dataloader(self, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True):
        return DataLoader(self.__train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    def valid_dataloader(self, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True):
        return DataLoader(self.__valid_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    def train_valid_dataloader(self, batch_size=1, shuffle=True, num_workers=0, pin_memory=True):
        return self.train_dataloader(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last), self.valid_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    def test_dataloader(self, batch_size=1, shuffle=False, num_workers=0, pin_memory=True):
        return DataLoader(self.__test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def __repr__(self):
        ret = ["Dataset CelebA"]
        ret.append("    # train: {}".format(len(self.__train_set)))
        ret.append("    # test: {}".format(len(self.__test_set)))
        ret.append("    # valid: {}".format(len(self.__valid_set)))
        return "\n".join(ret)

    def __str__(self):
        return self.__repr__()
