from tpdataset import RawDataSet, DataDownloader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from pyctlib import vector, path, fuzzy_obj
import math
import torch

class MNIST(fuzzy_obj):

    def __init__(self, root="", transform="default"):

        self.root = root
        if isinstance(transform, str) and transform == "default":
            self.trans = transforms.ToTensor()
        else:
            self.trans = transform

        self.__train_set = datasets.MNIST(root=str(root), train=True, transform=self.trans, download=True)
        self.__test_set = datasets.MNIST(root=str(root), train=False, transform=self.trans, download=False)

    @property
    def train_set(self):
        if hasattr(self, "_MNIST__train_set_vector"):
            return self.__train_set_vector
        self.__train_set_vector = vector(self.__train_set, str_function=lambda x: "\n".join(["Dataset MNIST", "    Number of datapoints: {}".format(x.length), "    Split: Train"]))
        return self.__train_set_vector

    @property
    def test_set(self):
        if hasattr(self, "_MNIST__test_set_vector"):
            return self.__test_set_vector
        self.__test_set_vector = vector(self.__test_set, str_function=lambda x: "\n".join(["Dataset MNIST", "    Number of datapoints: {}".format(x.length), "    Split: Test"]))
        return self.__test_set_vector

    def show_image(self, image, y_labels=None):

        import matplotlib.pyplot as plt
        if isinstance(image, torch.Tensor) and (image.dim() == 2 or image.shape[0] == 1):
            image = image.squeeze()
            plt.imshow(image, cmap="gray", interpolation=None)
            if y_labels is not None:
                plt.title("Ground Truth: {}".format(y_labels))
            plt.xticks([])
            plt.yticks([])
            plt.show()
        elif isinstance(image, list) or isinstance(image, tuple) or isinstance(image, torch.Tensor) and image.dim() == 3:
            if isinstance(image, tuple):
                image = vector([image])
            if isinstance(image, list):
                n = image.shape[0]
            else:
                n = length(image)
            if n > 100:
                raise RuntimeError("{} images are displaied simutaneously".format(n))
            width = math.ceil(math.sqrt(n))
            for index in range(n):
                plt.subplot(math.ceil(n / width), width, index + 1)
                plt.tight_layout()
                if isinstance(image[index], tuple):
                    plt.imshow(image[index][0].squeeze(), cmap="gray", interpolation=None)
                    plt.title("Ground Truth: {}".format(image[index][1]))
                else:
                    plt.imshow(image[index].squeeze(), cmap="gray", interpolation=None)
                    if y_labels is not None:
                        plt.title("Ground Truth: {}".format(y_labels[index]))
                plt.xticks([])
                plt.yticks()
            plt.show()

    def train_dataloader(self, batch_size=1, shuffle=True, num_workers=0, pin_memory=True):
        return DataLoader(self.__train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def test_dataloader(self, batch_size=1, shuffle=True, num_workers=0, pin_memory=True):
        return DataLoader(self.__test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
