from pyctlib import vector, path, EmptyClass, touch
from pyctlib.vector import NoDefault
from six.moves import urllib
import gzip
from torchvision.datasets import MNIST
import os
from http.client import HTTPResponse
from tqdm import trange, tqdm
from math import ceil, floor
import requests
import tarfile
from pyctlib.vector import totuple

__all__ = ["DataDownloader", "RawDataSet"]

class DataDownloader:

    def __init__(self, root="", name=None, urls=NoDefault, download=False):
        self._name = name
        if root:
            self.root = (path(root) / "dataset" / self.name).abs()
            self.root.assign_mainfolder(path(root).abs())
        else:
            self.root = path("./dataset/{}".format(self.name)).abs()
            self.root.assign_mainfolder(path(".").abs())

        self.raw_folder = self.root / "raw"
        self.processed_folder = self.root / "processed"

        temp = self.root.mkdir(True)
        mk_dirs = vector()
        if temp:
            mk_dirs.append(temp)

        self.urls = vector(urls)

        if download:
            mk_dirs.extend(vector(self.raw_folder, self.processed_folder).map(lambda x: x.mkdir(True)).filter())
            self.download()
            if not self.check(only_raw=True):
                print("Download failed")

        self._raw_files = None
        self._processed_files = None

        if bool(self.raw_files) and not bool(self.processed_files):
            if self.raw_files.length == 1:
                obj = self.raw_files[0]
                if obj.endswith("tar.gz"):
                    self.untar(obj, self.processed_folder)
                elif obj.endswith("zip"):
                    self.unzip(obj, self.processed_folder)

        if not self.check():
            print("Dataset not found. You can use download=True to download it.")
            mk_dirs.apply(lambda x: x.rm())

    @property
    def name(self):
        if touch(lambda: self._name):
            return self._name
        return self.__class__.__name__

    def download(self):

        if self.check(only_raw=True):
            return

        for url in self.urls:
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = self.raw_folder / filename
            self.download_data(file_path, url)

    @staticmethod
    def download_data(file, url):
        print('Downloading ' + url)
        try:
            with open(file, "wb") as f:
                r = requests.get(url, stream=True)
                length = r.headers["Content-Length"]
                if length:
                    length = int(length)
                    blocksize = 1024

                    for chunk in tqdm(r.iter_content(chunk_size=blocksize), total=ceil(length / blocksize), unit="KB", leave=True, desc=file.relative_path):
                        f.write(chunk)
                else:
                    data = urllib.request.urlopen(url)
                    f.write(data.read())
            return True
        except Exception as err:
            print(err)
            print("Download failed. Delete the error file.")
            file.rm()
            return False

    @staticmethod
    def untar(file, dirs):
        print("untar {} with target dir {}".format(file, dirs))
        with tarfile.open(file) as t:
            t.extractall(path=dirs)

    @staticmethod
    def unzip(file, dirs):
        print("unzip {} with target dir {}".format(file, dirs))
        import zipfile
        with zipfile.ZipFile(file, "r") as zip_input:
            zip_input.extractall(dirs)

    def check(self, only_raw=False):
        if only_raw:
            return self.raw_files.__bool__()
        return bool(self.raw_files) or bool(self.processed_files)

    @property
    def raw_files(self):
        if touch(lambda: self._raw_files, None):
            return self._raw_files
        if self.raw_folder.isdir():
            self._raw_files = self.raw_folder.ls()
            return self._raw_files

    @property
    def processed_files(self):
        if touch(lambda: self._processed_files, None):
            return self._processed_files
        if self.processed_folder.isdir():
            temp = self.processed_folder.ls()
            while len(temp) == 1 and temp[0].isdir():
                temp = temp[0].ls()
            self._processed_files = temp
        else:
            self._processed_files = vector()
        return self._processed_files

    def remove_raw_data(self):
        self.raw_folder.ls().map(lambda x: x.rm(False))

    def __repr__(self):
        fmt_str = "Dataset " + self.name
        fmt_str += "\n    Root location: " + self.root.abs()
        return fmt_str

class RawDataSet:

    def __init__(self, *data, split=("train", "test"), name=None):
        data = totuple(data)
        self._name = name
        self.split = split
        self.data = data
        if not len(data) == len(split):
            raise RuntimeError("# data: {} is incompatible with # split: {}".format(len(data), len(split)))

        for index in range(len(split)):
            self.__setattr__(split[index], data[index])

    def __getitem__(self, name):
        if name in split:
            return self.__getattribute__(name)
        raise RuntimeError("{} is not in [{}]".format(name, self.split))

    @property
    def name(self):
        if touch(lambda: self._name):
            return self._name
        return self.__class__.__name__

    def __repr__(self):
        fmt_str = "Dataset: {}\n".format(self.name)
        fmt_str += "    split: {}\n".format(self.split)
        for index in range(len(self.split)):
            fmt_str += "    # {}: {}\n".format(self.split[index], len(self.data[index]))
        return fmt_str
