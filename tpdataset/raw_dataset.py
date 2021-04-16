from pyctlib import vector, path, EmptyClass, touch
from pyctlib.vector import NoDefault
import urllib
import gzip
from torchvision.datasets import MNIST

minist = MNIST(path("."), download=True)

class RawDataSet:

    raw = "raw"
    processed = "processed"

    def __init__(root="", url=NoDefault, download=True):
        if root:
            self.root = path(root).abs().mkdir()
        else:
            self.root = path("./dataset").abs().mkdir()

        self.url = vector(url)
        self.download = download

    @property
    def name(self):
        if touch(lambda: self._name):
            return self._name
        return self.__class__.name

    def download(self):

        if self.check():
            return

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

    def check(self):
        if len((self.root / self.raw).ls()) > 0 and len((self.root / self.processed).ls()) > 0:
            return True
        return False
