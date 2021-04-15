from pyctlib import vector, path, EmptyClass, NoDefault

class RawDataSet:

    def __init__(root="", url=NoDefault, download=True):
        if root:
            self.root = path(root).abs()
        else:
            self.root = path("./dataset").abs()
        self.url = vector(url)
        self.download = download

    def download(self):

