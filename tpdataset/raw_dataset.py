from pyctlib import vector, path, EmptyClass, NoDefault

class RawDataSet:

    raw = "raw"
    processed = "processed"

    def __init__(root="", url=NoDefault, download=True):
        if root:
            self.root = path(root).abs()
        else:
            self.root = path("./dataset").abs()
        self.url = vector(url)
        self.download = download

    def download(self):

        if self.check():
            return

        if 

    def check(self):
        if len((self.root / self.raw).ls()) > 0 and len((self.root / self.processed).ls()) > 0:
            return True
        return False
