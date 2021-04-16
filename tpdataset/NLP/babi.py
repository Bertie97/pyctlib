from tpdataset import RawDataSet, DataDownloader
import tarfile
from torchtext.datasets import AG_NEWS

class BABI:

    def __init__(self, root="", name="", download=True):
        self.downloader = DataDownloader(root=root, name=(name if name else "BABI"), urls=["http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"], download=download)
        self.downloader._processed_files = self.downloader.processed_files.fuzzy_search("en-10k").ls()
