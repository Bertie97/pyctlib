from tpdataset import RawDataSet
import tarfile

class BABI(RawDataSet):

    def __init__(self, root="", name="", download=False):
        urls = ["http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"]
        super().__init__(root=root, name=name, urls=urls, download=download)

        RawDataSet.untar(self.raw_files[0], self.root / self.processed_folder)

