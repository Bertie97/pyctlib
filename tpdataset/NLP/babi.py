from tpdataset import RawDataSet, DataDownloader
from pyctlib import vector
import tarfile
import re

class BABI:

    def __init__(self, root="", name="", download=True):
        self.downloader = DataDownloader(root=root, name=(name if name else "BABI"), urls=["http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"], download=download)
        self.downloader._processed_files = self.downloader.processed_files.fuzzy_search("en-10k")[-1].ls()
        self.rawdata = vector()
        priority = {"train": 0, "test": 1}
        for index in range(20):
            files = self.downloader.processed_files.filter(lambda x: "qa" + str(index+1) == x.name.partition("_")[0])
            files = files.sort(lambda x: priority[x.name.rpartition("_")[2]])
            content = files.map(lambda x: x.readlines().vector())
            self.rawdata.append(RawDataSet(content, name=files[0].name.rpartition("_")[0]))
        self.id = dict(self.downloader.processed_files.map(lambda x: x.name).filter(lambda x: "train" in x).map(lambda x: x.rpartition("_")[0]).map(lambda x: tuple(x.split("_"))).sort(lambda x: int(x[0][2:])))

    def __getitem__(self, item):
        if not isinstance(item, str) or not re.match("qa\d{1,2}", item):
            print(self.id)
            print("usage: babi[qa{index}]")
        return self.rawdata[int(item[2:])]

    def __dir__(self):
        ret = super().__dir__()
        for index in range(1, 21):
            ret.append("qa{}".format(index))
        return ret
