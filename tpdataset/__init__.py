# This packge contain common dataset used by Machine Learning which provide simple and consistent experience

from .raw_dataset import RawDataSet, DataDownloader
from . import NLP
from .CV import MNIST, CelebA
from .device import AutoDeviceId, AutoDevice
