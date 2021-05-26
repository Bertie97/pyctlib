import sys
import os
from sys import getsizeof

sys.path.append(os.path.abspath("."))
import pyctlib
import pathlib
import numpy as np
from pyctlib import vector, IndexMapping, scope
from pyctlib.vector import chain_function
with scope("import"):
    from pyctlib import path, get_relative_path, file, vhelp
    from pyctlib import touch
    from pyctlib.wrapper import generate_typehint_wrapper
from tpdataset import RawDataSet
from tpdataset.NLP.babi import BABI
from tpdataset import MNIST, CelebA
import inspect
import torch
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz
import pydoc
from pyctlib import Logger

# babi = BABI(download=True)
# vhelp(babi, prefix="babi", enhanced=True)
# vhelp()

mnist = MNIST(root="..")
celeba = CelebA(root="..", download=False)
