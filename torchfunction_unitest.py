import sys
import os
from sys import getsizeof

sys.path.append(os.path.abspath("."))
import pyctlib
import pathlib
import numpy as np
from pyctlib import vector, IndexMapping, scope
from pyctlib.vector import chain_function
from pyctlib import path, get_relative_path, file, vhelp
from pyctlib import touch
from pyctlib.wrapper import generate_typehint_wrapper
import inspect
import torch
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz
import pydoc
from pyctlib import Logger
from torchfunction.data import RandomDataset, DataLoader

t = RandomDataset(100, [2], dtypes=[int])
dt = DataLoader(t, 10, shuffle=False)
small_dt = dt.partial_dataset(0.2)
