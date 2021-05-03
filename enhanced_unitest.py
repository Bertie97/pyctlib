import sys
import os
from sys import getsizeof

sys.path.append(os.path.abspath("."))
import pyctlib
import pathlib
import numpy as np
from pyctlib import vector, IndexMapping, scope, vhelp
from pyctlib.vector import chain_function
from fuzzywuzzy import fuzz
from pyctlib import path, get_relative_path, file
from pyctlib import touch
from pyctlib.wrapper import generate_typehint_wrapper

class L:

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 10

    def __iter__(self):
        for index in range(10):
            yield index

vhelp(L(), enhanced=True, only_content=True)
