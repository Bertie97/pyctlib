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
from pyctlib.filemanager import path, get_relative_path, file
from pyctlib import touch
from pyctlib.wrapper import generate_typehint_wrapper
import argparse
from time import sleep
from pyctlib import totuple
from pyctlib.touch import once
from pyctlib import Logger
import math

logger = Logger(True, True)

for step in range(100):

    logger.variable("train.loss", step ** 0.5)
    logger.variable("val.loss", step ** 0.5)
    logger.variable("loss[train]", step ** 0.5)
    logger.variable("loss[val]", step ** 0.5)
    logger.variable("train.x", step ** 0.5)
    logger.variable("train.y", step ** 0.5)
    logger.variable("train.z", step ** 0.5)
    logger.variable("train.w[a]", 1)
    logger.variable("train.w[b]", 1)
    logger.variable("train.w[c]", 1)
    logger.variable("train.w[d]", 1)

Logger.plot_variable_dict(logger.variable_dict)
