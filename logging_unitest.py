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

variable_dict = Logger.variable_from_logging_file("log_2021-0624-16.log")
Logger.plot_variable_dict(variable_dict, saved_path="log_2021-0624-16.pdf")
