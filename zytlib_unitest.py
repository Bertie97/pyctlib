import sys
import os
from sys import getsizeof

sys.path.append(os.path.abspath("."))
import zytlib
from zytlib.table import table
import numpy as np
from zytlib import vector, IndexMapping, scope, vhelp
from zytlib.vector import chain_function
from zytlib.filemanager import path, get_relative_path, file
from zytlib import touch
from zytlib.wrapper import generate_typehint_wrapper
import argparse
from time import sleep
from zytlib.utils import totuple
from zytlib.touch import once
import seaborn as sns
import matplotlib.pyplot as plt
from zytlib.visual.animation import TimeStamp
from zytlib.wrapper import repeat_trigger
from zytlib.wrapper_plus import advanced_data
from zytlib.sequence import sequence
import pickle
import time
from zytlib import vector, sequence

@advanced_data
def f(*args, **kwargs):
    for _ in args:
        print(_, type(_))

    for key, value in kwargs:
        print(value, type(value))

# print(vector.range(3).map_async(lambda x: x+1))
# from zytlib.wrapper import FunctionTimer

# timer = FunctionTimer()
# fast_timer = FunctionTimer(fast_threshold=1000)

# class A:

#     def __init__(self, timer):
#         self.t = timer
#         self.func = self.t.timer(self.func)

#     def func(self, x):
#         return x

# @timer.timer
# def lazy(x):
#     return x

# @fast_timer.timer
# def fast_lazy(x):
#     return x

# def fast(x):
#     return x

# with scope("lazy"):
#     for i in range(1000000):
#         lazy(i)

# with scope("fast_lazy"):
#     for i in range(1000000):
#         fast_lazy(i)

# with scope("fast"):
#     for i in range(1000000):
#         fast(i)


