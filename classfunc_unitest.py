import sys
import os
from sys import getsizeof

sys.path.append(os.path.abspath("."))
import pyctlib
from pyctlib.classfunc import save_args

class A:

    def __init__(self, a, b):
        save_args(vars())

a = A(1, 2)
