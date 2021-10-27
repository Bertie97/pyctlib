import sys
import os
from sys import getsizeof

sys.path.append(os.path.abspath("."))
import pyctlib
from zytlib.filemanager import path

p = path(".")
t = p / "Log" / "test.log"

