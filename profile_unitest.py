import sys
import os
sys.path.append(os.path.abspath("."))
from pyctlib import vector, vhelp
from pyctlib import visual
from pyctlib import profile

@profile
def gcd(a, b):
    if a > b:
        return gcd(b, a)
    if a == 0:
        return b
    return gcd(b % a, a)
