#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Package PyCTLib
##############################
__all__ = """
""".split()

try: from matplotlib import pyplot as plt
except ImportError:
    raise ImportError("'pyctlib.watch.debugger' cannot be used without dependency 'matplotlib'. ")

def main():
    pass

if __name__ == "__main__": main()