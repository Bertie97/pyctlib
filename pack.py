#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Project: PyCTLib
## File: Package packer
##############################

import sys, os
from pyinout import path
from pyctlib import vector

assert len(sys.argv) > 1
package_name = sys.argv[1]

os.system(f"python3 setup_{package_name}.py sdist --dist-dir dist_{package_name}")
os.system(f"twine upload {str(vector(path('dist_' + package_name)).max())}")
