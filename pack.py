#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Project: PyCTLib
## File: Package packer
##############################

import sys, os, re
from pyctlib import path, vector

assert len(sys.argv) > 1
package_name = sys.argv[1]

with open(f"setup_{package_name}.py") as fp:
    file_str = fp.read()
    lines = []
    for line in file_str.split('\n'):
        if line.strip().startswith('#'): continue
        for match in re.findall(r'open\(.+\).read\(\)', line):
            line = line.replace(match, '"""' + eval(match) + '"""')
        lines.append(line)
    with open("setup.py", 'w') as outfp:
        outfp.write('\n'.join(lines))

ppath = path('.')/"packing_package"
if ppath.exists(): os.system(f"rm -r {ppath}")
os.system(f"cp setup.py {ppath.mkdir()/'setup.py'}")
os.system(f"cp -r {package_name} packing_package/{package_name}")
os.system(f"cd packing_package; python3 setup.py sdist --dist-dir dist_{package_name}")
os.system(f"cd packing_package; twine upload {str(vector(ppath/('dist_' + package_name)).max()[-2:])}")
os.system("rm setup.py")
os.system("rm -r packing_package")
