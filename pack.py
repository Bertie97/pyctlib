#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Project: PyCTLib
## File: Package packer
##############################

import sys, os, re
from pyctlib import path, vector

assert len(sys.argv) > 1
package_names = sys.argv[1:]

if 'all' in package_names:
    package_names = []
    for path in os.listdir('.'):
        if path.startswith('setup_') and path.endswith('.py'):
            package_names.append(path[6:-3])

for p in package_names:
    if '==' in p: package_name, version = p.split('==')
    else: package_name = p; version = None
    version_match = None
    with open(f"setup_{package_name}.py") as fp:
        file_str = fp.read()
        lines = []
        for line in file_str.split('\n'):
            if line.strip().startswith('#'): continue
            for match in re.findall(r'open\(.+\).read\(\)', line):
                line = line.replace(match, '"""' + eval(match) + '"""')
            for match in re.findall(r'version *= *"[\d.]+"', line):
                v = match.split('"')[1]
                if version is None: version = '.'.join(v.split('.')[:-1] + [str(eval(v.split('.')[-1]) + 1)])
                version = f'version = "{version}"'
                line = line.replace(match, version)
                version_match = match
            lines.append(line)
        with open("setup.py", 'w') as outfp:
            outfp.write('\n'.join(lines))

    with open(f"setup_{package_name}.py", 'w') as fp:
        if version_match is not None: fp.write(file_str.replace(version_match, version))

    ppath = path('.')/"packing_package"
    if ppath.exists(): os.system(f"rm -r {ppath}")
    os.system(f"cp setup.py {ppath.mkdir()/'setup.py'}")
    os.system(f"cp -r {package_name} packing_package/{package_name}")
    os.system(f"cd packing_package; python3 setup.py sdist --dist-dir dist_{package_name}")
    os.system(f"cd packing_package; twine upload {str(vector(ppath/('dist_' + package_name)).max()[-2:])}")
    os.system("rm setup.py")
    os.system("rm -r packing_package")
