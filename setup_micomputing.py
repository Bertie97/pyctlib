'''
@File       :   setup.py
@Author     :   Yuncheng Zhou
@Time       :   2021-03
@Version    :   1.0
@Contact    :   bertiezhou@163.com
@Dect       :   None
'''
 
from setuptools import setup, find_packages
 
setup(
    name = "micomputing",
    version = "0.0.2",
    keywords = ("pip", "pyctlib", "torchplus", "micomputing"),
    description = "This package is based on torchplus and provides medical image computations. ",
    long_description = open("./micomputing/README.md").read(),
    long_description_content_type="text/markdown",
    license = "MIT Licence",
 
    url = "https://github.com/Bertie97/pyctlib/tree/main/micomputing",
    author = "Zhou Yuncheng",
    author_email = "bertiezhou@163.com",
 
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["numpy", "pyctlib", "torch>=1.7.0", "pynvml", "torchplus", "nibabel", "pydicom"]
)
