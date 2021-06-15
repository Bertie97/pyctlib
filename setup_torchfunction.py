'''
@File       :   setup.py
@Author     :   Yuncheng Zhou & Yiteng Zhang
@Time       :   2020-10
@Version    :   1.0
@Contact    :   bertiezhou@163.com
@Dect       :   None
'''
 
from setuptools import setup, find_packages
 
setup(
    name = "torchfunction",
    version = "0.0.18",
    keywords = ("pip", "pyctlib", "pytorch"),
    description = "This package is based on pytorch and try to provide a more user-friendly interface for pytorch. ",
    long_description = open("./torchplus/README.md").read(),
    long_description_content_type="text/markdown",
    license = "MIT Licence",
 
    url = "https://github.com/Bertie97/pyctlib/tree/main/torchplus",
    author = "Yiteng Zhang",
    author_email = "zytfdu@icloud.com",
 
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["numpy", "pyctlib", "torch>=1.7.0", "pynvml"]
)
