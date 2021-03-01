'''
@File       :   setup.py
@Author     :   Yuncheng Zhou & Yiteng Zhang
@Time       :   2020-10
@Version    :   1.0
@Contact    :   2247501256@qq.com
@Dect       :   None
'''
 
from setuptools import setup, find_packages
 
setup(
    name = "torchplus",
    version = "0.2.16",
    keywords = ("pip", "pyctlib", "torchplus"),
    description = "This package is based on pytorch and try to provide a more user-friendly interface for pytorch",
    long_description = open("./torchplus/README.md").read(),
    long_description_content_type="text/markdown",
    license = "MIT Licence",
 
    url = "https://github.com/Bertie97/pyctlib/tree/main/torchplus",
    author = "Zhang Yiteng & Zhou Yuncheng",
    author_email = "zytfdu@icloud.com",
 
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["numpy", "pyctlib", "torch>=1.7.0", "pynvml"]
)
