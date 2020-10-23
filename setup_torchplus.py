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
    version = "0.1.0",
    keywords = ("pip", "pyctlib", "torchplus"),
    description = "This package is based on pytorch and try to provide a more user-friendly interface for pytorch",
    long_description = "We encapsulated a new type on top of torch.Tenser, which we also call it Tensor. It has the same function as torch.Tensor, but it can change to cuda device automatically. Also, we try to provide more useful module for torch users to make deep learning earier to be implemented.",
    # long_description = open("./torchplus/README.md").read(),
    # long_description_content_type="text/markdown",
    license = "MIT Licence",
 
    url = "https://github.com/Bertie97/pyctlib/tree/main/torchplus",
    author = "Zhang Yiteng & Zhou Yuncheng",
    author_email = "zytfdu@icloud.com",
 
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["numpy", "pyctlib", "torch"]
)
