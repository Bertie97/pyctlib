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
    name = "pyctlib",
<<<<<<< HEAD
    version = "0.3.27",
=======
    version = "0.3.28",
>>>>>>> 07db65c2c889e7d4226d945be3e4d8611f49cfb5
    keywords = ("pip", "pyctlib"),
    description = "This is A foundamental package containing some basic self-designed functions and types for Project PyCTLib. ",
    long_description = open("./pyctlib/README.md").read(),
    long_description_content_type="text/markdown",
    license = "MIT Licence",
 
    url = "https://github.com/Bertie97/pyctlib/tree/main/pyctlib",
    author = "All contributors of PyCTLib",
    author_email = "2247501256@qq.com",
 
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["pyoverload", "rapidfuzz", "numpy"]
)
