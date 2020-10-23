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
    name = "pyctlib",
    version = "0.1.0",
    keywords = ("pip", "pyctlib"),
    description = "This is A foundamental package containing some basic self-designed functions and types for Project PyCTLib. ",
    long_description = "This is A foundamental package containing some basic self-designed functions and types for Project PyCTLib. ",
    # long_description = open("./pyctlib/README.md").read(),
    # long_description_content_type="text/markdown",
    license = "MIT Licence",
 
    url = "https://github.com/Bertie97/pyctlib/tree/main/pyctlib",
    author = "All contributors of PyCTLib",
    author_email = "2247501256@qq.com",
 
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["pyoverload"]
)
