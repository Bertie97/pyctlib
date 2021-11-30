'''
@File       :   setup.py
@Author     :   Yiteng Zhang
@Time       :   2020-10
@Version    :   1.0
@Contact    :   zytfdu@icloud.com
@Dect:   None
'''
 
from setuptools import setup, find_packages
 
setup(
    name = "zytlib",
    version = "0.3.218",
    keywords = ("pip", "zytlib"),
    description = "This is A foundamental package containing some basic self-designed functions and types for Project PyCTLib. ",
    long_description = open("./pyctlib/README.md").read(),
    long_description_content_type="text/markdown",
    license = "MIT Licence",

    url = "https://github.com/Bertie97/pyctlib/tree/main/pyctlib",
    author = "All contributors of PyCTLib",
    author_email = "zytfdu@icloud.com",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["pyoverload", "rapidfuzz", "numpy", "wrapt_timeout_decorator", "notion"]
)
