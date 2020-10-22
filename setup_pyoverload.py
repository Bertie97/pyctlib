'''
@File       :   setup_pyoverload.py
@Author     :   Yuncheng Zhou & Yiteng Zhang
@Time       :   2020-10
@Version    :   1.0
@Contact    :   2247501256@qq.com
@Dect       :   None
'''
 
from setuptools import setup, find_packages
 
setup(
    name = "pyoverload",
    version = "0.5.2",
    keywords = ("pip", "pyctlib", "overload"),
    description = "'pyoverload' overloads the functions by simply using typehints and adding decorator '@overload'.",
    long_description = open("./pyoverload/README.md").read(),
    license = "MIT Licence",
 
    url = "https://github.com/Bertie97/pyctlib/tree/main/pyoverload",
    author = "Yuncheng Zhou, Yiteng Zhang",
    author_email = "2247501256@qq.com",
 
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = []
)