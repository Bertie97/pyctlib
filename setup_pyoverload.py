'''
@File       :   setup_pyoverload.py
@Author     :   Yuncheng Zhou & Yiteng Zhang
@Time       :   2020-10
@Version    :   1.0
@Contact    :   2247501256@qq.com
@Dect       :   None
'''

import re
from setuptools import setup, find_packages

def wrap_code_fences(file_str):
    for code in re.findall(r"```[^(```)]+```", file_str):
        file_str = file_str.replace(code, '\n'.join(['\t' + l for l in code.split('\n') if '```' not in l]))
    return file_str
 
setup(
    name = "pyoverload",
    version = "0.5.3",
    keywords = ("pip", "pyctlib", "overload"),
    description = "'pyoverload' overloads the functions by simply using typehints and adding decorator '@overload'.",
    long_description = wrap_code_fences(open("./pyoverload/README.md").read()),
    license = "MIT Licence",
 
    url = "https://github.com/Bertie97/pyctlib/tree/main/pyoverload",
    author = "Yuncheng Zhou, Yiteng Zhang",
    author_email = "2247501256@qq.com",
 
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = []
)
