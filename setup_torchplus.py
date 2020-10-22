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
    keywords = ("pip", "pyctlib", "overload"),
    description = "模块描述",
    long_description = "模块详细描述",
    license = "MIT Licence",
 
    url = "https://github.com/jiangfubang/balabala",       # 项目相关文件地址，一般是github，有没有都行吧
    author = "Zhang Yiteng & Zhou Yuncheng",
    author_email = "zytfdu@icloud.com",
 
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["numpy"]        # 该模块需要的第三方库
)
