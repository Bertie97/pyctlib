from pyctlib import vector, scope
import math

with open("sherlock.txt", "r") as input:
    content = input.readlines()

with scope("test vector"):
    vcontent = vector(content)
    print(math.sqrt(vcontent.map(lambda x: x.split()).map(len).map(lambda x: x**2).sum()))

with scope("test vector"):
    vcontent = vector(content)
    print(math.sqrt(vcontent.map(lambda x: x.split(), len, lambda x: x**2).sum()))

with scope("test ctgenerator"):
    vcontent = vector(content).generator()
    print(math.sqrt(vcontent.map(lambda x: x.split()).map(len).map(lambda x: x**2).sum()))

with scope("test list comprehension"):
    print(math.sqrt(sum([x ** 2 for x in [len(y) for y in [t.split() for t in content]]])))
