from pyoverload import overload

@overload
def func(a:int, b:int):
    return a+a

@overload
def func(a:float, b:float):
    return a+2*b

@overload
def func(a:str):
    return len(a)
