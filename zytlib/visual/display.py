#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package visual
##############################

__all__ = """
    display
""".split()

from ..touch import touch, get_environ_vars, SPrint

def display(content=None, name='', format = '%.4f', **kwargs):
    output = SPrint()
    if content is None:
        assert len(kwargs) > 0
        for k, v in kwargs.items(): display(v, k)
        return
    elif isinstance(content, str):
        name = content
        local_vars = get_environ_vars()
        local_vars.update(locals())
        locals().update(local_vars.simplify())
        try:
            input_array = eval(name)
        except NameError:
            output(f"{name}: Unavailable data!")
    else:
        input_array = content
        if not name: name = "[unknown array]"

    nan = float('nan')
    np = touch(lambda: __import__("numpy"))
    type_str = getattr(input_array, 'type', lambda: str(type(input_array)).split("'")[1])()
    size_str = repr(list(getattr(input_array, 'shape', len(input_array))))
    if hasattr(input_array, 'numpy'): numpy_array = input_array.numpy()
    elif np: numpy_array = np.array(input_array)
    else: numpy_array = input_array
    dtype_str = str(getattr(numpy_array, 'dtype', ''))
    if "'" in dtype_str: dtype_str = dtype_str.split("'")[1]
    output(f"{name}: {type_str}({dtype_str}){size_str}")
    min_str = format%getattr(numpy_array, 'min', lambda: nan)()
    max_str = format%getattr(numpy_array, 'max', lambda: nan)()
    output(f"range: ({min_str}, {max_str})")
    if np: unique_values = set(np.unique(numpy_array).tolist())
    else:
        flattened = getattr(numpy_array, 'flatten', lambda: numpy_array)()
        unique_values = getattr(flattened, 'tolist', lambda: None)()
        if unique_values: unique_values = set(unique_values)
    if unique_values:
        if len(unique_values) > 100:
            output(f"{len(unique_values)} values. ")
        else:
            output(f"{len(unique_values)} values: {{{', '.join([format%v for v in sorted(unique_values)])}}}")
    print(output.text)
