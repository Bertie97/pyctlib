__all__ = """
    get_setting
    set_setting
""".split()

params = dict()

params["basic_torch"] = False

def get_setting(key):
    global params
    return params.get(key, None)

def set_setting(key, value):
    global params
    params[key] = value
