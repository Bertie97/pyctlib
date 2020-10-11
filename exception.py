def touch(f):
    try:
        return f()
    except:
        return False

def test(f):
    try: f()
    except: return False
    return True
