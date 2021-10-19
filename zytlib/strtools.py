def delete_surround(str_obj: str, left, right):
    assert str_obj.startswith(left)
    assert str_obj.endswith(right)
    return str_obj[len(left):-len(right)]
