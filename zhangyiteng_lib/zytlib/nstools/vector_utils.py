from ..container.vector import vector
from typing import Tuple, List, Optional

def k_fold(x: list, k: int, i: int) -> Tuple[List, List]:
    assert k > 0
    len_x = len(x)
    d_len_x = len_x // k
    train_x = x[:d_len_x * i] + x[d_len_x * (i + 1):]
    test_x = x[d_len_x * i: d_len_x * (i + 1)]
    return train_x, test_x
