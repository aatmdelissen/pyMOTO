from typing import Any, Iterable
import numpy as np


def _parse_to_list(*args: Any):
    if len(args) == 0:
        return []
    elif len(args) == 1:
        var_in = args[0]
    else:
        var_in = args

    if var_in is None:
        return []
    elif isinstance(var_in, np.ndarray):
        return [var_in]
    elif isinstance(var_in, Iterable):
        return list(var_in)
    else:
        return [var_in]