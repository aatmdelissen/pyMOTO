from typing import Any
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
    elif isinstance(var_in, list):
        return var_in
    elif isinstance(var_in, tuple) or isinstance(var_in, set):
        return list(var_in)
    else:
        return [var_in]


def _concatenate_to_array(var_list: list):
    values = np.array([])
    cumulative_inds = np.zeros(len(var_list)+1, dtype=int)

    for i, v in enumerate(var_list):
        if v is None:
            raise ValueError("Trying to add None to the array")

        values = np.append(values, v)
        cumulative_inds[i+1] = len(values)

    return values, cumulative_inds


def _split_from_array(values: np.ndarray, cumulative_inds: np.ndarray):
    assert cumulative_inds[-1] == values.size, "Size of the array does not match the indices"
    var_list = list()
    for i in range(cumulative_inds.size-1):
        var_list.append(values[cumulative_inds[i]:cumulative_inds[i+1]])
    return var_list
