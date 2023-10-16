import numpy as np


def oc_update(x, dfdx, xmin):
    l1, l2, move, maxvol = 0, 100000, 0.2, np.sum(x)
    while l2 - l1 > 1e-4:
        lmid = 0.5 * (l1 + l2)
        xnew = np.maximum(xmin, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x*np.sqrt(-dfdx/lmid)))))
        l1, l2 = (lmid, l2) if np.sum(xnew) - maxvol > 0 else (l1, lmid)
    return xnew, np.max(np.abs(xnew - x))
