import cython
cimport numpy as cnp
cnp.import_array()
import numpy as np

@cython.wraparound(True)
def rle(cnp.ndarray[cnp.int32_t, ndim=1] arr):
    # cnp.int32_t[:, :] spectrum_use):
    y = np.array(arr[1:] != arr[:-1])  # pairwise unequal (string safe)
    i = np.append(np.where(y), 1)  # must include last element posi
    z = np.diff(np.append(-1, i))  # run lengths
    p = np.cumsum(np.append(0, z))[:-1]  # positions
    return p, arr[i], z
