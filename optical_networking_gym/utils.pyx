import cython
cimport numpy as cnp
cnp.import_array() 
import numpy as np

def rle(cnp.ndarray[cnp.int32_t, ndim=1] array):
    cdef Py_ssize_t n = array.shape[0]
    cdef list initial_indices = []
    cdef list values = []
    cdef list lengths = []

    if n == 0:
        return (
            np.array(initial_indices, dtype=np.int32),
            np.array(values, dtype=np.int32),
            np.array(lengths, dtype=np.int32)
        )

    cdef int current_value = array[0]
    cdef Py_ssize_t start = 0

    for i in range(1, n):
        if array[i] != current_value:
            initial_indices.append(start)
            values.append(current_value)
            lengths.append(i - start)
            start = i
            current_value = array[i]

    # Adiciona o último run
    initial_indices.append(start)
    values.append(current_value)
    lengths.append(n - start)

    # Converte listas para arrays NumPy tipados
    return (
        np.array(initial_indices, dtype=np.int32),
        np.array(values, dtype=np.int32),
        np.array(lengths, dtype=np.int32)
    )

