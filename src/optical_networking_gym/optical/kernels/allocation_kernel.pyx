from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp


def candidate_starts_array(cnp.ndarray available_slots, int required_slots):
    cdef cnp.ndarray[cnp.npy_bool, ndim=1] free_slots
    cdef Py_ssize_t total_slots
    cdef Py_ssize_t slot_index
    cdef Py_ssize_t run_start = -1
    cdef Py_ssize_t block_end
    cdef Py_ssize_t block_length
    cdef Py_ssize_t min_block_length
    cdef Py_ssize_t last_candidate
    cdef Py_ssize_t candidate_index
    cdef int* buffer
    cdef Py_ssize_t candidate_count = 0
    cdef cnp.ndarray[cnp.int32_t, ndim=1] result

    if required_slots <= 0:
        raise ValueError("required_slots must be positive")

    free_slots = np.asarray(available_slots, dtype=np.bool_)
    if free_slots.ndim != 1:
        raise ValueError("available_slots must be a 1D array")

    total_slots = free_slots.shape[0]
    buffer = <int*>malloc(total_slots * sizeof(int))
    if buffer == NULL:
        raise MemoryError()

    try:
        for slot_index in range(total_slots):
            if free_slots[slot_index]:
                if run_start < 0:
                    run_start = slot_index
                continue

            if run_start >= 0:
                block_end = slot_index
                block_length = block_end - run_start
                min_block_length = required_slots + 1
                if block_length >= min_block_length:
                    last_candidate = block_end - min_block_length
                    for candidate_index in range(run_start, last_candidate + 1):
                        buffer[candidate_count] = <int>candidate_index
                        candidate_count += 1
                run_start = -1

        if run_start >= 0:
            block_end = total_slots
            block_length = block_end - run_start
            min_block_length = required_slots
            if block_length >= min_block_length:
                last_candidate = block_end - min_block_length
                for candidate_index in range(run_start, last_candidate + 1):
                    buffer[candidate_count] = <int>candidate_index
                    candidate_count += 1

        result = np.empty(candidate_count, dtype=np.int32)
        for slot_index in range(candidate_count):
            result[slot_index] = buffer[slot_index]
        return result
    finally:
        free(buffer)


def block_is_free(
    cnp.ndarray[cnp.int32_t, ndim=2] slot_allocation,
    cnp.ndarray[cnp.intp_t, ndim=1] link_indices,
    int slot_start,
    int slot_end_exclusive,
):
    cdef Py_ssize_t link_pos
    cdef Py_ssize_t slot_index
    cdef Py_ssize_t link_index

    for link_pos in range(link_indices.shape[0]):
        link_index = link_indices[link_pos]
        for slot_index in range(slot_start, slot_end_exclusive):
            if slot_allocation[link_index, slot_index] != -1:
                return False
    return True


def fill_range(
    cnp.ndarray[cnp.int32_t, ndim=2] slot_allocation,
    cnp.ndarray[cnp.intp_t, ndim=1] link_indices,
    int slot_start,
    int slot_end_exclusive,
    int value,
):
    cdef Py_ssize_t link_pos
    cdef Py_ssize_t slot_index
    cdef Py_ssize_t link_index

    for link_pos in range(link_indices.shape[0]):
        link_index = link_indices[link_pos]
        for slot_index in range(slot_start, slot_end_exclusive):
            slot_allocation[link_index, slot_index] = value
