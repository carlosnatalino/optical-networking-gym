from __future__ import annotations

cimport numpy as cnp
from libc.math cimport log, sqrt
import numpy as np

cnp.import_array()


cdef inline double _clamp_unit(double value) noexcept:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


cdef inline double _length_log_length(double length) noexcept:
    if length <= 0.0:
        return 0.0
    return length * log(length)


def analyze_free_mask_kernel(object free_mask_obj):
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] free_values = np.asarray(free_mask_obj, dtype=np.uint8)
    cdef Py_ssize_t total_slots = free_values.shape[0]
    cdef cnp.ndarray[cnp.int32_t, ndim=1] slot_to_run_index = np.full(total_slots, -1, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] empty = np.empty(0, dtype=np.int32)
    cdef Py_ssize_t max_runs = (total_slots + 1) // 2 if total_slots > 0 else 0
    cdef cnp.ndarray[cnp.int32_t, ndim=1] run_starts_buf
    cdef cnp.ndarray[cnp.int32_t, ndim=1] run_ends_buf
    cdef cnp.ndarray[cnp.int32_t, ndim=1] run_lengths_buf
    cdef cnp.ndarray[cnp.int32_t, ndim=1] largest_other
    cdef cnp.ndarray[cnp.int32_t, ndim=1] prefix
    cdef cnp.ndarray[cnp.int32_t, ndim=1] suffix
    cdef cnp.uint8_t[:] free_view = free_values
    cdef cnp.int32_t[:] slot_to_run_view = slot_to_run_index
    cdef cnp.int32_t[:] run_starts_view
    cdef cnp.int32_t[:] run_ends_view
    cdef cnp.int32_t[:] run_lengths_view
    cdef cnp.int32_t[:] largest_other_view
    cdef cnp.int32_t[:] prefix_view
    cdef cnp.int32_t[:] suffix_view
    cdef Py_ssize_t slot_index
    cdef int run_count = 0
    cdef int current_run_start = -1
    cdef int run_length
    cdef int fill_index
    cdef int total_free = 0
    cdef int largest = 0
    cdef int current_max = 0
    cdef double sum_squares = 0.0
    cdef double sum_length_log_length = 0.0
    cdef double entropy = 0.0
    cdef double rss = 0.0
    cdef double value

    if total_slots == 0:
        return (0, 0, 0, 0.0, 0.0, empty, empty, empty, slot_to_run_index, empty, 0.0, 0.0)

    run_starts_buf = np.empty(max_runs, dtype=np.int32)
    run_ends_buf = np.empty(max_runs, dtype=np.int32)
    run_lengths_buf = np.empty(max_runs, dtype=np.int32)
    run_starts_view = run_starts_buf
    run_ends_view = run_ends_buf
    run_lengths_view = run_lengths_buf

    for slot_index in range(total_slots):
        if free_view[slot_index] != 0:
            if current_run_start < 0:
                current_run_start = <int>slot_index
            continue
        if current_run_start < 0:
            continue
        run_length = <int>slot_index - current_run_start
        run_starts_view[run_count] = current_run_start
        run_ends_view[run_count] = <int>slot_index
        run_lengths_view[run_count] = run_length
        for fill_index in range(current_run_start, <int>slot_index):
            slot_to_run_view[fill_index] = run_count
        total_free += run_length
        if run_length > largest:
            largest = run_length
        sum_squares += run_length * run_length
        sum_length_log_length += _length_log_length(run_length)
        run_count += 1
        current_run_start = -1

    if current_run_start >= 0:
        run_length = <int>total_slots - current_run_start
        run_starts_view[run_count] = current_run_start
        run_ends_view[run_count] = <int>total_slots
        run_lengths_view[run_count] = run_length
        for fill_index in range(current_run_start, <int>total_slots):
            slot_to_run_view[fill_index] = run_count
        total_free += run_length
        if run_length > largest:
            largest = run_length
        sum_squares += run_length * run_length
        sum_length_log_length += _length_log_length(run_length)
        run_count += 1

    if run_count == 0:
        return (0, 0, 0, 0.0, 0.0, empty, empty, empty, slot_to_run_index, empty, 0.0, 0.0)

    largest_other = np.zeros(run_count, dtype=np.int32)
    prefix = np.zeros(run_count, dtype=np.int32)
    suffix = np.zeros(run_count, dtype=np.int32)
    largest_other_view = largest_other
    prefix_view = prefix
    suffix_view = suffix

    current_max = 0
    for slot_index in range(run_count):
        if run_lengths_view[slot_index] > current_max:
            current_max = run_lengths_view[slot_index]
        prefix_view[slot_index] = current_max

    current_max = 0
    for slot_index in range(run_count - 1, -1, -1):
        if run_lengths_view[slot_index] > current_max:
            current_max = run_lengths_view[slot_index]
        suffix_view[slot_index] = current_max

    for slot_index in range(run_count):
        largest_other_view[slot_index] = max(
            prefix_view[slot_index - 1] if slot_index > 0 else 0,
            suffix_view[slot_index + 1] if slot_index + 1 < run_count else 0,
        )

    if run_count > 1 and total_free > 0:
        entropy = _clamp_unit((log(total_free) - (sum_length_log_length / total_free)) / log(run_count))
    if total_free > 0:
        rss = _clamp_unit(sqrt(sum_squares) / total_free)

    return (
        run_count,
        largest,
        total_free,
        entropy,
        rss,
        np.asarray(run_starts_buf[:run_count], dtype=np.int32).copy(),
        np.asarray(run_ends_buf[:run_count], dtype=np.int32).copy(),
        np.asarray(run_lengths_buf[:run_count], dtype=np.int32).copy(),
        slot_to_run_index,
        largest_other,
        sum_squares,
        sum_length_log_length,
    )


def summary_after_allocation_kernel(
    object slot_to_run_index_obj,
    object run_starts_obj,
    object run_ends_obj,
    object run_lengths_obj,
    object largest_other_by_run_obj,
    double sum_squares,
    double sum_length_log_length,
    int summary_count,
    int summary_largest,
    int summary_total_free,
    int service_slot_start,
    int service_num_slots,
    int total_slots,
):
    cdef cnp.ndarray[cnp.int32_t, ndim=1] slot_to_run_index = np.asarray(slot_to_run_index_obj, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] run_starts = np.asarray(run_starts_obj, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] run_ends = np.asarray(run_ends_obj, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] run_lengths = np.asarray(run_lengths_obj, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] largest_other_by_run = np.asarray(largest_other_by_run_obj, dtype=np.int32)
    cdef cnp.int32_t[:] slot_to_run_view = slot_to_run_index
    cdef cnp.int32_t[:] run_starts_view = run_starts
    cdef cnp.int32_t[:] run_ends_view = run_ends
    cdef cnp.int32_t[:] run_lengths_view = run_lengths
    cdef cnp.int32_t[:] largest_other_view = largest_other_by_run
    cdef int run_index = slot_to_run_view[service_slot_start]
    cdef int run_start
    cdef int run_end
    cdef int removed_end
    cdef int left_length
    cdef int right_length
    cdef int removed_length
    cdef int post_total_free
    cdef int post_count
    cdef int post_largest
    cdef double post_sum_squares
    cdef double post_sum_length_log_length
    cdef double post_entropy = 0.0
    cdef double post_rss = 0.0

    if run_index < 0:
        return (summary_count, summary_largest, summary_total_free, 0.0, 0.0)

    run_start = run_starts_view[run_index]
    run_end = run_ends_view[run_index]
    removed_end = service_slot_start + service_num_slots
    if removed_end < total_slots:
        removed_end += 1

    left_length = service_slot_start - run_start
    if left_length < 0:
        left_length = 0
    right_length = run_end - removed_end
    if right_length < 0:
        right_length = 0
    removed_length = run_lengths_view[run_index] - left_length - right_length
    post_total_free = summary_total_free - removed_length
    if post_total_free < 0:
        post_total_free = 0
    post_count = summary_count - 1 + (1 if left_length > 0 else 0) + (1 if right_length > 0 else 0)
    post_largest = largest_other_view[run_index]
    if left_length > post_largest:
        post_largest = left_length
    if right_length > post_largest:
        post_largest = right_length

    post_sum_squares = (
        sum_squares
        - (run_lengths_view[run_index] * run_lengths_view[run_index])
        + (left_length * left_length)
        + (right_length * right_length)
    )
    if post_total_free > 0:
        post_rss = _clamp_unit(sqrt(post_sum_squares) / post_total_free)
    post_sum_length_log_length = (
        sum_length_log_length
        - _length_log_length(run_lengths_view[run_index])
        + _length_log_length(left_length)
        + _length_log_length(right_length)
    )
    if post_total_free > 0 and post_count > 1:
        post_entropy = _clamp_unit(
            (log(post_total_free) - (post_sum_length_log_length / post_total_free)) / log(post_count)
        )

    return (post_count, post_largest, post_total_free, post_entropy, post_rss)


def fragmentation_damage_by_candidates_kernel(
    object candidate_indices_obj,
    object slot_to_run_index_obj,
    object run_starts_obj,
    object run_ends_obj,
    object largest_other_by_run_obj,
    int summary_count,
    int summary_largest,
    int service_num_slots,
    int total_slots,
    float num_blocks_scale,
    float largest_block_scale,
):
    cdef cnp.ndarray[cnp.int32_t, ndim=1] candidate_indices = np.asarray(candidate_indices_obj, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] slot_to_run_index = np.asarray(slot_to_run_index_obj, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] run_starts = np.asarray(run_starts_obj, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] run_ends = np.asarray(run_ends_obj, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] largest_other_by_run = np.asarray(largest_other_by_run_obj, dtype=np.int32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] num_blocks_damage = np.zeros(candidate_indices.shape[0], dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] largest_block_damage = np.zeros(candidate_indices.shape[0], dtype=np.float32)
    cdef cnp.int32_t[:] candidate_view = candidate_indices
    cdef cnp.int32_t[:] slot_to_run_view = slot_to_run_index
    cdef cnp.int32_t[:] run_starts_view = run_starts
    cdef cnp.int32_t[:] run_ends_view = run_ends
    cdef cnp.int32_t[:] largest_other_view = largest_other_by_run
    cdef cnp.float32_t[:] num_blocks_view = num_blocks_damage
    cdef cnp.float32_t[:] largest_block_view = largest_block_damage
    cdef Py_ssize_t candidate_pos
    cdef int service_slot_start
    cdef int run_index
    cdef int run_start
    cdef int run_end
    cdef int removed_end
    cdef int left_length
    cdef int right_length
    cdef int post_count
    cdef int post_largest

    for candidate_pos in range(candidate_indices.shape[0]):
        service_slot_start = candidate_view[candidate_pos]
        run_index = slot_to_run_view[service_slot_start]
        if run_index < 0:
            continue

        run_start = run_starts_view[run_index]
        run_end = run_ends_view[run_index]
        removed_end = service_slot_start + service_num_slots
        if removed_end < total_slots:
            removed_end += 1

        left_length = service_slot_start - run_start
        if left_length < 0:
            left_length = 0
        right_length = run_end - removed_end
        if right_length < 0:
            right_length = 0

        post_count = summary_count - 1 + (1 if left_length > 0 else 0) + (1 if right_length > 0 else 0)
        post_largest = largest_other_view[run_index]
        if left_length > post_largest:
            post_largest = left_length
        if right_length > post_largest:
            post_largest = right_length

        if post_count > summary_count:
            num_blocks_view[candidate_pos] = (post_count - summary_count) / num_blocks_scale
        if summary_largest > post_largest:
            largest_block_view[candidate_pos] = (summary_largest - post_largest) / largest_block_scale

    return num_blocks_damage, largest_block_damage


def build_link_metrics_kernel(object slot_allocation_obj, int total_slots):
    cdef cnp.ndarray[cnp.int32_t, ndim=2] slot_allocation = np.asarray(slot_allocation_obj, dtype=np.int32)
    cdef int link_count = slot_allocation.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] metrics = np.zeros((link_count, 6), dtype=np.float32)
    cdef cnp.int32_t[:, :] slot_view = slot_allocation
    cdef cnp.float32_t[:, :] metrics_view = metrics
    cdef int max_block_count = (total_slots + 1) // 2 if total_slots > 0 else 1
    cdef int link_id
    cdef int slot_index
    cdef int occupied_slots
    cdef int first_used
    cdef int last_used
    cdef int span_width
    cdef int run_count
    cdef int largest_run
    cdef int total_free
    cdef int current_run_start
    cdef int run_length
    cdef double sum_squares
    cdef double sum_length_log_length
    cdef double entropy
    cdef double rss
    cdef double compactness

    for link_id in range(link_count):
        occupied_slots = 0
        first_used = -1
        last_used = -1
        run_count = 0
        largest_run = 0
        total_free = 0
        current_run_start = -1
        sum_squares = 0.0
        sum_length_log_length = 0.0
        entropy = 0.0
        rss = 0.0

        for slot_index in range(total_slots):
            if slot_view[link_id, slot_index] == -1:
                if current_run_start < 0:
                    current_run_start = slot_index
                continue
            occupied_slots += 1
            if first_used < 0:
                first_used = slot_index
            last_used = slot_index
            if current_run_start < 0:
                continue
            run_length = slot_index - current_run_start
            total_free += run_length
            if run_length > largest_run:
                largest_run = run_length
            sum_squares += run_length * run_length
            sum_length_log_length += _length_log_length(run_length)
            run_count += 1
            current_run_start = -1

        if current_run_start >= 0:
            run_length = total_slots - current_run_start
            total_free += run_length
            if run_length > largest_run:
                largest_run = run_length
            sum_squares += run_length * run_length
            sum_length_log_length += _length_log_length(run_length)
            run_count += 1

        if run_count > 1 and total_free > 0:
            entropy = _clamp_unit((log(total_free) - (sum_length_log_length / total_free)) / log(run_count))
        if total_free > 0:
            rss = _clamp_unit(sqrt(sum_squares) / total_free)

        if occupied_slots == 0 or occupied_slots == total_slots:
            compactness = 1.0
        else:
            span_width = (last_used - first_used) + 1
            compactness = occupied_slots / span_width if span_width > 0 else 1.0

        metrics_view[link_id, 0] = occupied_slots / total_slots if total_slots > 0 else 0.0
        metrics_view[link_id, 1] = entropy
        metrics_view[link_id, 2] = (1.0 - (largest_run / total_free)) if total_free > 0 else 0.0
        metrics_view[link_id, 3] = compactness
        metrics_view[link_id, 4] = run_count / max_block_count if max_block_count > 0 else 0.0
        metrics_view[link_id, 5] = rss

    return metrics


def build_global_features_kernel(
    object link_metrics_obj,
    float free_slots_ratio,
    float active_services_norm,
):
    cdef cnp.ndarray[cnp.float32_t, ndim=2] link_metrics = np.asarray(link_metrics_obj, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] result = np.zeros(8, dtype=np.float32)
    cdef cnp.float32_t[:, :] metrics_view = link_metrics
    cdef cnp.float32_t[:] result_view = result
    cdef int link_count = link_metrics.shape[0]
    cdef int link_index
    cdef double util_sum = 0.0
    cdef double util_sq_sum = 0.0
    cdef double util_max = 0.0
    cdef double entropy_sum = 0.0
    cdef double external_frag_sum = 0.0
    cdef double compactness_sum = 0.0
    cdef double util_mean
    cdef double util_variance

    result_view[3] = free_slots_ratio
    result_view[7] = active_services_norm
    if link_count == 0:
        return result

    util_max = metrics_view[0, 0]
    for link_index in range(link_count):
        util_sum += metrics_view[link_index, 0]
        util_sq_sum += metrics_view[link_index, 0] * metrics_view[link_index, 0]
        entropy_sum += metrics_view[link_index, 1]
        external_frag_sum += metrics_view[link_index, 2]
        compactness_sum += metrics_view[link_index, 3]
        if metrics_view[link_index, 0] > util_max:
            util_max = metrics_view[link_index, 0]

    util_mean = util_sum / link_count
    util_variance = (util_sq_sum / link_count) - (util_mean * util_mean)
    if util_variance < 0.0:
        util_variance = 0.0

    result_view[0] = util_mean
    result_view[1] = sqrt(util_variance)
    result_view[2] = util_max
    result_view[4] = entropy_sum / link_count
    result_view[5] = external_frag_sum / link_count
    result_view[6] = compactness_sum / link_count
    return result


def build_path_features_kernel(
    object common_free_masks_obj,
    object link_metrics_obj,
    object path_link_ids_obj,
    object path_link_counts_obj,
    object path_length_norms_obj,
    object path_hops_norms_obj,
    int total_slots,
    float block_count_scale,
):
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] common_free_masks = np.asarray(common_free_masks_obj, dtype=np.uint8)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] link_metrics = np.asarray(link_metrics_obj, dtype=np.float32)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] path_link_ids = np.asarray(path_link_ids_obj, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] path_link_counts = np.asarray(path_link_counts_obj, dtype=np.int32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] path_length_norms = np.asarray(path_length_norms_obj, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] path_hops_norms = np.asarray(path_hops_norms_obj, dtype=np.float32)
    cdef int path_count = common_free_masks.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = np.zeros((path_count, 13), dtype=np.float32)
    cdef cnp.uint8_t[:, :] common_view = common_free_masks
    cdef cnp.float32_t[:, :] metrics_view = link_metrics
    cdef cnp.int32_t[:, :] link_ids_view = path_link_ids
    cdef cnp.int32_t[:] link_counts_view = path_link_counts
    cdef cnp.float32_t[:] path_length_view = path_length_norms
    cdef cnp.float32_t[:] path_hops_view = path_hops_norms
    cdef cnp.float32_t[:, :] result_view = result
    cdef int path_index
    cdef int slot_index
    cdef int current_run_start
    cdef int run_length
    cdef int run_count
    cdef int largest_run
    cdef int total_free
    cdef int link_count
    cdef int link_pos
    cdef int link_id
    cdef double sum_squares
    cdef double sum_length_log_length
    cdef double entropy
    cdef double util_sum
    cdef double route_cuts_sum
    cdef double route_rss_sum
    cdef double entropy_sum
    cdef double external_frag_sum
    cdef double compactness_sum
    cdef double util_max

    for path_index in range(path_count):
        run_count = 0
        largest_run = 0
        total_free = 0
        current_run_start = -1
        sum_squares = 0.0
        sum_length_log_length = 0.0
        entropy = 0.0

        for slot_index in range(total_slots + 1):
            if slot_index < total_slots and common_view[path_index, slot_index] != 0:
                if current_run_start < 0:
                    current_run_start = slot_index
                continue
            if current_run_start < 0:
                continue
            run_length = slot_index - current_run_start
            total_free += run_length
            if run_length > largest_run:
                largest_run = run_length
            sum_squares += run_length * run_length
            sum_length_log_length += _length_log_length(run_length)
            run_count += 1
            current_run_start = -1

        if run_count > 1 and total_free > 0:
            entropy = _clamp_unit((log(total_free) - (sum_length_log_length / total_free)) / log(run_count))

        result_view[path_index, 0] = path_length_view[path_index]
        result_view[path_index, 1] = path_hops_view[path_index]
        result_view[path_index, 4] = (<double>total_free / total_slots) if total_slots > 0 else 0.0
        result_view[path_index, 5] = (<double>largest_run / total_slots) if total_slots > 0 else 0.0
        result_view[path_index, 6] = (<double>run_count / block_count_scale) if block_count_scale > 0.0 else 0.0
        result_view[path_index, 7] = entropy

        link_count = link_counts_view[path_index]
        if link_count <= 0:
            result_view[path_index, 12] = 1.0
            continue

        util_sum = 0.0
        route_cuts_sum = 0.0
        route_rss_sum = 0.0
        entropy_sum = 0.0
        external_frag_sum = 0.0
        compactness_sum = 0.0
        util_max = 0.0

        for link_pos in range(link_count):
            link_id = link_ids_view[path_index, link_pos]
            if link_id < 0:
                continue
            util_sum += metrics_view[link_id, 0]
            route_cuts_sum += metrics_view[link_id, 4]
            route_rss_sum += metrics_view[link_id, 5]
            entropy_sum += metrics_view[link_id, 1]
            external_frag_sum += metrics_view[link_id, 2]
            compactness_sum += metrics_view[link_id, 3]
            if link_pos == 0 or metrics_view[link_id, 0] > util_max:
                util_max = metrics_view[link_id, 0]

        result_view[path_index, 2] = util_sum / link_count
        result_view[path_index, 3] = util_max
        result_view[path_index, 8] = route_cuts_sum / link_count
        result_view[path_index, 9] = route_rss_sum / link_count
        result_view[path_index, 10] = entropy_sum / link_count
        result_view[path_index, 11] = external_frag_sum / link_count
        result_view[path_index, 12] = compactness_sum / link_count

    return result


def build_path_mod_features_kernel(
    object resource_valid_starts_obj,
    object qot_valid_starts_obj,
    object osnr_margin_by_start_obj,
    object nli_share_by_start_obj,
    object worst_link_nli_share_by_start_obj,
    object fragmentation_damage_num_blocks_obj,
    object fragmentation_damage_largest_block_obj,
    object required_slots_by_path_mod_obj,
):
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] resource_valid_starts = np.asarray(resource_valid_starts_obj, dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] qot_valid_starts = np.asarray(qot_valid_starts_obj, dtype=np.uint8)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] osnr_margin_by_start = np.asarray(osnr_margin_by_start_obj, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] nli_share_by_start = np.asarray(nli_share_by_start_obj, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] worst_link_nli_share_by_start = np.asarray(
        worst_link_nli_share_by_start_obj,
        dtype=np.float32,
    )
    cdef cnp.ndarray[cnp.float32_t, ndim=3] fragmentation_damage_num_blocks = np.asarray(
        fragmentation_damage_num_blocks_obj,
        dtype=np.float32,
    )
    cdef cnp.ndarray[cnp.float32_t, ndim=3] fragmentation_damage_largest_block = np.asarray(
        fragmentation_damage_largest_block_obj,
        dtype=np.float32,
    )
    cdef cnp.ndarray[cnp.int16_t, ndim=2] required_slots_by_path_mod = np.asarray(
        required_slots_by_path_mod_obj,
        dtype=np.int16,
    )
    cdef int path_count = resource_valid_starts.shape[0]
    cdef int modulation_count = resource_valid_starts.shape[1]
    cdef int total_slots = resource_valid_starts.shape[2]
    cdef cnp.ndarray[cnp.float32_t, ndim=3] result = np.zeros((path_count, modulation_count, 10), dtype=np.float32)
    cdef cnp.uint8_t[:, :, :] resource_view = resource_valid_starts
    cdef cnp.uint8_t[:, :, :] qot_view = qot_valid_starts
    cdef cnp.float32_t[:, :, :] margin_view = osnr_margin_by_start
    cdef cnp.float32_t[:, :, :] nli_view = nli_share_by_start
    cdef cnp.float32_t[:, :, :] worst_nli_view = worst_link_nli_share_by_start
    cdef cnp.float32_t[:, :, :] frag_blocks_view = fragmentation_damage_num_blocks
    cdef cnp.float32_t[:, :, :] frag_largest_view = fragmentation_damage_largest_block
    cdef cnp.int16_t[:, :] required_view = required_slots_by_path_mod
    cdef cnp.float32_t[:, :, :] result_view = result
    cdef int path_index
    cdef int modulation_offset
    cdef int slot_index
    cdef int required_slots
    cdef int denominator
    cdef int resource_count
    cdef int qot_count
    cdef int first_resource_slot
    cdef int first_qot_slot
    cdef int last_qot_slot
    cdef int best_slot
    cdef float best_margin
    cdef float margin
    cdef float best_nli_share
    cdef float best_worst_link_share
    cdef float damage_num_blocks
    cdef float damage_largest_block

    for path_index in range(path_count):
        for modulation_offset in range(modulation_count):
            required_slots = required_view[path_index, modulation_offset]
            if required_slots <= 0:
                continue

            denominator = total_slots - required_slots + 1
            if denominator < 1:
                denominator = 1
            resource_count = 0
            qot_count = 0
            first_resource_slot = -1
            first_qot_slot = -1
            last_qot_slot = -1
            best_slot = -1
            best_margin = 0.0
            best_nli_share = 0.0
            best_worst_link_share = 0.0
            damage_num_blocks = 0.0
            damage_largest_block = 0.0

            for slot_index in range(total_slots):
                if resource_view[path_index, modulation_offset, slot_index] != 0:
                    resource_count += 1
                    if first_resource_slot < 0:
                        first_resource_slot = slot_index
                if qot_view[path_index, modulation_offset, slot_index] == 0:
                    continue
                qot_count += 1
                if first_qot_slot < 0:
                    first_qot_slot = slot_index
                last_qot_slot = slot_index
                margin = margin_view[path_index, modulation_offset, slot_index]
                if margin != margin:
                    continue
                if best_slot < 0 or margin > best_margin:
                    best_slot = slot_index
                    best_margin = margin

            if best_slot >= 0:
                best_nli_share = nli_view[path_index, modulation_offset, best_slot]
                best_worst_link_share = worst_nli_view[path_index, modulation_offset, best_slot]
                damage_num_blocks = frag_blocks_view[path_index, modulation_offset, best_slot]
                damage_largest_block = frag_largest_view[path_index, modulation_offset, best_slot]
            elif first_resource_slot >= 0:
                damage_num_blocks = frag_blocks_view[path_index, modulation_offset, first_resource_slot]
                damage_largest_block = frag_largest_view[path_index, modulation_offset, first_resource_slot]

            result_view[path_index, modulation_offset, 0] = required_slots / total_slots if total_slots > 0 else 0.0
            result_view[path_index, modulation_offset, 1] = resource_count / denominator
            result_view[path_index, modulation_offset, 2] = qot_count / denominator
            result_view[path_index, modulation_offset, 3] = (
                first_qot_slot / (total_slots - 1) if first_qot_slot >= 0 and total_slots > 1 else 0.0
            )
            result_view[path_index, modulation_offset, 4] = (
                last_qot_slot / (total_slots - 1) if last_qot_slot >= 0 and total_slots > 1 else 0.0
            )
            result_view[path_index, modulation_offset, 5] = _clamp_unit(best_margin / 10.0)
            result_view[path_index, modulation_offset, 6] = best_nli_share
            result_view[path_index, modulation_offset, 7] = best_worst_link_share
            result_view[path_index, modulation_offset, 8] = _clamp_unit(damage_num_blocks)
            result_view[path_index, modulation_offset, 9] = _clamp_unit(damage_largest_block)

    return result


def build_path_slot_features_kernel(
    object common_free_masks_obj,
    object resource_valid_starts_obj,
    object qot_valid_starts_obj,
    object osnr_margin_by_start_obj,
    object nli_share_by_start_obj,
    int window=5,
):
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] common_free_masks = np.asarray(common_free_masks_obj, dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] resource_valid_starts = np.asarray(resource_valid_starts_obj, dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] qot_valid_starts = np.asarray(qot_valid_starts_obj, dtype=np.uint8)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] osnr_margin_by_start = np.asarray(osnr_margin_by_start_obj, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] nli_share_by_start = np.asarray(nli_share_by_start_obj, dtype=np.float32)
    cdef int path_count = common_free_masks.shape[0]
    cdef int total_slots = common_free_masks.shape[1]
    cdef int modulation_count = resource_valid_starts.shape[1]
    cdef cnp.ndarray[cnp.float32_t, ndim=3] result = np.zeros((path_count, total_slots, 9), dtype=np.float32)
    cdef cnp.uint8_t[:, :] common_view = common_free_masks
    cdef cnp.uint8_t[:, :, :] resource_view = resource_valid_starts
    cdef cnp.uint8_t[:, :, :] qot_view = qot_valid_starts
    cdef cnp.float32_t[:, :, :] margin_view = osnr_margin_by_start
    cdef cnp.float32_t[:, :, :] nli_view = nli_share_by_start
    cdef cnp.float32_t[:, :, :] result_view = result
    cdef int left_pad
    cdef int right_pad
    cdef int path_index
    cdef int slot_index
    cdef int neighbor_index
    cdef int modulation_offset
    cdef int current_run_start
    cdef int run_length
    cdef int local_free
    cdef int start
    cdef int end
    cdef bint has_resource_candidate
    cdef bint has_qot_candidate
    cdef bint best_margin_found
    cdef float best_margin
    cdef float best_nli
    cdef float margin

    if total_slots <= 0:
        return result
    if window < 1:
        window = 1
    if window > total_slots:
        window = total_slots
    left_pad = window // 2
    right_pad = window - left_pad - 1

    for path_index in range(path_count):
        current_run_start = -1
        for slot_index in range(total_slots + 1):
            if slot_index < total_slots and common_view[path_index, slot_index] != 0:
                if current_run_start < 0:
                    current_run_start = slot_index
                continue
            if current_run_start < 0:
                continue
            run_length = slot_index - current_run_start
            for neighbor_index in range(current_run_start, slot_index):
                result_view[path_index, neighbor_index, 1] = run_length / total_slots
                result_view[path_index, neighbor_index, 2] = (neighbor_index - current_run_start) / total_slots
                result_view[path_index, neighbor_index, 3] = (slot_index - neighbor_index - 1) / total_slots
            current_run_start = -1

        for slot_index in range(total_slots):
            result_view[path_index, slot_index, 0] = 1.0 if common_view[path_index, slot_index] != 0 else 0.0

            local_free = 0
            start = slot_index - left_pad
            end = slot_index + right_pad
            for neighbor_index in range(start, end + 1):
                if 0 <= neighbor_index < total_slots and common_view[path_index, neighbor_index] != 0:
                    local_free += 1
            result_view[path_index, slot_index, 4] = 1.0 - (local_free / window)

            has_resource_candidate = False
            has_qot_candidate = False
            best_margin_found = False
            best_margin = 0.0
            best_nli = 0.0
            for modulation_offset in range(modulation_count):
                if resource_view[path_index, modulation_offset, slot_index] != 0:
                    has_resource_candidate = True
                if qot_view[path_index, modulation_offset, slot_index] == 0:
                    continue
                has_qot_candidate = True
                margin = margin_view[path_index, modulation_offset, slot_index]
                if margin != margin:
                    continue
                if not best_margin_found or margin > best_margin:
                    best_margin_found = True
                    best_margin = margin
                    best_nli = nli_view[path_index, modulation_offset, slot_index]

            result_view[path_index, slot_index, 5] = 1.0 if has_resource_candidate else 0.0
            result_view[path_index, slot_index, 6] = 1.0 if has_qot_candidate else 0.0
            if best_margin_found:
                result_view[path_index, slot_index, 7] = _clamp_unit(best_margin / 10.0)
                result_view[path_index, slot_index, 8] = best_nli

    return result
