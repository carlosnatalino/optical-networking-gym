# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True

from libc.math cimport asinh, exp, fabs, log10
import numpy as np
cimport numpy as cnp


cdef double ABS_BETA_2 = 21.3e-27
cdef double GAMMA = 1.3e-3
cdef double H_PLANCK = 6.626e-34
cdef double PI_VALUE = 3.14159265358979323846
cdef double PI_SQUARED = PI_VALUE * PI_VALUE
cdef double NLI_PREFACTOR_BASE = 8.0 / (27.0 * PI_VALUE * ABS_BETA_2)


cdef inline void _accumulate_link_noise_impl(
    const cnp.float64_t[:] span_lengths_km,
    int span_start,
    int span_end,
    const cnp.float64_t[:] span_attenuation_normalized,
    const cnp.float64_t[:] span_noise_figure_normalized,
    const cnp.int32_t[:] running_service_ids,
    int running_start,
    int running_end,
    const cnp.float64_t[:] running_center_frequencies,
    const cnp.float64_t[:] running_bandwidths,
    const cnp.float64_t[:] running_phi_modulation,
    int current_service_id,
    double center_frequency,
    double bandwidth,
    double launch_power,
    double nli_prefactor,
    bint include_nli,
    double* out_acc_gsnr,
    double* out_acc_ase,
    double* out_acc_nli,
):
    cdef Py_ssize_t span_index
    cdef Py_ssize_t running_index
    cdef double acc_gsnr = 0.0
    cdef double acc_ase = 0.0
    cdef double acc_nli = 0.0
    cdef double attenuation
    cdef double span_length_m
    cdef double l_eff_a
    cdef double l_eff
    cdef double sum_phi
    cdef double phi
    cdef double power_nli_span
    cdef double power_ase
    cdef double delta_frequency
    cdef double running_bandwidth

    for span_index in range(span_start, span_end):
        span_length_m = span_lengths_km[span_index] * 1e3
        attenuation = span_attenuation_normalized[span_index]
        power_nli_span = 0.0

        if include_nli:
            l_eff_a = 1.0 / (2.0 * attenuation)
            l_eff = (1.0 - exp(-2.0 * attenuation * span_length_m)) / (2.0 * attenuation)
            sum_phi = asinh(PI_SQUARED * ABS_BETA_2 * (bandwidth * bandwidth) / (4.0 * attenuation))

            for running_index in range(running_start, running_end):
                if running_service_ids[running_index] == current_service_id:
                    continue
                delta_frequency = running_center_frequencies[running_index] - center_frequency
                if delta_frequency == 0.0:
                    continue
                running_bandwidth = running_bandwidths[running_index]
                phi = (
                    asinh(
                        PI_SQUARED
                        * ABS_BETA_2
                        * l_eff_a
                        * running_bandwidth
                        * (delta_frequency + (running_bandwidth / 2.0))
                    )
                    - asinh(
                        PI_SQUARED
                        * ABS_BETA_2
                        * l_eff_a
                        * running_bandwidth
                        * (delta_frequency - (running_bandwidth / 2.0))
                    )
                ) - (
                    running_phi_modulation[running_index]
                    * (running_bandwidth / fabs(delta_frequency))
                    * (5.0 / 3.0)
                    * (l_eff / span_length_m)
                )
                sum_phi += phi

            power_nli_span = nli_prefactor * l_eff * sum_phi

        power_ase = (
            bandwidth
            * H_PLANCK
            * center_frequency
            * (exp(2.0 * attenuation * span_length_m) - 1.0)
            * span_noise_figure_normalized[span_index]
        )

        if include_nli:
            acc_gsnr += (power_ase + power_nli_span) / launch_power
            if power_nli_span > 0.0:
                acc_nli += power_nli_span / launch_power
        else:
            acc_gsnr += power_ase / launch_power

        acc_ase += power_ase / launch_power

    out_acc_gsnr[0] = acc_gsnr
    out_acc_ase[0] = acc_ase
    out_acc_nli[0] = acc_nli


def accumulate_link_noise(
    cnp.ndarray[cnp.float64_t, ndim=1] span_lengths_km,
    cnp.ndarray[cnp.float64_t, ndim=1] span_attenuation_normalized,
    cnp.ndarray[cnp.float64_t, ndim=1] span_noise_figure_normalized,
    cnp.ndarray[cnp.int32_t, ndim=1] running_service_ids,
    cnp.ndarray[cnp.float64_t, ndim=1] running_center_frequencies,
    cnp.ndarray[cnp.float64_t, ndim=1] running_bandwidths,
    cnp.ndarray[cnp.float64_t, ndim=1] running_phi_modulation,
    *,
    int current_service_id,
    double center_frequency,
    double bandwidth,
    double launch_power,
    bint include_nli,
):
    cdef double acc_gsnr = 0.0
    cdef double acc_ase = 0.0
    cdef double acc_nli = 0.0
    cdef double nli_prefactor = ((launch_power / bandwidth) ** 3) * NLI_PREFACTOR_BASE * (GAMMA ** 2) * bandwidth

    _accumulate_link_noise_impl(
        span_lengths_km,
        0,
        span_lengths_km.shape[0],
        span_attenuation_normalized,
        span_noise_figure_normalized,
        running_service_ids,
        0,
        running_service_ids.shape[0],
        running_center_frequencies,
        running_bandwidths,
        running_phi_modulation,
        current_service_id,
        center_frequency,
        bandwidth,
        launch_power,
        nli_prefactor,
        include_nli,
        &acc_gsnr,
        &acc_ase,
        &acc_nli,
    )

    return acc_gsnr, acc_ase, acc_nli


def summarize_candidate_starts(
    cnp.ndarray[cnp.int32_t, ndim=1] span_offsets,
    cnp.ndarray[cnp.float64_t, ndim=1] span_lengths_km,
    cnp.ndarray[cnp.float64_t, ndim=1] span_attenuation_normalized,
    cnp.ndarray[cnp.float64_t, ndim=1] span_noise_figure_normalized,
    cnp.ndarray[cnp.int32_t, ndim=1] running_offsets,
    cnp.ndarray[cnp.int32_t, ndim=1] running_service_ids,
    cnp.ndarray[cnp.float64_t, ndim=1] running_center_frequencies,
    cnp.ndarray[cnp.float64_t, ndim=1] running_bandwidths,
    cnp.ndarray[cnp.float64_t, ndim=1] running_phi_modulation,
    cnp.ndarray[cnp.int32_t, ndim=1] candidate_starts,
    *,
    int current_service_id,
    double frequency_start,
    double frequency_slot_bandwidth,
    int service_num_slots,
    double launch_power,
    double threshold,
    bint include_nli,
):
    cdef Py_ssize_t candidate_count = candidate_starts.shape[0]
    cdef Py_ssize_t candidate_pos
    cdef Py_ssize_t link_pos
    cdef Py_ssize_t link_count = span_offsets.shape[0] - 1
    cdef cnp.int32_t start_slot
    cdef int span_start
    cdef int span_end
    cdef int running_start
    cdef int running_end
    cdef double bandwidth = frequency_slot_bandwidth * service_num_slots
    cdef double center_frequency_offset = frequency_slot_bandwidth * (service_num_slots / 2.0)
    cdef double center_frequency
    cdef double nli_prefactor = ((launch_power / bandwidth) ** 3) * NLI_PREFACTOR_BASE * (GAMMA ** 2) * bandwidth
    cdef double acc_gsnr
    cdef double acc_ase
    cdef double acc_nli
    cdef double link_acc_gsnr
    cdef double link_acc_ase
    cdef double link_acc_nli
    cdef double link_nli_share
    cdef double worst_link_nli_share
    cdef double osnr
    cdef double total_nli_share
    cdef cnp.ndarray[cnp.npy_bool, ndim=1] meets_threshold = np.zeros(candidate_count, dtype=np.bool_)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] osnr_margin = np.zeros(candidate_count, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] nli_share = np.zeros(candidate_count, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] worst_link_nli_share_values = np.zeros(candidate_count, dtype=np.float64)
    cdef const cnp.int32_t[:] span_offsets_view = span_offsets
    cdef const cnp.float64_t[:] span_lengths_view = span_lengths_km
    cdef const cnp.float64_t[:] span_attenuation_view = span_attenuation_normalized
    cdef const cnp.float64_t[:] span_noise_figure_view = span_noise_figure_normalized
    cdef const cnp.int32_t[:] running_offsets_view = running_offsets
    cdef const cnp.int32_t[:] running_service_ids_view = running_service_ids
    cdef const cnp.float64_t[:] running_center_frequencies_view = running_center_frequencies
    cdef const cnp.float64_t[:] running_bandwidths_view = running_bandwidths
    cdef const cnp.float64_t[:] running_phi_modulation_view = running_phi_modulation
    cdef const cnp.int32_t[:] candidate_starts_view = candidate_starts
    cdef cnp.npy_bool[:] meets_threshold_view = meets_threshold
    cdef cnp.float64_t[:] osnr_margin_view = osnr_margin
    cdef cnp.float64_t[:] nli_share_view = nli_share
    cdef cnp.float64_t[:] worst_link_nli_share_view = worst_link_nli_share_values

    for candidate_pos in range(candidate_count):
        start_slot = candidate_starts_view[candidate_pos]
        center_frequency = (
            frequency_start
            + (frequency_slot_bandwidth * start_slot)
            + center_frequency_offset
        )
        acc_gsnr = 0.0
        acc_ase = 0.0
        acc_nli = 0.0
        worst_link_nli_share = 0.0

        for link_pos in range(link_count):
            span_start = span_offsets_view[link_pos]
            span_end = span_offsets_view[link_pos + 1]
            running_start = running_offsets_view[link_pos]
            running_end = running_offsets_view[link_pos + 1]

            _accumulate_link_noise_impl(
                span_lengths_view,
                span_start,
                span_end,
                span_attenuation_view,
                span_noise_figure_view,
                running_service_ids_view,
                running_start,
                running_end,
                running_center_frequencies_view,
                running_bandwidths_view,
                running_phi_modulation_view,
                current_service_id,
                center_frequency,
                bandwidth,
                launch_power,
                nli_prefactor,
                include_nli,
                &link_acc_gsnr,
                &link_acc_ase,
                &link_acc_nli,
            )
            acc_gsnr += link_acc_gsnr
            acc_ase += link_acc_ase
            acc_nli += link_acc_nli
            if link_acc_nli > 0.0 or link_acc_ase > 0.0:
                link_nli_share = link_acc_nli / (link_acc_ase + link_acc_nli)
                if link_nli_share > worst_link_nli_share:
                    worst_link_nli_share = link_nli_share

        osnr = 10.0 * log10(1.0 / acc_gsnr)
        total_nli_share = acc_nli / (acc_ase + acc_nli) if (acc_ase > 0.0 or acc_nli > 0.0) else 0.0
        meets_threshold_view[candidate_pos] = osnr >= threshold
        osnr_margin_view[candidate_pos] = osnr - threshold
        nli_share_view[candidate_pos] = total_nli_share
        worst_link_nli_share_view[candidate_pos] = worst_link_nli_share

    return meets_threshold, osnr_margin, nli_share, worst_link_nli_share_values


__all__ = ["accumulate_link_noise", "summarize_candidate_starts"]
