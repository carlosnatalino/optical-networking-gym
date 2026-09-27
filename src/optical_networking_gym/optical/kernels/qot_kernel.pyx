# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True
"""Compiled QoT kernel. See ``qot_kernel.py`` (the pure-Python twin with the
same API) for the physical model and references, and ``optical/cfm2.py`` for
the CFM2 modulation-format correction factors."""

from libc.math cimport asinh, exp, fabs, log10, pow
import numpy as np
cimport numpy as cnp

from optical_networking_gym.optical.cfm2 import ABS_BETA2_PS2_PER_KM as _ABS_BETA2_PS2_PER_KM
from optical_networking_gym.optical.cfm2 import CFM2_COEFFICIENTS as _CFM2


cdef double ABS_BETA_2 = 21.3e-27
cdef double GAMMA = 1.3e-3
cdef double H_PLANCK = 6.626e-34
cdef double PI_VALUE = 3.14159265358979323846
cdef double PI_SQUARED = PI_VALUE * PI_VALUE
cdef double NLI_PREFACTOR_BASE = 8.0 / (27.0 * PI_VALUE * ABS_BETA_2)

# CFM2 SCI factor ``rho_CUT`` (a9..a18 of ``optical/cfm2.py``, the single
# source of the coefficients); the XCI factors ``rho_nch`` are computed by the
# caller, since they do not depend on the channel under test.
cdef double CFM2_A9 = _CFM2[8]
cdef double CFM2_A10 = _CFM2[9]
cdef double CFM2_A11 = _CFM2[10]
cdef double CFM2_A12 = _CFM2[11]
cdef double CFM2_A13 = _CFM2[12]
cdef double CFM2_A14 = _CFM2[13]
cdef double CFM2_A15 = _CFM2[14]
cdef double CFM2_A16 = _CFM2[15]
cdef double CFM2_A17 = _CFM2[16]
cdef double CFM2_A18 = _CFM2[17]
cdef double BETA2_PS2_PER_KM = _ABS_BETA2_PS2_PER_KM

EMPTY_POWER_OFFSETS = np.zeros((0, 0), dtype=np.float64)


cdef inline Py_ssize_t _slot_index(
    double frequency,
    double frequency_start,
    double slot_bandwidth,
    Py_ssize_t n_slots,
) nogil:
    cdef Py_ssize_t index = <Py_ssize_t>((frequency - frequency_start) / slot_bandwidth)
    if index < 0:
        return 0
    if index >= n_slots:
        return n_slots - 1
    return index


cdef inline double _nli_prefactor(double power_in_fibre, double bandwidth) noexcept nogil:
    """``(P/B)^3 * 8/(27 pi |beta2|) * gamma^2 * B``, evaluated as on ``main``."""
    return pow(power_in_fibre / bandwidth, 3.0) * NLI_PREFACTOR_BASE * pow(GAMMA, 2.0) * bandwidth


cdef inline cnp.ndarray _contiguous(cnp.ndarray values):
    if cnp.PyArray_IS_C_CONTIGUOUS(values):
        return values
    return np.ascontiguousarray(values)


cdef inline const double* _f64_ptr(cnp.ndarray values) noexcept:
    return <const double*> cnp.PyArray_DATA(values)


# Per-span terms that do not depend on the candidate frequency, computed once per
# kernel call (row-major ``(n_spans, N_SPAN_TERMS)``) with the same expressions
# as the per-candidate loop used before, so the results are bit-identical.
cdef enum:
    N_SPAN_TERMS = 6
    TERM_LENGTH_M = 0
    TERM_L_EFF_A = 1
    TERM_L_EFF = 2
    TERM_SUM_PHI_SELF = 3
    TERM_GAIN_MINUS_ONE = 4
    TERM_L_EFF_OVER_LENGTH = 5


cdef inline void _precompute_span_terms(
    Py_ssize_t n_spans,
    const double* span_lengths_km,
    const double* span_attenuation_normalized,
    const double* span_input_loss,
    const double* span_output_loss,
    double bandwidth,
    bint cfm2,
    double cut_phi,
    double cut_start_distance_km,
    double* terms,
) noexcept nogil:
    cdef Py_ssize_t span_index
    cdef double span_length_m
    cdef double attenuation
    cdef double l_eff
    cdef double* row
    # CFM2: rho_CUT depends on the span only through the dispersion the CUT
    # has accumulated since its transmitter, so the per-call parts are hoisted
    # and the factor is folded into the SCI term (zero cost per candidate).
    cdef double distance_km = cut_start_distance_km
    cdef double rho_cut_base = 0.0
    cdef double rho_cut_scale = 0.0
    cdef double rho_cut_rate = 0.0
    if cfm2:
        rho_cut_base = CFM2_A9 + CFM2_A10 * pow(cut_phi, CFM2_A11)
        rho_cut_scale = CFM2_A12 * pow(cut_phi, CFM2_A13)
        rho_cut_rate = 1.0 + CFM2_A14 * pow(bandwidth * 1e-12, CFM2_A15)
    for span_index in range(n_spans):
        row = terms + span_index * N_SPAN_TERMS
        span_length_m = span_lengths_km[span_index] * 1e3
        attenuation = span_attenuation_normalized[span_index]
        l_eff = (1.0 - exp(-2.0 * attenuation * span_length_m)) / (2.0 * attenuation)
        row[TERM_LENGTH_M] = span_length_m
        row[TERM_L_EFF_A] = 1.0 / (2.0 * attenuation)
        row[TERM_L_EFF] = l_eff
        row[TERM_SUM_PHI_SELF] = asinh(PI_SQUARED * ABS_BETA_2 * (bandwidth * bandwidth) / (4.0 * attenuation))
        if cfm2:
            row[TERM_SUM_PHI_SELF] *= rho_cut_base + rho_cut_scale * (
                rho_cut_rate + CFM2_A16 * pow(BETA2_PS2_PER_KM * distance_km + CFM2_A17, CFM2_A18)
            )
            distance_km += span_lengths_km[span_index]
        row[TERM_GAIN_MINUS_ONE] = (
            exp(2.0 * attenuation * span_length_m) * span_input_loss[span_index] * span_output_loss[span_index]
            - 1.0
        )
        row[TERM_L_EFF_OVER_LENGTH] = l_eff / span_length_m


cdef inline cnp.ndarray _span_terms(
    Py_ssize_t n_spans,
    cnp.ndarray lengths,
    cnp.ndarray attenuations,
    cnp.ndarray input_losses,
    cnp.ndarray output_losses,
    double bandwidth,
    bint cfm2,
    double cut_phi,
    double cut_start_distance_km,
):
    cdef cnp.ndarray terms = np.empty(n_spans * N_SPAN_TERMS, dtype=np.float64)
    _precompute_span_terms(
        n_spans,
        _f64_ptr(lengths),
        _f64_ptr(attenuations),
        _f64_ptr(input_losses),
        _f64_ptr(output_losses),
        bandwidth,
        cfm2,
        cut_phi,
        cut_start_distance_km,
        <double*> cnp.PyArray_DATA(terms),
    )
    return terms


# The span loop runs once per (candidate start, link), so it takes raw pointers
# (C-contiguous arrays prepared by the callers) and is ``noexcept nogil``:
# passing typed memoryviews by value costs a struct copy and a refcount per
# argument per call, which doubled the kernel time on the hot path.
cdef inline void _accumulate_link_noise_impl(
    const double* span_terms,
    Py_ssize_t span_start,
    Py_ssize_t span_end,
    const double* span_noise_figure_normalized,
    const double* span_input_loss,
    const double* span_power_offset_db,
    Py_ssize_t n_offset_slots,
    const cnp.int32_t* running_service_ids,
    Py_ssize_t running_start,
    Py_ssize_t running_end,
    const double* running_center_frequencies,
    const double* running_bandwidths,
    const double* running_phi_modulation,
    const double* running_launch_powers,
    const double* running_rho,
    int current_service_id,
    double center_frequency,
    double bandwidth,
    double launch_power,
    double nominal_nli_prefactor,
    bint include_nli,
    double frequency_start,
    double frequency_slot_bandwidth,
    bint interferer_psd_actual,
    double* out_acc_gsnr,
    double* out_acc_ase,
    double* out_acc_nli,
) noexcept nogil:
    cdef Py_ssize_t span_index
    cdef Py_ssize_t running_index
    cdef bint use_offsets = n_offset_slots > 0
    cdef bint nominal
    cdef Py_ssize_t cut_slot = 0
    cdef Py_ssize_t running_slot
    cdef double acc_gsnr = 0.0
    cdef double acc_ase = 0.0
    cdef double acc_nli = 0.0
    cdef const double* row
    cdef double span_length_m
    cdef double input_loss
    cdef double offset_linear
    cdef double running_offset
    cdef double cut_power_out
    cdef double cut_power_fibre
    cdef double cut_psd = 0.0
    cdef double span_nli_prefactor
    cdef double running_psd
    cdef double ratio
    cdef double l_eff_a
    cdef double l_eff
    cdef double l_eff_over_length
    cdef double sum_phi
    cdef double phi
    cdef double power_nli_span
    cdef double power_ase
    cdef double delta_frequency
    cdef double running_bandwidth
    # CFM2 XCI factors of this link: row-major ``(n_spans_link, n_running_link)``,
    # or NULL for the legacy closed-form EGN correction.
    cdef bint use_rho = running_rho != NULL
    cdef Py_ssize_t n_link_running = running_end - running_start
    cdef const double* rho_row = NULL

    if use_offsets:
        cut_slot = _slot_index(center_frequency, frequency_start, frequency_slot_bandwidth, n_offset_slots)

    for span_index in range(span_start, span_end):
        row = span_terms + span_index * N_SPAN_TERMS
        span_length_m = row[TERM_LENGTH_M]
        input_loss = span_input_loss[span_index]
        # Nominal span (no connector loss, no power offset): reuse the per-call
        # prefactor and the exact arithmetic of the incoherent model on main.
        nominal = (not use_offsets) and input_loss == 1.0
        if nominal:
            cut_power_out = launch_power
            cut_power_fibre = launch_power
            span_nli_prefactor = nominal_nli_prefactor
        else:
            offset_linear = 1.0
            if use_offsets:
                offset_linear = pow(10.0, span_power_offset_db[span_index * n_offset_slots + cut_slot] / 10.0)
            cut_power_out = launch_power * offset_linear
            cut_power_fibre = cut_power_out / input_loss
            span_nli_prefactor = _nli_prefactor(cut_power_fibre, bandwidth)
        if interferer_psd_actual:
            cut_psd = cut_power_fibre / bandwidth
        power_nli_span = 0.0

        if include_nli:
            l_eff_a = row[TERM_L_EFF_A]
            l_eff = row[TERM_L_EFF]
            l_eff_over_length = row[TERM_L_EFF_OVER_LENGTH]
            sum_phi = row[TERM_SUM_PHI_SELF]
            if use_rho:
                rho_row = running_rho + (span_index - span_start) * n_link_running

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
                )
                if use_rho:
                    # CFM2: multiplicative factor of the interferer.
                    phi *= rho_row[running_index - running_start]
                else:
                    # Closed-form EGN XCI correction (Poggiolini 2015).
                    phi -= (
                        running_phi_modulation[running_index]
                        * (running_bandwidth / fabs(delta_frequency))
                        * (5.0 / 3.0)
                        * l_eff_over_length
                    )
                if interferer_psd_actual:
                    running_offset = 1.0
                    if use_offsets:
                        running_slot = _slot_index(
                            running_center_frequencies[running_index],
                            frequency_start,
                            frequency_slot_bandwidth,
                            n_offset_slots,
                        )
                        running_offset = pow(
                            10.0, span_power_offset_db[span_index * n_offset_slots + running_slot] / 10.0
                        )
                    running_psd = (
                        running_launch_powers[running_index] * running_offset / input_loss
                    ) / running_bandwidth
                    ratio = running_psd / cut_psd
                    phi *= ratio * ratio
                sum_phi += phi

            power_nli_span = span_nli_prefactor * l_eff * sum_phi

        power_ase = (
            bandwidth
            * H_PLANCK
            * center_frequency
            * row[TERM_GAIN_MINUS_ONE]
            * span_noise_figure_normalized[span_index]
        )

        if nominal:
            # Same expressions (and rounding) as the incoherent model on main.
            if include_nli:
                acc_gsnr += (power_ase + power_nli_span) / launch_power
                if power_nli_span > 0.0:
                    acc_nli += power_nli_span / launch_power
            else:
                acc_gsnr += power_ase / launch_power
            acc_ase += power_ase / launch_power
        else:
            # ASE is referred to the amplifier output, NLI to the fibre input.
            if include_nli:
                acc_gsnr += power_ase / cut_power_out + power_nli_span / cut_power_fibre
                if power_nli_span > 0.0:
                    acc_nli += power_nli_span / cut_power_fibre
            else:
                acc_gsnr += power_ase / cut_power_out
            acc_ase += power_ase / cut_power_out

    out_acc_gsnr[0] = acc_gsnr
    out_acc_ase[0] = acc_ase
    out_acc_nli[0] = acc_nli


cdef inline bint _is_ready_f64(object values):
    """True for a C-contiguous float64 ndarray (usable without a Python-level copy)."""
    return (
        isinstance(values, cnp.ndarray)
        and cnp.PyArray_TYPE(<cnp.ndarray> values) == cnp.NPY_FLOAT64
        and cnp.PyArray_IS_C_CONTIGUOUS(<cnp.ndarray> values)
    )


cdef inline cnp.ndarray _as_f64(object values, Py_ssize_t size, double fill):
    if values is None:
        return np.full(size, fill, dtype=np.float64)
    if _is_ready_f64(values):
        return <cnp.ndarray> values
    return np.ascontiguousarray(values, dtype=np.float64)


cdef inline cnp.ndarray _as_offsets(object values):
    if values is None:
        return EMPTY_POWER_OFFSETS
    if _is_ready_f64(values):
        return <cnp.ndarray> values
    return np.ascontiguousarray(values, dtype=np.float64)


cdef inline cnp.ndarray _link_rho_bases(
    const cnp.int32_t[:] span_offsets,
    const cnp.int32_t[:] running_offsets,
    Py_ssize_t link_count,
    bint cfm2,
):
    """Start of each link's CFM2 ``(n_spans_link, n_running_link)`` block in the
    flat ``running_rho`` array; the last entry is the expected total size (all
    zeros when CFM2 is disabled)."""
    cdef cnp.ndarray bases = np.zeros(link_count + 1, dtype=np.intp)
    cdef Py_ssize_t[:] bases_view = bases
    cdef Py_ssize_t link_pos
    cdef Py_ssize_t total = 0
    if cfm2:
        for link_pos in range(link_count):
            bases_view[link_pos] = total
            total += <Py_ssize_t>(span_offsets[link_pos + 1] - span_offsets[link_pos]) * (
                running_offsets[link_pos + 1] - running_offsets[link_pos]
            )
        bases_view[link_count] = total
    return bases


cdef inline object _as_rho(object running_rho, bint cfm2, Py_ssize_t expected):
    """Validated CFM2 XCI factors, or ``None`` (legacy correction)."""
    if not cfm2 or expected == 0:
        return None
    if running_rho is None:
        raise ValueError("running_rho is required when cfm2 is enabled and the path has interferers")
    cdef cnp.ndarray rho = np.ascontiguousarray(running_rho, dtype=np.float64)
    if rho.shape[0] != expected:
        raise ValueError(f"running_rho has {rho.shape[0]} entries, expected {expected}")
    return rho


cdef inline const double* _rho_ptr(object rho, Py_ssize_t base) noexcept:
    if rho is None:
        return NULL
    return _f64_ptr(<cnp.ndarray> rho) + base


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
    object span_input_loss=None,
    object span_output_loss=None,
    object span_power_offset_db=None,
    object running_launch_powers=None,
    double frequency_start=0.0,
    double frequency_slot_bandwidth=12.5e9,
    bint interferer_psd_actual=False,
    bint cfm2=False,
    double cut_phi=0.0,
    object running_rho=None,
    double cut_start_distance_km=0.0,
):
    cdef Py_ssize_t n_spans = span_lengths_km.shape[0]
    cdef Py_ssize_t n_running = running_service_ids.shape[0]
    cdef object rho = _as_rho(running_rho, cfm2, n_spans * n_running)
    cdef double acc_gsnr = 0.0
    cdef double acc_ase = 0.0
    cdef double acc_nli = 0.0
    cdef cnp.ndarray lengths = _contiguous(span_lengths_km)
    cdef cnp.ndarray attenuations = _contiguous(span_attenuation_normalized)
    cdef cnp.ndarray noise_figures = _contiguous(span_noise_figure_normalized)
    cdef cnp.ndarray service_ids = _contiguous(running_service_ids)
    cdef cnp.ndarray frequencies = _contiguous(running_center_frequencies)
    cdef cnp.ndarray bandwidths = _contiguous(running_bandwidths)
    cdef cnp.ndarray phis = _contiguous(running_phi_modulation)
    cdef cnp.ndarray input_losses = _as_f64(span_input_loss, n_spans, 1.0)
    cdef cnp.ndarray output_losses = _as_f64(span_output_loss, n_spans, 1.0)
    cdef cnp.ndarray offsets = _as_offsets(span_power_offset_db)
    cdef cnp.ndarray powers = _as_f64(running_launch_powers, n_running, launch_power)
    cdef Py_ssize_t n_offset_slots = offsets.shape[1]
    cdef cnp.ndarray terms = _span_terms(
        n_spans,
        lengths,
        attenuations,
        input_losses,
        output_losses,
        bandwidth,
        cfm2,
        cut_phi,
        cut_start_distance_km,
    )

    _accumulate_link_noise_impl(
        _f64_ptr(terms),
        0,
        n_spans,
        _f64_ptr(noise_figures),
        _f64_ptr(input_losses),
        _f64_ptr(offsets) if n_offset_slots > 0 else NULL,
        n_offset_slots,
        <const cnp.int32_t*> cnp.PyArray_DATA(service_ids),
        0,
        n_running,
        _f64_ptr(frequencies),
        _f64_ptr(bandwidths),
        _f64_ptr(phis),
        _f64_ptr(powers),
        _rho_ptr(rho, 0),
        current_service_id,
        center_frequency,
        bandwidth,
        launch_power,
        _nli_prefactor(launch_power, bandwidth),
        include_nli,
        frequency_start,
        frequency_slot_bandwidth,
        interferer_psd_actual,
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
    object span_input_loss=None,
    object span_output_loss=None,
    object span_power_offset_db=None,
    object running_launch_powers=None,
    bint interferer_psd_actual=False,
    double nli_scale=1.0,
    double extra_nsr=0.0,
    bint cfm2=False,
    double cut_phi=0.0,
    object running_rho=None,
):
    cdef Py_ssize_t n_spans = span_lengths_km.shape[0]
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
    cdef double acc_gsnr
    cdef double acc_ase
    cdef double acc_nli
    cdef double acc_nli_raw
    cdef double link_acc_gsnr
    cdef double link_acc_ase
    cdef double link_acc_nli
    cdef double link_nli_share
    cdef double worst_link_nli_share
    cdef double osnr
    cdef double total_nli_share
    cdef cnp.ndarray lengths = _contiguous(span_lengths_km)
    cdef cnp.ndarray attenuations = _contiguous(span_attenuation_normalized)
    cdef cnp.ndarray noise_figures = _contiguous(span_noise_figure_normalized)
    cdef cnp.ndarray service_ids = _contiguous(running_service_ids)
    cdef cnp.ndarray frequencies = _contiguous(running_center_frequencies)
    cdef cnp.ndarray bandwidths = _contiguous(running_bandwidths)
    cdef cnp.ndarray phis = _contiguous(running_phi_modulation)
    cdef cnp.ndarray input_losses = _as_f64(span_input_loss, n_spans, 1.0)
    cdef cnp.ndarray output_losses = _as_f64(span_output_loss, n_spans, 1.0)
    cdef cnp.ndarray offsets = _as_offsets(span_power_offset_db)
    cdef cnp.ndarray powers = _as_f64(running_launch_powers, running_service_ids.shape[0], launch_power)
    cdef Py_ssize_t n_offset_slots = offsets.shape[1]
    cdef cnp.ndarray terms = _span_terms(
        n_spans, lengths, attenuations, input_losses, output_losses, bandwidth, cfm2, cut_phi, 0.0
    )
    cdef cnp.ndarray rho_bases = _link_rho_bases(span_offsets, running_offsets, link_count, cfm2)
    cdef const Py_ssize_t[:] rho_bases_view = rho_bases
    cdef object rho = _as_rho(running_rho, cfm2, rho_bases_view[link_count])
    cdef const double* rho_ptr = _rho_ptr(rho, 0)
    cdef const double* terms_ptr = _f64_ptr(terms)
    cdef const double* noise_figures_ptr = _f64_ptr(noise_figures)
    cdef const double* input_losses_ptr = _f64_ptr(input_losses)
    cdef const double* offsets_ptr = _f64_ptr(offsets) if n_offset_slots > 0 else NULL
    cdef const cnp.int32_t* service_ids_ptr = <const cnp.int32_t*> cnp.PyArray_DATA(service_ids)
    cdef const double* frequencies_ptr = _f64_ptr(frequencies)
    cdef const double* bandwidths_ptr = _f64_ptr(bandwidths)
    cdef const double* phis_ptr = _f64_ptr(phis)
    cdef const double* powers_ptr = _f64_ptr(powers)
    cdef double nominal_nli_prefactor = _nli_prefactor(launch_power, bandwidth)
    cdef cnp.ndarray[cnp.npy_bool, ndim=1] meets_threshold = np.zeros(candidate_count, dtype=np.bool_)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] osnr_margin = np.zeros(candidate_count, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] nli_share = np.zeros(candidate_count, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] worst_link_nli_share_values = np.zeros(candidate_count, dtype=np.float64)
    cdef const cnp.int32_t[:] span_offsets_view = span_offsets
    cdef const cnp.int32_t[:] running_offsets_view = running_offsets
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
        acc_nli_raw = 0.0
        worst_link_nli_share = 0.0

        for link_pos in range(link_count):
            span_start = span_offsets_view[link_pos]
            span_end = span_offsets_view[link_pos + 1]
            running_start = running_offsets_view[link_pos]
            running_end = running_offsets_view[link_pos + 1]

            _accumulate_link_noise_impl(
                terms_ptr,
                span_start,
                span_end,
                noise_figures_ptr,
                input_losses_ptr,
                offsets_ptr,
                n_offset_slots,
                service_ids_ptr,
                running_start,
                running_end,
                frequencies_ptr,
                bandwidths_ptr,
                phis_ptr,
                powers_ptr,
                rho_ptr + rho_bases_view[link_pos] if rho_ptr != NULL else NULL,
                current_service_id,
                center_frequency,
                bandwidth,
                launch_power,
                nominal_nli_prefactor,
                include_nli,
                frequency_start,
                frequency_slot_bandwidth,
                interferer_psd_actual,
                &link_acc_gsnr,
                &link_acc_ase,
                &link_acc_nli,
            )
            acc_gsnr += link_acc_gsnr
            acc_ase += link_acc_ase
            acc_nli += link_acc_nli
            acc_nli_raw += link_acc_gsnr - link_acc_ase
            if link_acc_nli > 0.0 or link_acc_ase > 0.0:
                link_nli_share = link_acc_nli / (link_acc_ase + link_acc_nli)
                if link_nli_share > worst_link_nli_share:
                    worst_link_nli_share = link_nli_share

        # Defaults (nli_scale=1, extra_nsr=0) leave acc_gsnr bit-identical to
        # the incoherent per-link sum.
        if nli_scale != 1.0:
            acc_gsnr += (nli_scale - 1.0) * acc_nli_raw
            acc_nli *= nli_scale
        acc_gsnr += extra_nsr
        osnr = 10.0 * log10(1.0 / acc_gsnr)
        total_nli_share = acc_nli / (acc_ase + acc_nli) if (acc_ase > 0.0 or acc_nli > 0.0) else 0.0
        meets_threshold_view[candidate_pos] = osnr >= threshold
        osnr_margin_view[candidate_pos] = osnr - threshold
        nli_share_view[candidate_pos] = total_nli_share
        worst_link_nli_share_view[candidate_pos] = worst_link_nli_share

    return meets_threshold, osnr_margin, nli_share, worst_link_nli_share_values


def path_noise(
    cnp.ndarray[cnp.int32_t, ndim=1] span_offsets,
    cnp.ndarray[cnp.float64_t, ndim=1] span_lengths_km,
    cnp.ndarray[cnp.float64_t, ndim=1] span_attenuation_normalized,
    cnp.ndarray[cnp.float64_t, ndim=1] span_noise_figure_normalized,
    cnp.ndarray[cnp.float64_t, ndim=1] span_input_loss,
    cnp.ndarray[cnp.float64_t, ndim=1] span_output_loss,
    cnp.ndarray[cnp.float64_t, ndim=2] span_power_offset_db,
    cnp.ndarray[cnp.int32_t, ndim=1] running_offsets,
    cnp.ndarray[cnp.int32_t, ndim=1] running_service_ids,
    cnp.ndarray[cnp.float64_t, ndim=1] running_center_frequencies,
    cnp.ndarray[cnp.float64_t, ndim=1] running_bandwidths,
    cnp.ndarray[cnp.float64_t, ndim=1] running_phi_modulation,
    cnp.ndarray[cnp.float64_t, ndim=1] running_launch_powers,
    *,
    int current_service_id,
    double center_frequency,
    double bandwidth,
    double launch_power,
    bint include_nli,
    double frequency_start,
    double frequency_slot_bandwidth,
    bint interferer_psd_actual,
    double nli_scale=1.0,
    double extra_nsr=0.0,
    bint cfm2=False,
    double cut_phi=0.0,
    object running_rho=None,
):
    """Per-link and path-total NSR of one channel on one path (see ``qot_kernel.py``)."""
    cdef Py_ssize_t link_count = span_offsets.shape[0] - 1
    cdef Py_ssize_t link_pos
    cdef double gsnr
    cdef double ase
    cdef double nli
    cdef double share
    cdef double acc_gsnr = 0.0
    cdef double acc_ase = 0.0
    cdef double acc_nli = 0.0
    cdef double acc_nli_raw = 0.0
    cdef double worst_link_nli_share = 0.0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] link_gsnr = np.zeros(link_count, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] link_ase = np.zeros(link_count, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] link_nli = np.zeros(link_count, dtype=np.float64)
    cdef const cnp.int32_t[:] span_offsets_view = span_offsets
    cdef const cnp.int32_t[:] running_offsets_view = running_offsets
    cdef cnp.ndarray lengths = _contiguous(span_lengths_km)
    cdef cnp.ndarray attenuations = _contiguous(span_attenuation_normalized)
    cdef cnp.ndarray noise_figures = _contiguous(span_noise_figure_normalized)
    cdef cnp.ndarray input_losses = _contiguous(span_input_loss)
    cdef cnp.ndarray output_losses = _contiguous(span_output_loss)
    cdef cnp.ndarray offsets = _contiguous(span_power_offset_db)
    cdef cnp.ndarray service_ids = _contiguous(running_service_ids)
    cdef cnp.ndarray frequencies = _contiguous(running_center_frequencies)
    cdef cnp.ndarray bandwidths = _contiguous(running_bandwidths)
    cdef cnp.ndarray phis = _contiguous(running_phi_modulation)
    cdef cnp.ndarray powers = _contiguous(running_launch_powers)
    cdef Py_ssize_t n_offset_slots = offsets.shape[1]
    cdef double nominal_nli_prefactor = _nli_prefactor(launch_power, bandwidth)
    cdef cnp.ndarray terms = _span_terms(
        lengths.shape[0], lengths, attenuations, input_losses, output_losses, bandwidth, cfm2, cut_phi, 0.0
    )
    cdef cnp.ndarray rho_bases = _link_rho_bases(span_offsets, running_offsets, link_count, cfm2)
    cdef const Py_ssize_t[:] rho_bases_view = rho_bases
    cdef object rho = _as_rho(running_rho, cfm2, rho_bases_view[link_count])

    for link_pos in range(link_count):
        _accumulate_link_noise_impl(
            _f64_ptr(terms),
            span_offsets_view[link_pos],
            span_offsets_view[link_pos + 1],
            _f64_ptr(noise_figures),
            _f64_ptr(input_losses),
            _f64_ptr(offsets) if n_offset_slots > 0 else NULL,
            n_offset_slots,
            <const cnp.int32_t*> cnp.PyArray_DATA(service_ids),
            running_offsets_view[link_pos],
            running_offsets_view[link_pos + 1],
            _f64_ptr(frequencies),
            _f64_ptr(bandwidths),
            _f64_ptr(phis),
            _f64_ptr(powers),
            _rho_ptr(rho, rho_bases_view[link_pos]),
            current_service_id,
            center_frequency,
            bandwidth,
            launch_power,
            nominal_nli_prefactor,
            include_nli,
            frequency_start,
            frequency_slot_bandwidth,
            interferer_psd_actual,
            &gsnr,
            &ase,
            &nli,
        )
        link_gsnr[link_pos] = gsnr
        link_ase[link_pos] = ase
        link_nli[link_pos] = nli
        acc_gsnr += gsnr
        acc_ase += ase
        acc_nli += nli
        acc_nli_raw += gsnr - ase
        if nli > 0.0 or ase > 0.0:
            share = nli / (ase + nli)
            if share > worst_link_nli_share:
                worst_link_nli_share = share

    if nli_scale != 1.0:
        acc_gsnr += (nli_scale - 1.0) * acc_nli_raw
        acc_nli *= nli_scale
    acc_gsnr += extra_nsr
    return link_gsnr, link_ase, link_nli, acc_gsnr, acc_ase, acc_nli, worst_link_nli_share


__all__ = [
    "EMPTY_POWER_OFFSETS",
    "accumulate_link_noise",
    "path_noise",
    "summarize_candidate_starts",
]
