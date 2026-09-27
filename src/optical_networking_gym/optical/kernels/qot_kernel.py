"""Pure-Python twin of ``qot_kernel.pyx`` (used when the extension is not built).

Per-span noise model
--------------------
For every span ``s`` of a link the kernel returns noise-to-signal ratios (NSR,
linear) referred to the channel of interest (CUT):

* ASE: ``P_ASE,s = B h f (G_s - 1) NF_s`` with the amplifier gain
  ``G_s = exp(2 alpha L_s) * L_in,s * L_out,s`` compensating fibre and lumped
  (connector/splice) losses; ``NSR_ASE,s = P_ASE,s / (P * 10^(o_s(f)/10))``
  [Poggiolini_2014_GNModelFiberNonLinear].
* NLI: closed-form incoherent GN/EGN per span with the modulation-format
  correction ``Phi`` [Poggiolini_2017_RecentAdvancesModeling], evaluated with the
  power that enters the fibre, ``P_fibre = P * 10^(o_s(f)/10) / L_in,s``. Each
  interferer ``j`` contributes with weight ``(G_j / G_CUT)^2`` where ``G`` is the
  power spectral density, because XCI scales as ``G_CUT * G_j^2``
  [Poggiolini_2014_GNModelFiberNonLinear]. In ``interferer_psd_actual=False``
  mode (legacy behaviour) all interferers are assumed to have the CUT's PSD.

``o_s(f)`` is an optional per-span channel-power offset in dB (e.g. accumulated
EDFA gain ripple since the last power equalisation), looked up on the slot grid.
With unit losses, no offsets and equal PSDs the model reduces exactly to the
original kernel.

References are listed at the end of the file.
"""

from __future__ import annotations

import math

import numpy as np

_ABS_BETA_2 = abs(-21.3e-27)
_GAMMA = 1.3e-3
_H_PLANCK = 6.626e-34
_PI_SQUARED = math.pi * math.pi
_NLI_PREFACTOR_BASE = 8.0 / (27.0 * math.pi * _ABS_BETA_2)

EMPTY_POWER_OFFSETS = np.zeros((0, 0), dtype=np.float64)


def _nli_prefactor(power_in_fibre: float, bandwidth: float) -> float:
    """``(P/B)^3 * 8/(27 pi |beta2|) * gamma^2 * B``, evaluated as on ``main``."""
    return ((power_in_fibre / bandwidth) ** 3) * _NLI_PREFACTOR_BASE * (_GAMMA**2) * bandwidth


def _slot_index(frequency: float, frequency_start: float, slot_bandwidth: float, n_slots: int) -> int:
    index = int((frequency - frequency_start) / slot_bandwidth)
    if index < 0:
        return 0
    if index >= n_slots:
        return n_slots - 1
    return index


def _accumulate_spans(
    span_start: int,
    span_end: int,
    lengths: np.ndarray,
    attenuations: np.ndarray,
    noise_figures: np.ndarray,
    input_losses: np.ndarray,
    output_losses: np.ndarray,
    power_offsets_db: np.ndarray,
    running_start: int,
    running_end: int,
    service_ids: np.ndarray,
    center_frequencies: np.ndarray,
    running_bandwidth_values: np.ndarray,
    phi_modulation: np.ndarray,
    running_powers: np.ndarray,
    current_service_id: int,
    center_frequency: float,
    bandwidth: float,
    launch_power: float,
    include_nli: bool,
    frequency_start: float,
    frequency_slot_bandwidth: float,
    interferer_psd_actual: bool,
) -> tuple[float, float, float]:
    acc_gsnr = 0.0
    acc_ase = 0.0
    acc_nli = 0.0
    n_offset_slots = power_offsets_db.shape[1] if power_offsets_db.ndim == 2 else 0
    use_offsets = n_offset_slots > 0
    cut_slot = (
        _slot_index(center_frequency, frequency_start, frequency_slot_bandwidth, n_offset_slots)
        if use_offsets
        else 0
    )
    nominal_nli_prefactor = _nli_prefactor(launch_power, bandwidth)

    for span_index in range(span_start, span_end):
        span_length_m = lengths[span_index] * 1e3
        attenuation = attenuations[span_index]
        input_loss = input_losses[span_index]
        # Nominal span (no connector loss, no power offset): per-call prefactor
        # and the exact arithmetic of the incoherent model on main.
        nominal = (not use_offsets) and input_loss == 1.0
        if nominal:
            cut_power_out = launch_power
            cut_power_fibre = launch_power
            span_nli_prefactor = nominal_nli_prefactor
        else:
            offset_linear = 10.0 ** (power_offsets_db[span_index, cut_slot] / 10.0) if use_offsets else 1.0
            cut_power_out = launch_power * offset_linear
            cut_power_fibre = cut_power_out / input_loss
            span_nli_prefactor = _nli_prefactor(cut_power_fibre, bandwidth)
        cut_psd = cut_power_fibre / bandwidth
        power_nli_span = 0.0

        if include_nli:
            l_eff_a = 1.0 / (2.0 * attenuation)
            l_eff = (1.0 - math.exp(-2.0 * attenuation * span_length_m)) / (2.0 * attenuation)
            sum_phi = math.asinh(_PI_SQUARED * _ABS_BETA_2 * (bandwidth**2) / (4.0 * attenuation))

            for running_index in range(running_start, running_end):
                if service_ids[running_index] == current_service_id:
                    continue
                delta_frequency = center_frequencies[running_index] - center_frequency
                if delta_frequency == 0.0:
                    continue
                running_bandwidth = running_bandwidth_values[running_index]
                phi = (
                    math.asinh(
                        _PI_SQUARED
                        * _ABS_BETA_2
                        * l_eff_a
                        * running_bandwidth
                        * (delta_frequency + (running_bandwidth / 2.0))
                    )
                    - math.asinh(
                        _PI_SQUARED
                        * _ABS_BETA_2
                        * l_eff_a
                        * running_bandwidth
                        * (delta_frequency - (running_bandwidth / 2.0))
                    )
                ) - (
                    phi_modulation[running_index]
                    * (running_bandwidth / abs(delta_frequency))
                    * (5.0 / 3.0)
                    * (l_eff / span_length_m)
                )
                if interferer_psd_actual:
                    running_offset = 1.0
                    if use_offsets:
                        running_slot = _slot_index(
                            center_frequencies[running_index],
                            frequency_start,
                            frequency_slot_bandwidth,
                            n_offset_slots,
                        )
                        running_offset = 10.0 ** (power_offsets_db[span_index, running_slot] / 10.0)
                    running_psd = (
                        running_powers[running_index] * running_offset / input_loss
                    ) / running_bandwidth
                    ratio = running_psd / cut_psd
                    phi *= ratio * ratio
                sum_phi += phi

            power_nli_span = span_nli_prefactor * l_eff * sum_phi

        gain = math.exp(2.0 * attenuation * span_length_m) * input_loss * output_losses[span_index]
        power_ase = bandwidth * _H_PLANCK * center_frequency * (gain - 1.0) * noise_figures[span_index]

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

    return acc_gsnr, acc_ase, acc_nli


def accumulate_link_noise(
    span_lengths_km: np.ndarray,
    span_attenuation_normalized: np.ndarray,
    span_noise_figure_normalized: np.ndarray,
    running_service_ids: np.ndarray,
    running_center_frequencies: np.ndarray,
    running_bandwidths: np.ndarray,
    running_phi_modulation: np.ndarray,
    *,
    current_service_id: int,
    center_frequency: float,
    bandwidth: float,
    launch_power: float,
    include_nli: bool,
    span_input_loss: np.ndarray | None = None,
    span_output_loss: np.ndarray | None = None,
    span_power_offset_db: np.ndarray | None = None,
    running_launch_powers: np.ndarray | None = None,
    frequency_start: float = 0.0,
    frequency_slot_bandwidth: float = 12.5e9,
    interferer_psd_actual: bool = False,
) -> tuple[float, float, float]:
    """Accumulate the NSR contributions of one link for one channel.

    Args:
        span_lengths_km, span_attenuation_normalized, span_noise_figure_normalized:
            Per-span length, field attenuation (1/m) and linear noise figure.
        running_*: Descriptors of the channels currently on the link.
        current_service_id: Id of the CUT (excluded from interferers).
        center_frequency, bandwidth, launch_power: CUT parameters (Hz, Hz, W).
        include_nli: Whether NLI contributes to the GSNR.
        span_input_loss, span_output_loss: Linear lumped losses (>= 1) before and
            after the fibre of each span (default: lossless).
        span_power_offset_db: ``(n_spans, n_slots)`` channel-power offsets in dB at
            each span input, or ``None``/zero columns to disable.
        running_launch_powers: Launch power (W) of each running channel; required
            when ``interferer_psd_actual`` is true.
        frequency_start, frequency_slot_bandwidth: Slot grid (for offset lookup).
        interferer_psd_actual: Use each interferer's own PSD (``True``) or assume
            the CUT's PSD for all channels (``False``, legacy).

    Returns:
        ``(nsr_total, nsr_ase, nsr_nli)`` summed over the spans of the link.
    """
    lengths = np.asarray(span_lengths_km, dtype=np.float64)
    n_spans = lengths.shape[0]
    service_ids = np.asarray(running_service_ids, dtype=np.int32)
    n_running = service_ids.shape[0]
    input_losses = (
        np.ones(n_spans) if span_input_loss is None else np.asarray(span_input_loss, dtype=np.float64)
    )
    output_losses = (
        np.ones(n_spans) if span_output_loss is None else np.asarray(span_output_loss, dtype=np.float64)
    )
    offsets = (
        EMPTY_POWER_OFFSETS
        if span_power_offset_db is None
        else np.asarray(span_power_offset_db, dtype=np.float64)
    )
    powers = (
        np.full(n_running, launch_power)
        if running_launch_powers is None
        else np.asarray(running_launch_powers, dtype=np.float64)
    )
    acc = _accumulate_spans(
        0,
        n_spans,
        lengths,
        np.asarray(span_attenuation_normalized, dtype=np.float64),
        np.asarray(span_noise_figure_normalized, dtype=np.float64),
        input_losses,
        output_losses,
        offsets,
        0,
        n_running,
        service_ids,
        np.asarray(running_center_frequencies, dtype=np.float64),
        np.asarray(running_bandwidths, dtype=np.float64),
        np.asarray(running_phi_modulation, dtype=np.float64),
        powers,
        current_service_id,
        center_frequency,
        bandwidth,
        launch_power,
        include_nli,
        frequency_start,
        frequency_slot_bandwidth,
        interferer_psd_actual,
    )
    return float(acc[0]), float(acc[1]), float(acc[2])


def summarize_candidate_starts(
    span_offsets: np.ndarray,
    span_lengths_km: np.ndarray,
    span_attenuation_normalized: np.ndarray,
    span_noise_figure_normalized: np.ndarray,
    running_offsets: np.ndarray,
    running_service_ids: np.ndarray,
    running_center_frequencies: np.ndarray,
    running_bandwidths: np.ndarray,
    running_phi_modulation: np.ndarray,
    candidate_starts: np.ndarray,
    *,
    current_service_id: int,
    frequency_start: float,
    frequency_slot_bandwidth: float,
    service_num_slots: int,
    launch_power: float,
    threshold: float,
    include_nli: bool,
    span_input_loss: np.ndarray | None = None,
    span_output_loss: np.ndarray | None = None,
    span_power_offset_db: np.ndarray | None = None,
    running_launch_powers: np.ndarray | None = None,
    interferer_psd_actual: bool = False,
    nli_scale: float = 1.0,
    extra_nsr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the GSNR margin of every candidate start slot on one path.

    The per-link NSRs are summed over the links of the path (incoherent
    accumulation); the path NLI is then multiplied by ``nli_scale`` (coherent
    accumulation factor, 1 = incoherent) and ``extra_nsr`` (node/transceiver
    terms, constant along the path) is added before converting to GSNR.

    Returns:
        ``(meets_threshold, osnr_margin, nli_share, worst_link_nli_share)``.
    """
    span_offsets_arr = np.asarray(span_offsets, dtype=np.int32)
    lengths = np.asarray(span_lengths_km, dtype=np.float64)
    n_spans = lengths.shape[0]
    attenuations = np.asarray(span_attenuation_normalized, dtype=np.float64)
    noise_figures = np.asarray(span_noise_figure_normalized, dtype=np.float64)
    running_offsets_arr = np.asarray(running_offsets, dtype=np.int32)
    service_ids = np.asarray(running_service_ids, dtype=np.int32)
    center_frequencies = np.asarray(running_center_frequencies, dtype=np.float64)
    running_bandwidth_values = np.asarray(running_bandwidths, dtype=np.float64)
    phi_modulation_values = np.asarray(running_phi_modulation, dtype=np.float64)
    starts = np.asarray(candidate_starts, dtype=np.int32)
    input_losses = (
        np.ones(n_spans) if span_input_loss is None else np.asarray(span_input_loss, dtype=np.float64)
    )
    output_losses = (
        np.ones(n_spans) if span_output_loss is None else np.asarray(span_output_loss, dtype=np.float64)
    )
    offsets = (
        EMPTY_POWER_OFFSETS
        if span_power_offset_db is None
        else np.asarray(span_power_offset_db, dtype=np.float64)
    )
    powers = (
        np.full(service_ids.shape[0], launch_power)
        if running_launch_powers is None
        else np.asarray(running_launch_powers, dtype=np.float64)
    )

    candidate_count = starts.shape[0]
    link_count = max(0, span_offsets_arr.shape[0] - 1)
    bandwidth = frequency_slot_bandwidth * service_num_slots
    center_frequency_offset = frequency_slot_bandwidth * (service_num_slots / 2.0)

    meets_threshold = np.zeros(candidate_count, dtype=np.bool_)
    osnr_margin = np.zeros(candidate_count, dtype=np.float64)
    nli_share = np.zeros(candidate_count, dtype=np.float64)
    worst_link_nli_share_values = np.zeros(candidate_count, dtype=np.float64)

    for candidate_pos, start_slot in enumerate(starts):
        center_frequency = frequency_start + (frequency_slot_bandwidth * start_slot) + center_frequency_offset
        acc_gsnr = 0.0
        acc_ase = 0.0
        acc_nli = 0.0
        acc_nli_raw = 0.0
        worst_link_nli_share = 0.0

        for link_pos in range(link_count):
            link_gsnr, link_ase, link_nli = _accumulate_spans(
                int(span_offsets_arr[link_pos]),
                int(span_offsets_arr[link_pos + 1]),
                lengths,
                attenuations,
                noise_figures,
                input_losses,
                output_losses,
                offsets,
                int(running_offsets_arr[link_pos]),
                int(running_offsets_arr[link_pos + 1]),
                service_ids,
                center_frequencies,
                running_bandwidth_values,
                phi_modulation_values,
                powers,
                current_service_id,
                center_frequency,
                bandwidth,
                launch_power,
                include_nli,
                frequency_start,
                frequency_slot_bandwidth,
                interferer_psd_actual,
            )
            acc_gsnr += link_gsnr
            acc_ase += link_ase
            acc_nli += link_nli
            acc_nli_raw += link_gsnr - link_ase
            if link_nli > 0.0 or link_ase > 0.0:
                link_nli_share = link_nli / (link_ase + link_nli)
                if link_nli_share > worst_link_nli_share:
                    worst_link_nli_share = link_nli_share

        # Defaults (nli_scale=1, extra_nsr=0) leave acc_gsnr bit-identical to
        # the incoherent per-link sum.
        if nli_scale != 1.0:
            acc_gsnr += (nli_scale - 1.0) * acc_nli_raw
            acc_nli *= nli_scale
        acc_gsnr += extra_nsr
        osnr = 10.0 * math.log10(1.0 / acc_gsnr)
        total_nli_share = acc_nli / (acc_ase + acc_nli) if (acc_ase > 0.0 or acc_nli > 0.0) else 0.0
        meets_threshold[candidate_pos] = osnr >= threshold
        osnr_margin[candidate_pos] = osnr - threshold
        nli_share[candidate_pos] = total_nli_share
        worst_link_nli_share_values[candidate_pos] = worst_link_nli_share

    return meets_threshold, osnr_margin, nli_share, worst_link_nli_share_values


def path_noise(
    span_offsets: np.ndarray,
    span_lengths_km: np.ndarray,
    span_attenuation_normalized: np.ndarray,
    span_noise_figure_normalized: np.ndarray,
    span_input_loss: np.ndarray,
    span_output_loss: np.ndarray,
    span_power_offset_db: np.ndarray,
    running_offsets: np.ndarray,
    running_service_ids: np.ndarray,
    running_center_frequencies: np.ndarray,
    running_bandwidths: np.ndarray,
    running_phi_modulation: np.ndarray,
    running_launch_powers: np.ndarray,
    *,
    current_service_id: int,
    center_frequency: float,
    bandwidth: float,
    launch_power: float,
    include_nli: bool,
    frequency_start: float,
    frequency_slot_bandwidth: float,
    interferer_psd_actual: bool,
    nli_scale: float = 1.0,
    extra_nsr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Per-link and path-total NSR of one channel on one path.

    Same physics and inputs as :func:`summarize_candidate_starts` (all links of
    the path concatenated, ``span_offsets``/``running_offsets`` delimiting each
    link), for a single channel.

    Returns:
        ``(link_nsr_total, link_nsr_ase, link_nsr_nli, path_nsr_total,
        path_nsr_ase, path_nsr_nli, worst_link_nli_share)``. Per-link values are
        incoherent; the path NLI and total include the coherence scale and the
        path total includes ``extra_nsr``.
    """
    offsets_arr = np.asarray(span_offsets, dtype=np.int32)
    running_offsets_arr = np.asarray(running_offsets, dtype=np.int32)
    link_count = max(0, offsets_arr.shape[0] - 1)
    link_gsnr = np.zeros(link_count, dtype=np.float64)
    link_ase = np.zeros(link_count, dtype=np.float64)
    link_nli = np.zeros(link_count, dtype=np.float64)
    offsets = np.asarray(span_power_offset_db, dtype=np.float64)
    if offsets.ndim != 2:
        offsets = EMPTY_POWER_OFFSETS
    acc_gsnr = 0.0
    acc_ase = 0.0
    acc_nli = 0.0
    acc_nli_raw = 0.0
    worst_link_nli_share = 0.0
    for link_pos in range(link_count):
        gsnr, ase, nli = _accumulate_spans(
            int(offsets_arr[link_pos]),
            int(offsets_arr[link_pos + 1]),
            np.asarray(span_lengths_km, dtype=np.float64),
            np.asarray(span_attenuation_normalized, dtype=np.float64),
            np.asarray(span_noise_figure_normalized, dtype=np.float64),
            np.asarray(span_input_loss, dtype=np.float64),
            np.asarray(span_output_loss, dtype=np.float64),
            offsets,
            int(running_offsets_arr[link_pos]),
            int(running_offsets_arr[link_pos + 1]),
            np.asarray(running_service_ids, dtype=np.int32),
            np.asarray(running_center_frequencies, dtype=np.float64),
            np.asarray(running_bandwidths, dtype=np.float64),
            np.asarray(running_phi_modulation, dtype=np.float64),
            np.asarray(running_launch_powers, dtype=np.float64),
            current_service_id,
            center_frequency,
            bandwidth,
            launch_power,
            include_nli,
            frequency_start,
            frequency_slot_bandwidth,
            interferer_psd_actual,
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

# References
# [Poggiolini_2014_GNModelFiberNonLinear] P. Poggiolini, G. Bosco, A. Carena, V. Curri,
#     Y. Jiang, and F. Forghieri, "The GN-Model of Fiber Non-Linear Propagation and Its
#     Applications," Journal of Lightwave Technology, vol. 32, no. 4, pp. 694-721,
#     Feb. 2014, doi: 10.1109/JLT.2013.2295208.
# [Poggiolini_2017_RecentAdvancesModeling] P. Poggiolini and Y. Jiang, "Recent Advances
#     in the Modeling of the Impact of Nonlinear Fiber Propagation Effects on
#     Uncompensated Coherent Transmission Systems," Journal of Lightwave Technology,
#     vol. 35, no. 3, pp. 458-480, Feb. 2017, doi: 10.1109/JLT.2016.2613893.
