from __future__ import annotations

import math

import numpy as np


_ABS_BETA_2 = abs(-21.3e-27)
_GAMMA = 1.3e-3
_H_PLANCK = 6.626e-34
_PI_SQUARED = math.pi * math.pi
_NLI_PREFACTOR_BASE = 8.0 / (27.0 * math.pi * _ABS_BETA_2)


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
) -> tuple[float, float, float]:
    lengths = np.asarray(span_lengths_km, dtype=np.float64)
    attenuations = np.asarray(span_attenuation_normalized, dtype=np.float64)
    noise_figures = np.asarray(span_noise_figure_normalized, dtype=np.float64)
    service_ids = np.asarray(running_service_ids, dtype=np.int32)
    center_frequencies = np.asarray(running_center_frequencies, dtype=np.float64)
    running_bandwidth_values = np.asarray(running_bandwidths, dtype=np.float64)
    phi_modulation = np.asarray(running_phi_modulation, dtype=np.float64)

    acc_gsnr = 0.0
    acc_ase = 0.0
    acc_nli = 0.0
    nli_prefactor = ((launch_power / bandwidth) ** 3) * _NLI_PREFACTOR_BASE * (_GAMMA**2) * bandwidth

    for span_index, span_length_km in enumerate(lengths):
        span_length_m = span_length_km * 1e3
        attenuation = attenuations[span_index]
        power_nli_span = 0.0

        if include_nli:
            l_eff_a = 1.0 / (2.0 * attenuation)
            l_eff = (1.0 - math.exp(-2.0 * attenuation * span_length_m)) / (2.0 * attenuation)
            sum_phi = math.asinh(_PI_SQUARED * _ABS_BETA_2 * (bandwidth**2) / (4.0 * attenuation))

            for running_index, running_service_id in enumerate(service_ids):
                if running_service_id == current_service_id:
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
                sum_phi += phi

            power_nli_span = nli_prefactor * l_eff * sum_phi

        power_ase = (
            bandwidth
            * _H_PLANCK
            * center_frequency
            * (math.exp(2.0 * attenuation * span_length_m) - 1.0)
            * noise_figures[span_index]
        )

        if include_nli:
            acc_gsnr += (power_ase + power_nli_span) / launch_power
            if power_nli_span > 0.0:
                acc_nli += power_nli_span / launch_power
        else:
            acc_gsnr += power_ase / launch_power

        acc_ase += power_ase / launch_power

    return float(acc_gsnr), float(acc_ase), float(acc_nli)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    span_offsets_arr = np.asarray(span_offsets, dtype=np.int32)
    lengths = np.asarray(span_lengths_km, dtype=np.float64)
    attenuations = np.asarray(span_attenuation_normalized, dtype=np.float64)
    noise_figures = np.asarray(span_noise_figure_normalized, dtype=np.float64)
    running_offsets_arr = np.asarray(running_offsets, dtype=np.int32)
    service_ids = np.asarray(running_service_ids, dtype=np.int32)
    center_frequencies = np.asarray(running_center_frequencies, dtype=np.float64)
    running_bandwidth_values = np.asarray(running_bandwidths, dtype=np.float64)
    phi_modulation_values = np.asarray(running_phi_modulation, dtype=np.float64)
    starts = np.asarray(candidate_starts, dtype=np.int32)

    candidate_count = starts.shape[0]
    link_count = max(0, span_offsets_arr.shape[0] - 1)
    bandwidth = frequency_slot_bandwidth * service_num_slots
    center_frequency_offset = frequency_slot_bandwidth * (service_num_slots / 2.0)
    nli_prefactor = ((launch_power / bandwidth) ** 3) * _NLI_PREFACTOR_BASE * (_GAMMA**2) * bandwidth

    meets_threshold = np.zeros(candidate_count, dtype=np.bool_)
    osnr_margin = np.zeros(candidate_count, dtype=np.float64)
    nli_share = np.zeros(candidate_count, dtype=np.float64)
    worst_link_nli_share_values = np.zeros(candidate_count, dtype=np.float64)

    for candidate_pos, start_slot in enumerate(starts):
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
            span_start = int(span_offsets_arr[link_pos])
            span_end = int(span_offsets_arr[link_pos + 1])
            running_start = int(running_offsets_arr[link_pos])
            running_end = int(running_offsets_arr[link_pos + 1])
            link_acc_gsnr = 0.0
            link_acc_ase = 0.0
            link_acc_nli = 0.0

            for span_index in range(span_start, span_end):
                span_length_m = lengths[span_index] * 1e3
                attenuation = attenuations[span_index]
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
                            phi_modulation_values[running_index]
                            * (running_bandwidth / abs(delta_frequency))
                            * (5.0 / 3.0)
                            * (l_eff / span_length_m)
                        )
                        sum_phi += phi

                    power_nli_span = nli_prefactor * l_eff * sum_phi

                power_ase = (
                    bandwidth
                    * _H_PLANCK
                    * center_frequency
                    * (math.exp(2.0 * attenuation * span_length_m) - 1.0)
                    * noise_figures[span_index]
                )

                if include_nli:
                    link_acc_gsnr += (power_ase + power_nli_span) / launch_power
                    if power_nli_span > 0.0:
                        link_acc_nli += power_nli_span / launch_power
                else:
                    link_acc_gsnr += power_ase / launch_power

                link_acc_ase += power_ase / launch_power

            acc_gsnr += link_acc_gsnr
            acc_ase += link_acc_ase
            acc_nli += link_acc_nli
            if link_acc_nli > 0.0 or link_acc_ase > 0.0:
                link_nli_share = link_acc_nli / (link_acc_ase + link_acc_nli)
                if link_nli_share > worst_link_nli_share:
                    worst_link_nli_share = link_nli_share

        osnr = 10.0 * math.log10(1.0 / acc_gsnr)
        total_nli_share = acc_nli / (acc_ase + acc_nli) if (acc_ase > 0.0 or acc_nli > 0.0) else 0.0
        meets_threshold[candidate_pos] = osnr >= threshold
        osnr_margin[candidate_pos] = osnr - threshold
        nli_share[candidate_pos] = total_nli_share
        worst_link_nli_share_values[candidate_pos] = worst_link_nli_share

    return meets_threshold, osnr_margin, nli_share, worst_link_nli_share_values


__all__ = ["accumulate_link_noise", "summarize_candidate_starts"]
