
#####################
# Imports e Typing
#####################
from math import pi
from libc.math cimport asinh, pi, exp, log10

import cython
cimport numpy as np
import numpy as np

import typing

if typing.TYPE_CHECKING:
    from optical_networking_gym.envs.qrmsa import QRMSAEnv


#############################
# 1) calculate_osnr
#############################
cpdef calculate_osnr(env: QRMSAEnv, current_service: object):
    cdef double beta_2 = -21.3e-27
    cdef double gamma = 1.3e-3
    cdef double h_plank = 6.626e-34
    cdef double acc_gsnr = 0.0
    cdef double acc_ase = 0.0
    cdef double acc_nli = 0.0
    cdef double gsnr = 0.0
    cdef double ase = 0.0
    cdef double nli = 0.0
    cdef double l_eff_a = 0.0
    cdef double l_eff = 0.0
    cdef double phi = 0.0
    cdef double sum_phi = 0.0
    cdef double power_ase = 0.0
    cdef double power_nli_span = 0.0

    cdef np.ndarray[np.double_t, ndim=1] phi_modulation_format = np.array(
        [1.0, 1.0, 2.0/3.0, 17.0/25.0, 69.0/100.0, 13.0/21.0],
        dtype=np.float64
    )

    cdef object link
    cdef object span
    cdef object running_service

    # Percorre cada link do caminho do serviço
    for link in current_service.path.links:
        # Para cada span do link
        for span in env.topology[link.node1][link.node2]["link"].spans:
            # Cálculo da L_eff e L_eff_a
            l_eff_a = 1.0 / (2.0 * span.attenuation_normalized)
            l_eff = (
                1.0 - np.exp(-2.0 * span.attenuation_normalized * span.length * 1e3)
            ) / (2.0 * span.attenuation_normalized)

            # Inicia sum_phi para este span
            sum_phi = asinh(
                pi**2 * abs(beta_2) * (current_service.bandwidth**2) /
                (4.0 * span.attenuation_normalized)
            )

            # Soma das contribuições NLI de outros serviços rodando nesse link
            for running_service in env.topology[link.node1][link.node2]["running_services"]:
                if running_service.service_id != current_service.service_id:
                    try:
                        # Cálculo do phi
                        phi = (
                            asinh(
                                pi**2 * abs(beta_2) * l_eff_a * running_service.bandwidth *
                                (
                                    running_service.center_frequency
                                    - current_service.center_frequency
                                    + (running_service.bandwidth / 2.0)
                                )
                            )
                            - asinh(
                                pi**2 * abs(beta_2) * l_eff_a * running_service.bandwidth *
                                (
                                    running_service.center_frequency
                                    - current_service.center_frequency
                                    - (running_service.bandwidth / 2.0)
                                )
                            )
                        ) - (
                            phi_modulation_format[running_service.current_modulation.spectral_efficiency - 1]
                            * (
                                running_service.bandwidth
                                / abs(running_service.center_frequency - current_service.center_frequency)
                            )
                            * (5.0 / 3.0)
                            * (l_eff / (span.length * 1e3))
                        )
                        sum_phi += phi
                    except Exception as e:
                        print(f"Error: {e}")
                        print("================= error =================")
                        print("current_time: ", env.current_time)
                        print(f'current_service: {current_service}\n')
                        print(f'running_service: {running_service}\n')
                        index = env.topology[link.node1][link.node2]["index"]
                        print(f'Link {link.node1,link.node2}: {env.topology.graph["available_slots"][index,:]}\n\n')
                        print(f"running services:" )
                        for service in env.topology[link.node1][link.node2]["running_services"]:
                            print(f"ID: {service.service_id}, src: {service.source}, tgt: {service.destination}, Path: {service.path}, init_slot: {service.initial_slot}, numb_slots: {service.number_slots}, BW: {service.bandwidth}, center_freq: {service.center_frequency}, mod: {service.current_modulation}, OSNR: {service.OSNR}, ASE: {service.ASE}, NLI: {service.NLI}\n")
                        raise ValueError("Error in calculate_osnr")

            # Potência de NLI no span
            power_nli_span = (
                (current_service.launch_power / current_service.bandwidth)**3
                * (8.0 / (27.0 * pi * abs(beta_2)))
                * (gamma**2)
                * l_eff
                * sum_phi
                * current_service.bandwidth
            )

            # Potência de ASE no span
            power_ase = (
                current_service.bandwidth
                * h_plank
                * current_service.center_frequency
                * (exp(2.0 * span.attenuation_normalized * span.length * 1e3) - 1.0)
                * span.noise_figure_normalized
            )

            # Somatório para GSNR, ASE e NLI
            #   --> 1 / (SNR) = 1 / (P_signal / P_ruído) = P_ruído / P_signal
            #       mas P_signal = current_service.launch_power
            #   --> SNR_total = launch_power / (power_ase + power_nli_span)
            #   --> SNR_ase   = launch_power / power_ase
            #   --> SNR_nli   = launch_power / power_nli_span
            acc_gsnr += 1.0 / (current_service.launch_power / (power_ase + power_nli_span))
            acc_ase  += 1.0 / (current_service.launch_power / power_ase)
            acc_nli  += 1.0 / (current_service.launch_power / power_nli_span)

    # Converte cada acúmulo para dB
    gsnr = 10.0 * np.log10(1.0 / acc_gsnr)
    ase =  10.0 * np.log10(1.0 / acc_ase)
    nli =  10.0 * np.log10(1.0 / acc_nli)

    return gsnr, ase, nli


#############################
# 2) calculate_osnr_default_attenuation
#############################
cpdef calculate_osnr_default_attenuation(
    env: QRMSAEnv,
    current_service: object,
    attenuation_normalized: cython.double,
    noise_figure_normalized: cython.double
):
    cdef double beta_2 = -21.3e-27
    cdef double gamma = 1.3e-3
    cdef double h_plank = 6.626e-34
    cdef double acc_gsnr = 0.0
    cdef double acc_ase = 0.0
    cdef double acc_nli = 0.0
    cdef double gsnr = 0.0
    cdef double ase = 0.0
    cdef double nli = 0.0
    cdef double l_eff_a = 0.0
    cdef double l_eff = 0.0
    cdef double phi = 0.0
    cdef double sum_phi = 0.0
    cdef double power_ase = 0.0
    cdef double power_nli_span = 0.0

    # Formato de modulação
    cdef np.ndarray[np.double_t, ndim=1] phi_modulation_format = np.array(
        [1.0, 1.0, 2.0/3.0, 17.0/25.0, 69.0/100.0, 13.0/21.0],
        dtype=np.float64
    )

    cdef object link
    cdef object span
    cdef object running_service

    # Percorre cada link do caminho do serviço
    for link in current_service.path.links:
        for span in link.spans:
            # Cálculo da L_eff e L_eff_a
            l_eff_a = 1.0 / (2.0 * attenuation_normalized)
            l_eff = (
                1.0 - np.exp(-2.0 * attenuation_normalized * span.length * 1e3)
            ) / (2.0 * attenuation_normalized)

            # Inicia sum_phi para este span
            sum_phi = asinh(
                pi**2 * abs(beta_2) * (current_service.bandwidth**2)
                / (4.0 * attenuation_normalized)
            )

            # Contribuição NLI de outros serviços
            for running_service in env.topology[link.node1][link.node2]["running_services"]:
                if running_service.service_id != current_service.service_id:
                    phi = (
                        asinh(
                            pi**2 * abs(beta_2) * l_eff_a * running_service.bandwidth
                            * (
                                running_service.center_frequency
                                - current_service.center_frequency
                                + (running_service.bandwidth / 2.0)
                            )
                        )
                        - asinh(
                            pi**2 * abs(beta_2) * l_eff_a * running_service.bandwidth
                            * (
                                running_service.center_frequency
                                - current_service.center_frequency
                                - (running_service.bandwidth / 2.0)
                            )
                        )
                    ) - (
                        phi_modulation_format[running_service.current_modulation.spectral_efficiency - 1]
                        * (
                            running_service.bandwidth
                            / abs(running_service.center_frequency - current_service.center_frequency)
                        )
                        * (5.0 / 3.0)
                        * (l_eff / (span.length * 1e3))
                    )
                    sum_phi += phi

            # Potência de NLI e ASE
            power_nli_span = (
                (current_service.launch_power / current_service.bandwidth)**3
                * (8.0 / (27.0 * pi * abs(beta_2)))
                * (gamma**2)
                * l_eff
                * sum_phi
                * current_service.bandwidth
            )
            power_ase = (
                current_service.bandwidth
                * h_plank
                * current_service.center_frequency
                * (exp(2.0 * attenuation_normalized * span.length * 1e3) - 1.0)
                * noise_figure_normalized
            )

            # Somatórios
            acc_gsnr += 1.0 / (current_service.launch_power / (power_ase + power_nli_span))
            acc_ase  += 1.0 / (current_service.launch_power / power_ase)
            acc_nli  += 1.0 / (current_service.launch_power / power_nli_span)

    # Converte para dB
    gsnr = 10.0 * np.log10(1.0 / acc_gsnr)
    ase =  10.0 * np.log10(1.0 / acc_ase)
    nli =  10.0 * np.log10(1.0 / acc_nli)

    return gsnr, ase, nli


#############################
# 3) calculate_osnr_observation
#############################
cpdef double calculate_osnr_observation(
    object env,  # QRMSAEnv 
    tuple path_links,  # Lista de  Link
    double service_bandwidth,
    double service_center_frequency,
    int service_id,
    double service_launch_power,
    double gsnr_th
):
    cdef double beta_2 = -21.3e-27
    cdef double gamma = 1.3e-3
    cdef double h_plank = 6.626e-34
    cdef double acc_gsnr = 0.0
    cdef double acc_ase = 0.0   # se quiser usar
    cdef double acc_nli = 0.0   # se quiser usar
    cdef double gsnr = 0.0
    cdef double l_eff_a = 0.0
    cdef double l_eff = 0.0
    cdef double phi = 0.0
    cdef double sum_phi = 0.0
    cdef double power_ase = 0.0
    cdef double power_nli_span = 0.0

    # Modulation array
    cdef np.ndarray[np.double_t, ndim=1] phi_modulation_format = np.array(
        [1.0, 1.0, 2.0/3.0, 17.0/25.0, 69.0/100.0, 13.0/21.0],
        dtype=np.float64
    )

    cdef object link
    cdef object span
    cdef object running_service

    # Percorre cada link do path
    for link in path_links:
        for span in env.topology[link.node1][link.node2]["link"].spans:
            l_eff_a = 1.0 / (2.0 * span.attenuation_normalized)
            l_eff = (
                1.0 - exp(-2.0 * span.attenuation_normalized * span.length * 1e3)
            ) / (2.0 * span.attenuation_normalized)

            # sum_phi para este span
            sum_phi = asinh(
                pi**2
                * abs(beta_2)
                * (service_bandwidth**2)
                / (4.0 * span.attenuation_normalized)
            )

            # Contribuição dos serviços em execução
            for running_service in env.topology[link.node1][link.node2]["running_services"]:
                if running_service.service_id != service_id:
                    phi = (
                        asinh(
                            pi**2
                            * abs(beta_2)
                            * l_eff_a
                            * running_service.bandwidth
                            * (
                                running_service.center_frequency
                                - service_center_frequency
                                + (running_service.bandwidth / 2.0)
                            )
                        )
                        - asinh(
                            pi**2
                            * abs(beta_2)
                            * l_eff_a
                            * running_service.bandwidth
                            * (
                                running_service.center_frequency
                                - service_center_frequency
                                - (running_service.bandwidth / 2.0)
                            )
                        )
                    ) - (
                        phi_modulation_format[running_service.current_modulation.spectral_efficiency - 1]
                        * (
                            running_service.bandwidth
                            / abs(running_service.center_frequency - service_center_frequency)
                        )
                        * (5.0 / 3.0)
                        * (l_eff / (span.length * 1e3))
                    )
                    sum_phi += phi

            # Potência NLI e ASE
            power_nli_span = (
                (service_launch_power / service_bandwidth)**3
                * (8.0 / (27.0 * pi * abs(beta_2)))
                * (gamma**2)
                * l_eff
                * sum_phi
                * service_bandwidth
            )
            power_ase = (
                service_bandwidth
                * h_plank
                * service_center_frequency
                * (exp(2.0 * span.attenuation_normalized * span.length * 1e3) - 1.0)
                * span.noise_figure_normalized
            )

            # Somatório
            acc_gsnr += 1.0 / (service_launch_power / (power_ase + power_nli_span))

    # GSNR final
    gsnr = 10.0 * log10(1.0 / acc_gsnr)
    # Normalização
    cdef double normalized_gsnr = np.round((gsnr - gsnr_th) / abs(gsnr_th), 5)
    return normalized_gsnr
