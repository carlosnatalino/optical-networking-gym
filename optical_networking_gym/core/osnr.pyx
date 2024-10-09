from math import pi

from libc.math cimport log, exp, asinh, log10

import cython
import numpy as np

import typing

if typing.TYPE_CHECKING:
    from optical_networking_gym.envs.qrmsa import QRMSAEnv, Service, Span, Link


cpdef calculate_osnr(env: QRMSAEnv, current_service: Service):
    beta_2: cython.double = -21.3e-27  
    gamma: cython.double = 1.3e-3  
    h_plank: cython.double = 6.626e-34  
    acc_gsnr: cython.double = 0
    acc_ase: cython.double = 0
    acc_nli: cython.double = 0
    gsnr: cython.double = 0
    ase: cython.double = 0
    nli: cython.double = 0
    l_eff_a: cython.double = 0
    l_eff: cython.double = 0
    phi: cython.double = 0
    sum_phi: cython.double = 0
    power_ase: cython.double = 0
    power_nli_span: cython.double = 0
    phi_modulation_format = np.array((1, 1, 2/3, 17/25, 69/100, 13/21))
    service: Service
    link: Link
    span: Span

    # print("#"*30)
    # print("Service:", current_service)
    # acc_gsnr = 0
    for link in current_service.path.links:
        # print("\tLink:", link)
        # node1 = env.current_service.path.node_list[index]
        # node2 = env.current_service.path.node_list[index + 1]
        # gsnr_link = 0
        for span in env.topology[link.node1][link.node2]["link"].spans:
            # print("\t\tSpan:", span)
            l_eff_a = 1 / (2 * span.attenuation_normalized)
            # print(f"\t\t\t{l_eff_a=}")
            l_eff = (
                1 - np.exp(-2 * span.attenuation_normalized * span.length * 1e3)
            ) / (
                2 * span.attenuation_normalized
            )
            # print(f"\t\t\t{l_eff=}")

            # calculate SCI
            sum_phi = asinh(
                pi ** 2 * \
                abs(beta_2) * \
                (current_service.bandwidth) ** 2 / \
                (4 * span.attenuation_normalized)
            )

            # print(f"\t\t\t{sum_phi=}")

            for service in env.topology[link.node1][link.node2]["running_services"]:
                # if service.center_frequency - current_service.center_frequency == 0:
                #     print(service)
                #     print(current_service)
                if service.service_id != current_service.service_id:
                    # print(f"\t\t\t\t{service=}")
                    phi = (
                        asinh(
                            pi ** 2 * \
                            abs(beta_2) * \
                            l_eff_a * \
                            service.bandwidth * \
                            (  # TODO: double-check this part below
                                service.center_frequency - \
                                current_service.center_frequency + \
                                (service.bandwidth / 2)
                            )
                        ) - \
                        asinh(
                            pi ** 2 * \
                            abs(beta_2) * \
                            l_eff_a * \
                            service.bandwidth * \
                            (  # TODO: double-check this part below
                                service.center_frequency - \
                                current_service.center_frequency - \
                                (service.bandwidth / 2)
                            )
                        )
                    ) - \
                    (
                        phi_modulation_format[service.current_modulation.spectral_efficiency - 1] * \
                        (
                            service.bandwidth / \
                            abs(service.center_frequency - current_service.center_frequency)
                        ) * \
                        5 / 3 * \
                        (l_eff / (span.length * 1e3))
                    )
                    # print(f"\t\t\t\t{phi=}")
                sum_phi += phi
                # print(f"\t\t\t\t{sum_phi=}")

            power_nli_span = (current_service.launch_power / (current_service.bandwidth)) ** 3 * \
            (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff * sum_phi * current_service.bandwidth
            power_ase = current_service.bandwidth * h_plank * current_service.center_frequency * \
                (exp(2 * span.attenuation_normalized * span.length * 1e3) - 1) * span.noise_figure_normalized

            acc_gsnr = acc_gsnr + 1 / (current_service.launch_power / (power_ase + power_nli_span))
            acc_ase = acc_ase + 1 / (env.current_service.launch_power / power_ase)
            acc_nli = acc_nli + 1 / (env.current_service.launch_power / power_nli_span)

    gsnr = 10 * np.log10(1 / acc_gsnr)
    ase = 10 * np.log10(1 / acc_ase)
    nli = 10 * np.log10(1 / acc_nli)
    return gsnr, ase, nli


cpdef calculate_osnr_default_attenuation(env: QRMSAEnv, current_service: Service, attenuation_normalized: cython.double, noise_figure_normalized: cython.double):
    beta_2: cython.double = -21.3e-27  # group velocity dispersion (s^2/m)
    gamma: cython.double = 1.3e-3  # nonlinear parameter 1/(W.m)
    h_plank: cython.double = 6.626e-34  # Planck's constant (J s)
    acc_gsnr: cython.double = 0
    acc_ase: cython.double = 0
    acc_nli: cython.double = 0
    gsnr: cython.double = 0
    ase: cython.double = 0
    nli: cython.double = 0
    # gsnr_link: cython.double = 0
    l_eff_a: cython.double = 0
    l_eff: cython.double = 0
    phi: cython.double = 0
    sum_phi: cython.double = 0
    phi_modulation_format = np.array((1, 1, 2/3, 17/25, 69/100, 13/21))
    service: Service
    link: Link
    span: Span

    # acc_gsnr = 0
    for link in current_service.path.links:
        # node1 = env.current_service.path.node_list[index]
        # node2 = env.current_service.path.node_list[index + 1]
        # gsnr_link = 0

        for span in link.spans:
            
            l_eff_a = 1 / (2 * attenuation_normalized)
            l_eff = (
                1 - np.exp(-2 * attenuation_normalized * span.length * 1e3)
            ) / (
                2 * attenuation_normalized
            )

            # calculate SCI
            sum_phi = asinh(
                pi ** 2 * \
                abs(beta_2) * \
                (current_service.bandwidth) ** 2 / \
                (4 * attenuation_normalized)
            )

            for service in env.topology[link.node1][link.node2]["running_services"]:
                if service.center_frequency - current_service.center_frequency == 0:
                    print(service)
                    print(current_service)
                if service.service_id != current_service.service_id:
                    phi = (
                        asinh(
                            pi ** 2 * \
                            abs(beta_2) * \
                            l_eff_a * \
                            service.bandwidth * \
                            (  # TODO: double-check this part below
                                service.center_frequency - \
                                current_service.center_frequency + \
                                (service.bandwidth / 2)
                            )
                        ) - \
                        asinh(
                            pi ** 2 * \
                            abs(beta_2) * \
                            l_eff_a * \
                            service.bandwidth * \
                            (  # TODO: double-check this part below
                                service.center_frequency - \
                                current_service.center_frequency - \
                                (service.bandwidth / 2)
                            )
                        )
                    ) - \
                    (
                        phi_modulation_format[service.current_modulation.spectral_efficiency - 1] * \
                        (
                            service.bandwidth / \
                            abs(service.center_frequency - current_service.center_frequency)
                        ) * \
                        5 / 3 * \
                        (l_eff / (span.length * 1e3))
                    )
                sum_phi += phi

            power_nli_span = (env.current_service.launch_power / (env.current_service.bandwidth)) ** 3 * \
            (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff * sum_phi * (env.current_service.bandwidth)
            power_ase = env.current_service.bandwidth * h_plank * env.current_service.center_frequency * \
                (exp(2 * attenuation_normalized * span.length * 1e3) - 1) * noise_figure_normalized

            # need:
            # GSNR_Path_AseNLI = GSNR_Path_AseNLI + (Ptx/(P_ASE+P_NLI_span))^-1;
            acc_gsnr = acc_gsnr + 1 / (env.current_service.launch_power / (power_ase + power_nli_span))
            acc_ase = acc_gsnr + 1 / (env.current_service.launch_power / power_ase)
            acc_nli = acc_gsnr + 1 / (env.current_service.launch_power / power_nli_span)

            # gnsr_span_db = 10 * log10(env.current_service.launch_power / (power_nli_span + power_ase))
    # need:
    # GSNR_LP_dB = 10*log10(GSNR_Path_AseNLI.^-1);
    gsnr = 10 * np.log10(1 / acc_gsnr)
    ase = 10 * np.log10(1 / acc_ase)
    nli = 10 * np.log10(1 / acc_nli)
    return gsnr, ase, nli
