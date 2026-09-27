"""CFM2 modulation-format correction factors for the closed-form GN model.

The closed-form GN model (CFM1) evaluated by ``kernels/qot_kernel`` assumes
Gaussian-distributed signals and therefore overestimates the nonlinear
interference (NLI) of real constellations. CFM2 [RanjbarZefreh_2020_AccurateClosedFormRealTime]
turns it into an approximation of the EGN model [Carena_2014_EGNModelNonlinear]
by multiplying, in every span ``n``, the self-channel term (SCI) by
``rho_CUT`` and each cross-channel term (XCI) by ``rho_nch`` (Eqs. (2), (11)):

    G_NLI^(n) ∝ G_CUT * (rho_CUT^(n) * G_CUT^2 * I_CUT + sum_nch 2 rho_nch^(n) * G_nch^2 * I_nch)

    rho_nch^(n) = a1 + a2 * Phi_nch^a3 + a4 * Phi_nch^a5 * (1 + a6 * (|b2acc(n, nch)| + a7)^a8)
    rho_CUT^(n) = a9 + a10 * Phi_CUT^a11
                  + a12 * Phi_CUT^a13 * (1 + a14 * R_CUT^a15 + a16 * (|b2acc(n, CUT)| + a17)^a18)

where ``Phi`` is the EGN modulation-format constant (``2 - E|a|^4 / E^2|a|^2``,
0 for Gaussian signals), ``R_CUT`` the CUT symbol rate in TBaud and
``b2acc(n, ch) = sum_{k<n} beta2 * L_k`` the dispersion (ps^2) accumulated by
channel ``ch`` from its transmitter to the input of span ``n``. The
coefficients ``a1..a18`` were fitted to the EGN model over 8500 randomised
C-band systems (32-128 GBaud, 80-120 km spans; Table II of the paper).

Network use: CFM2 was derived for a single line where every channel enters at
its start. In a mesh network each channel's ``b2acc`` is taken from its own
transmitter, measured along the link order of its path record, since that is
the dispersion that has made its signal Gaussian-like. The occupied bandwidth
(slots x slot width) stands in for the symbol rate, as in the rest of the QoT
model, and ``beta2`` is the kernel's constant ``|beta2|`` (no dispersion slope).
The coherence (CFM3) and roll-off (CFM4) refinements are not modelled.
"""

from __future__ import annotations

import numpy as np

# Table II of [RanjbarZefreh_2020_AccurateClosedFormRealTime], a1..a18.
CFM2_COEFFICIENTS: tuple[float, ...] = (
    +9.3143e-1,
    -7.7122e-1,
    +9.1090e-1,
    -1.4555e1,
    +8.5816e-1,
    -9.9415e-1,
    +1.0812e0,
    +5.2247e-3,
    +9.9313e-1,
    -1.8838e0,
    +6.2974e-1,
    -1.1421e1,
    +6.7368e-1,
    -1.1759e0,
    +6.4482e-3,
    +1.8738e5,
    +1.9527e3,
    -2.0016e0,
)

# |beta2| of the QoT kernel (21.3e-27 s^2/m) in ps^2/km, the unit of Eq. (11).
ABS_BETA2_PS2_PER_KM = 21.3

# EGN constant Phi per spectral efficiency (bits/symbol/polarisation) of the
# PM-QAM family: Table I of [RanjbarZefreh_2020_AccurateClosedFormRealTime].
PHI_BY_SPECTRAL_EFFICIENCY: dict[int, float] = {
    1: 1.0,  # PM-BPSK
    2: 1.0,  # PM-QPSK
    3: 2.0 / 3.0,  # PM-8QAM
    4: 17.0 / 25.0,  # PM-16QAM
    5: 69.0 / 100.0,  # PM-32QAM
    6: 13.0 / 21.0,  # PM-64QAM
    7: 1105.0 / 1681.0,  # PM-128QAM
    8: 257.0 / 425.0,  # PM-256QAM
}


def rho_interferer(phi: np.ndarray | float, accumulated_dispersion_ps2: np.ndarray | float) -> np.ndarray:
    """XCI factor ``rho_nch`` of Eq. (11) (vectorised, broadcasting)."""
    a1, a2, a3, a4, a5, a6, a7, a8 = CFM2_COEFFICIENTS[:8]
    phi_arr = np.asarray(phi, dtype=np.float64)
    dispersion = np.abs(np.asarray(accumulated_dispersion_ps2, dtype=np.float64))
    return a1 + a2 * phi_arr**a3 + a4 * phi_arr**a5 * (1.0 + a6 * (dispersion + a7) ** a8)


def rho_cut(phi: float, symbol_rate_tbaud: float, accumulated_dispersion_ps2: float) -> float:
    """SCI factor ``rho_CUT`` of Eq. (11) for one span (scalar)."""
    a9, a10, a11, a12, a13, a14, a15, a16, a17, a18 = CFM2_COEFFICIENTS[8:]
    return (
        a9
        + a10 * phi**a11
        + a12
        * phi**a13
        * (1.0 + a14 * symbol_rate_tbaud**a15 + a16 * (abs(accumulated_dispersion_ps2) + a17) ** a18)
    )


__all__ = [
    "ABS_BETA2_PS2_PER_KM",
    "CFM2_COEFFICIENTS",
    "PHI_BY_SPECTRAL_EFFICIENCY",
    "rho_cut",
    "rho_interferer",
]

# References
# [RanjbarZefreh_2020_AccurateClosedFormRealTime] M. Ranjbar Zefreh, F. Forghieri,
#     S. Piciaccia, and P. Poggiolini, "Accurate closed-form real-time EGN model
#     formula leveraging machine-learning over 8500 thoroughly randomized full
#     C-band systems," J. Lightw. Technol., vol. 38, no. 18, pp. 4987-4999, 2020,
#     doi: 10.1109/JLT.2020.2997395.
# [Carena_2014_EGNModelNonlinear] A. Carena, G. Bosco, V. Curri, Y. Jiang,
#     P. Poggiolini, and F. Forghieri, "EGN model of non-linear fiber
#     propagation," Opt. Express, vol. 22, no. 13, pp. 16335-16362, 2014,
#     doi: 10.1364/OE.22.016335.
