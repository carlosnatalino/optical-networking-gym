"""Tests for the generalised QoT kernel: per-channel launch power, interferer PSD
weighting, lumped span losses, per-span channel-power offsets, and the
path-level coherence/extra-noise terms. The compiled kernel must agree with its
pure-Python twin, and default arguments must reproduce the historical model."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from optical_networking_gym.optical.kernels import qot_kernel as compiled_kernel

_TWIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "optical_networking_gym"
    / "optical"
    / "kernels"
    / "qot_kernel.py"
)


def _load_python_twin():
    spec = importlib.util.spec_from_file_location("qot_kernel_python_twin", _TWIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


python_kernel = _load_python_twin()

SLOT = 12.5e9
F_START = 3e8 / 1565e-9
ALPHA = 0.2 / (2 * 10 * math.log10(math.e) * 1e3)


def _link(n_spans: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.linspace(60.0, 80.0, n_spans)
    attenuation = np.full(n_spans, ALPHA)
    noise_figure = np.full(n_spans, 10 ** (5.5 / 10))
    return lengths, attenuation, noise_figure


def _running(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    slots = rng.choice(np.arange(10, 300, 6), size=n, replace=False)
    widths = rng.integers(2, 6, size=n)
    return {
        "ids": np.arange(100, 100 + n, dtype=np.int32),
        "freqs": F_START + SLOT * slots + SLOT * widths / 2.0,
        "bw": SLOT * widths.astype(np.float64),
        "phi": rng.choice([1.0, 2.0 / 3.0, 17.0 / 25.0], size=n),
        "powers": 10 ** ((rng.uniform(-6, 0, size=n) - 30) / 10),
    }


def _call(kernel, *, running, launch_power=1e-3, center_slot=150, width=4, **extra):
    lengths, attenuation, noise_figure = _link()
    return kernel.accumulate_link_noise(
        lengths,
        attenuation,
        noise_figure,
        running["ids"],
        running["freqs"],
        running["bw"],
        running["phi"],
        current_service_id=0,
        center_frequency=F_START + SLOT * center_slot + SLOT * width / 2.0,
        bandwidth=SLOT * width,
        launch_power=launch_power,
        include_nli=True,
        frequency_start=F_START,
        frequency_slot_bandwidth=SLOT,
        **extra,
    )


def _no_running() -> dict[str, np.ndarray]:
    return {
        "ids": np.empty(0, dtype=np.int32),
        "freqs": np.empty(0),
        "bw": np.empty(0),
        "phi": np.empty(0),
        "powers": np.empty(0),
    }


def test_compiled_kernel_is_in_use() -> None:
    assert compiled_kernel.__file__.endswith((".so", ".pyd")), "Cython kernel not built"


def test_defaults_reproduce_historical_model_exactly() -> None:
    running = _running(8, np.random.default_rng(1))
    legacy = _call(compiled_kernel, running=running)
    explicit = _call(
        compiled_kernel,
        running=running,
        span_input_loss=np.ones(3),
        span_output_loss=np.ones(3),
        span_power_offset_db=np.zeros((3, 0)),
        running_launch_powers=running["powers"],
        interferer_psd_actual=False,
    )
    assert legacy == explicit


def _historical_link_noise(running, *, include_nli, launch_power=1e-3, center_slot=150, width=4):
    """Oracle: the incoherent link-noise loop of the kernel before the generalisation.

    Copied expression by expression (same operation order), so a default call of
    the generalised kernel must reproduce it bit for bit on the same platform.
    """
    lengths, attenuations, noise_figures = _link()
    center_frequency = F_START + SLOT * center_slot + SLOT * width / 2.0
    bandwidth = SLOT * width
    abs_beta_2 = abs(-21.3e-27)
    pi_squared = math.pi * math.pi
    prefactor_base = 8.0 / (27.0 * math.pi * abs_beta_2)
    nli_prefactor = ((launch_power / bandwidth) ** 3) * prefactor_base * (1.3e-3**2) * bandwidth
    acc_gsnr = acc_ase = acc_nli = 0.0
    for span_index, span_length_km in enumerate(lengths):
        span_length_m = span_length_km * 1e3
        attenuation = attenuations[span_index]
        power_nli_span = 0.0
        if include_nli:
            l_eff_a = 1.0 / (2.0 * attenuation)
            l_eff = (1.0 - math.exp(-2.0 * attenuation * span_length_m)) / (2.0 * attenuation)
            sum_phi = math.asinh(pi_squared * abs_beta_2 * (bandwidth**2) / (4.0 * attenuation))
            for index, service_id in enumerate(running["ids"]):
                if service_id == 0:
                    continue
                delta_frequency = running["freqs"][index] - center_frequency
                if delta_frequency == 0.0:
                    continue
                width_j = running["bw"][index]
                scale = pi_squared * abs_beta_2 * l_eff_a * width_j
                phi = (
                    math.asinh(scale * (delta_frequency + (width_j / 2.0)))
                    - math.asinh(scale * (delta_frequency - (width_j / 2.0)))
                ) - (
                    running["phi"][index]
                    * (width_j / abs(delta_frequency))
                    * (5.0 / 3.0)
                    * (l_eff / span_length_m)
                )
                sum_phi += phi
            power_nli_span = nli_prefactor * l_eff * sum_phi
        power_ase = (
            bandwidth
            * 6.626e-34
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
    return acc_gsnr, acc_ase, acc_nli


def _call_defaults(kernel, running, *, include_nli):
    """A call with only the historical arguments (all new ones at their defaults)."""
    return kernel.accumulate_link_noise(
        *_link(),
        running["ids"],
        running["freqs"],
        running["bw"],
        running["phi"],
        current_service_id=0,
        center_frequency=F_START + SLOT * 150 + SLOT * 4 / 2.0,
        bandwidth=SLOT * 4,
        launch_power=1e-3,
        include_nli=include_nli,
    )


@pytest.mark.parametrize("include_nli", [True, False])
def test_default_kernel_equals_historical_formula(include_nli: bool) -> None:
    """Guards the default path against drift, e.g. a reordered sum or pow -> x*x*x."""
    running = _running(8, np.random.default_rng(2))
    expected = _historical_link_noise(running, include_nli=include_nli)
    twin = _call_defaults(python_kernel, running, include_nli=include_nli)
    compiled = _call_defaults(compiled_kernel, running, include_nli=include_nli)
    # Same platform, pure Python on both sides: bit-identical.
    assert twin == expected
    # The C compiler may contract multiply-adds (FMA), so allow a few ulps.
    np.testing.assert_allclose(compiled, expected, rtol=1e-13, atol=0.0)


def test_actual_psd_with_equal_psd_matches_cut_mode() -> None:
    running = _running(6, np.random.default_rng(2))
    width = 4
    power = 1e-3
    # Equal PSD: every interferer launches power proportional to its bandwidth.
    equal_psd_powers = power * running["bw"] / (SLOT * width)
    cut_mode = _call(compiled_kernel, running=running, width=width, launch_power=power)
    actual = _call(
        compiled_kernel,
        running=running,
        width=width,
        launch_power=power,
        running_launch_powers=equal_psd_powers,
        interferer_psd_actual=True,
    )
    np.testing.assert_allclose(actual, cut_mode, rtol=1e-12)


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_compiled_matches_python_twin_on_heterogeneous_inputs(seed: int) -> None:
    rng = np.random.default_rng(seed)
    running = _running(10, rng)
    extra = dict(
        span_input_loss=10 ** (rng.uniform(0, 1.5, 3) / 10),
        span_output_loss=10 ** (rng.uniform(0, 1.5, 3) / 10),
        span_power_offset_db=rng.uniform(-0.5, 0.5, size=(3, 320)),
        running_launch_powers=running["powers"],
        interferer_psd_actual=True,
    )
    compiled = _call(compiled_kernel, running=running, **extra)
    python = _call(python_kernel, running=running, **extra)
    np.testing.assert_allclose(compiled, python, rtol=1e-12)


def test_ase_scales_inverse_and_sci_scales_square_with_power() -> None:
    running = _no_running()
    _, ase_1, nli_1 = _call(compiled_kernel, running=running, launch_power=1e-3)
    _, ase_2, nli_2 = _call(compiled_kernel, running=running, launch_power=2e-3)
    assert ase_2 == pytest.approx(ase_1 / 2.0, rel=1e-12)
    assert nli_2 == pytest.approx(nli_1 * 4.0, rel=1e-12)


def test_xci_weight_is_square_of_interferer_psd_ratio() -> None:
    running = _running(1, np.random.default_rng(6))
    width_cut = 4
    power = 1e-3
    base_power = power * running["bw"] / (SLOT * width_cut)  # same PSD as the CUT
    kwargs = dict(width=width_cut, launch_power=power, interferer_psd_actual=True)
    _, _, sci_only = _call(compiled_kernel, running=_no_running(), **kwargs)
    _, _, nli_same = _call(
        compiled_kernel, running=running, running_launch_powers=base_power, **kwargs
    )
    _, _, nli_double = _call(
        compiled_kernel, running=running, running_launch_powers=2 * base_power, **kwargs
    )
    assert nli_double - sci_only == pytest.approx(4.0 * (nli_same - sci_only), rel=1e-9)


def test_lumped_input_loss_raises_ase_and_lowers_nli() -> None:
    running = _no_running()
    loss_db = 1.0
    loss = 10 ** (loss_db / 10)
    _, ase_0, nli_0 = _call(compiled_kernel, running=running)
    _, ase_1, nli_1 = _call(
        compiled_kernel,
        running=running,
        span_input_loss=np.full(3, loss),
        span_output_loss=np.ones(3),
    )
    # ASE: the amplifier gain G_s grows by the lumped loss, P_ASE ~ (G_s - 1).
    lengths, _, _ = _link()
    gains = np.exp(2.0 * ALPHA * lengths * 1e3)
    expected = np.sum(gains * loss - 1.0) / np.sum(gains - 1.0)
    assert ase_1 / ase_0 == pytest.approx(expected, rel=1e-12)
    # NLI NSR ~ P_fibre^2 and P_fibre drops by the input loss.
    assert nli_1 / nli_0 == pytest.approx(1.0 / loss**2, rel=1e-12)


def test_power_offset_acts_like_a_local_launch_power_change() -> None:
    running = _no_running()
    offsets = np.zeros((3, 320))
    offsets[:, 150] = 3.0  # +3 dB at the CUT's centre slot on every span
    _, ase_0, nli_0 = _call(compiled_kernel, running=running, center_slot=148, width=4)
    _, ase_1, nli_1 = _call(
        compiled_kernel,
        running=running,
        center_slot=148,
        width=4,
        span_power_offset_db=offsets,
    )
    gain = 10 ** 0.3
    assert ase_1 == pytest.approx(ase_0 / gain, rel=1e-12)
    assert nli_1 == pytest.approx(nli_0 * gain**2, rel=1e-12)


def _summarize(kernel, **extra):
    lengths, attenuation, noise_figure = _link()
    running = _running(5, np.random.default_rng(7))
    return kernel.summarize_candidate_starts(
        np.array([0, 2, 3], dtype=np.int32),
        lengths,
        attenuation,
        noise_figure,
        np.array([0, 3, 5], dtype=np.int32),
        running["ids"],
        running["freqs"],
        running["bw"],
        running["phi"],
        np.array([40, 120, 200], dtype=np.int32),
        current_service_id=0,
        frequency_start=F_START,
        frequency_slot_bandwidth=SLOT,
        service_num_slots=4,
        launch_power=1e-3,
        threshold=10.0,
        include_nli=True,
        **extra,
    )


def test_summary_nli_scale_and_extra_nsr() -> None:
    base = _summarize(compiled_kernel)
    scaled = _summarize(compiled_kernel, nli_scale=1.3)
    extra = _summarize(compiled_kernel, extra_nsr=1e-3)
    # More NLI and extra noise both lower the margin; the NLI share grows with scale.
    assert np.all(scaled[1] < base[1])
    assert np.all(scaled[2] > base[2])
    assert np.all(extra[1] < base[1])
    for compiled, python in zip(
        _summarize(compiled_kernel, nli_scale=1.3, extra_nsr=1e-3),
        _summarize(python_kernel, nli_scale=1.3, extra_nsr=1e-3),
    ):
        np.testing.assert_allclose(compiled, python, rtol=1e-12)


def _path_noise(kernel, rng: np.random.Generator):
    lengths = np.concatenate([_link()[0], _link(2)[0]])
    n = lengths.shape[0]
    running = _running(6, rng)
    return kernel.path_noise(
        np.array([0, 3, 5], dtype=np.int32),
        lengths,
        np.full(n, ALPHA),
        np.full(n, 10 ** 0.55),
        10 ** (rng.uniform(0, 1.0, n) / 10),
        10 ** (rng.uniform(0, 1.0, n) / 10),
        rng.uniform(-0.3, 0.3, size=(n, 320)),
        np.array([0, 4, 6], dtype=np.int32),
        running["ids"],
        running["freqs"],
        running["bw"],
        running["phi"],
        running["powers"],
        current_service_id=0,
        center_frequency=F_START + SLOT * 152,
        bandwidth=SLOT * 4,
        launch_power=5e-4,
        include_nli=True,
        frequency_start=F_START,
        frequency_slot_bandwidth=SLOT,
        interferer_psd_actual=True,
        nli_scale=1.2,
        extra_nsr=2e-3,
    )


def test_path_noise_matches_python_twin_and_summary() -> None:
    compiled = _path_noise(compiled_kernel, np.random.default_rng(8))
    python = _path_noise(python_kernel, np.random.default_rng(8))
    for got, expected in zip(compiled, python):
        np.testing.assert_allclose(got, expected, rtol=1e-12)
    link_gsnr, link_ase, _, total, _, _, _ = compiled
    raw_nli = np.sum(link_gsnr - link_ase)
    assert total == pytest.approx(link_gsnr.sum() + 0.2 * raw_nli + 2e-3, rel=1e-12)


def test_summary_defaults_match_python_twin_exactly() -> None:
    for compiled, python in zip(_summarize(compiled_kernel), _summarize(python_kernel)):
        np.testing.assert_allclose(compiled, python, rtol=1e-12)
