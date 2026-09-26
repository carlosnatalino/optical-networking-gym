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


def test_summary_defaults_match_python_twin_exactly() -> None:
    for compiled, python in zip(_summarize(compiled_kernel), _summarize(python_kernel)):
        np.testing.assert_allclose(compiled, python, rtol=1e-12)
