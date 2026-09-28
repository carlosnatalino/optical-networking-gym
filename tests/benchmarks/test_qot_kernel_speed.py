"""Speed guards for the QoT kernel: the CFM2 modulation-format correction and
the SCI/XCI split must not slow the hot paths down, and the per-link evaluation
of the XCI terms must stay effective.

Timings are relative (same process, same inputs, best of N), so the guard does
not depend on the machine speed. It only runs against the compiled kernel.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import pytest

from optical_networking_gym.optical.kernels import qot_kernel

pytestmark = pytest.mark.skipif(
    not qot_kernel.__file__.endswith((".so", ".pyd")),
    reason="speed guard needs the compiled QoT kernel",
)

SLOT = 12.5e9
F_START = 3e8 / 1565e-9
ALPHA = 0.2 / (2 * 10 * math.log10(math.e) * 1e3)
MAX_SLOWDOWN = 1.25

# Realistic request-analysis sizes: 3 links x 8 spans, 60 interferers per link,
# every start slot of a 320-slot grid.
N_LINKS, SPANS_PER_LINK, RUNNING_PER_LINK = 3, 8, 60


def _inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    n_spans = N_LINKS * SPANS_PER_LINK
    n_running = N_LINKS * RUNNING_PER_LINK
    slots = np.concatenate(
        [rng.choice(np.arange(0, 310, 5), RUNNING_PER_LINK, replace=False) for _ in range(N_LINKS)]
    )
    return {
        "span_offsets": np.arange(0, n_spans + 1, SPANS_PER_LINK, dtype=np.int32),
        "lengths": rng.uniform(70.0, 100.0, n_spans),
        "attenuation": np.full(n_spans, ALPHA),
        "noise_figure": np.full(n_spans, 10**0.55),
        "running_offsets": np.arange(0, n_running + 1, RUNNING_PER_LINK, dtype=np.int32),
        "ids": np.arange(100, 100 + n_running, dtype=np.int32),
        "freqs": F_START + SLOT * slots + SLOT * 2.0,
        "bw": np.full(n_running, 4 * SLOT),
        "phi": rng.choice([1.0, 17 / 25, 13 / 21], n_running),
        "powers": np.full(n_running, 1e-3),
        "rho": rng.uniform(0.1, 0.9, n_spans * RUNNING_PER_LINK),
    }


def _best_time(fn: Callable[[], object], repeats: int, inner: int) -> float:
    fn()
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        best = min(best, (time.perf_counter() - start) / inner)
    return best


def _cfm2_kwargs(inputs: dict[str, np.ndarray]) -> dict[str, object]:
    return {"cfm2": True, "cut_phi": 17 / 25, "running_rho": inputs["rho"]}


def test_cfm2_candidate_batch_is_not_slower() -> None:
    inputs = _inputs()
    starts = np.arange(0, 316, 2, dtype=np.int32)

    def run(**extra: object) -> Callable[[], object]:
        return lambda: qot_kernel.summarize_candidate_starts(
            inputs["span_offsets"],
            inputs["lengths"],
            inputs["attenuation"],
            inputs["noise_figure"],
            inputs["running_offsets"],
            inputs["ids"],
            inputs["freqs"],
            inputs["bw"],
            inputs["phi"],
            starts,
            current_service_id=0,
            frequency_start=F_START,
            frequency_slot_bandwidth=SLOT,
            service_num_slots=4,
            launch_power=1e-3,
            threshold=10.0,
            include_nli=True,
            **extra,
        )

    legacy = _best_time(run(), repeats=7, inner=3)
    cfm2 = _best_time(run(**_cfm2_kwargs(inputs)), repeats=7, inner=3)
    assert cfm2 <= MAX_SLOWDOWN * legacy, f"CFM2 {cfm2 * 1e3:.2f} ms vs default {legacy * 1e3:.2f} ms"


def test_cfm2_single_channel_is_not_slower() -> None:
    inputs = _inputs()
    n_spans = inputs["lengths"].shape[0]

    def run(**extra: object) -> Callable[[], object]:
        return lambda: qot_kernel.path_noise(
            inputs["span_offsets"],
            inputs["lengths"],
            inputs["attenuation"],
            inputs["noise_figure"],
            np.ones(n_spans),
            np.ones(n_spans),
            np.zeros((n_spans, 0)),
            inputs["running_offsets"],
            inputs["ids"],
            inputs["freqs"],
            inputs["bw"],
            inputs["phi"],
            inputs["powers"],
            current_service_id=0,
            center_frequency=F_START + SLOT * 152,
            bandwidth=4 * SLOT,
            launch_power=1e-3,
            include_nli=True,
            frequency_start=F_START,
            frequency_slot_bandwidth=SLOT,
            interferer_psd_actual=False,
            **extra,
        )

    legacy = _best_time(run(), repeats=50, inner=20)
    cfm2 = _best_time(run(**_cfm2_kwargs(inputs)), repeats=50, inner=20)
    assert cfm2 <= MAX_SLOWDOWN * legacy, f"CFM2 {cfm2 * 1e6:.1f} us vs default {legacy * 1e6:.1f} us"


def _batch(inputs: dict[str, np.ndarray], attenuation: np.ndarray | None = None, **extra: object) -> Callable[[], object]:
    starts = np.arange(0, 316, 2, dtype=np.int32)
    return lambda: qot_kernel.summarize_candidate_starts(
        inputs["span_offsets"],
        inputs["lengths"],
        inputs["attenuation"] if attenuation is None else attenuation,
        inputs["noise_figure"],
        inputs["running_offsets"],
        inputs["ids"],
        inputs["freqs"],
        inputs["bw"],
        inputs["phi"],
        starts,
        current_service_id=0,
        frequency_start=F_START,
        frequency_slot_bandwidth=SLOT,
        service_num_slots=4,
        launch_power=1e-3,
        threshold=10.0,
        include_nli=True,
        **extra,
    )


def test_xci_terms_are_evaluated_once_per_link() -> None:
    # Uniform attenuation within each link (8 spans): the asinh terms are
    # evaluated once per link. A per-span attenuation pattern forces the
    # per-span evaluation, which must be clearly slower.
    inputs = _inputs()
    mixed = inputs["attenuation"] * np.tile([1.0, 1.001], inputs["attenuation"].shape[0] // 2)
    per_link = _best_time(_batch(inputs), repeats=7, inner=3)
    per_span = _best_time(_batch(inputs, attenuation=mixed), repeats=7, inner=3)
    assert per_link <= 0.5 * per_span, f"per link {per_link * 1e3:.2f} ms vs per span {per_span * 1e3:.2f} ms"


def test_nli_split_is_not_slower() -> None:
    inputs = _inputs()
    n_spans = inputs["lengths"].shape[0]

    def run(**extra: object) -> Callable[[], object]:
        return lambda: qot_kernel.path_noise(
            inputs["span_offsets"],
            inputs["lengths"],
            inputs["attenuation"],
            inputs["noise_figure"],
            np.ones(n_spans),
            np.ones(n_spans),
            np.zeros((n_spans, 0)),
            inputs["running_offsets"],
            inputs["ids"],
            inputs["freqs"],
            inputs["bw"],
            inputs["phi"],
            inputs["powers"],
            current_service_id=0,
            center_frequency=F_START + SLOT * 152,
            bandwidth=4 * SLOT,
            launch_power=1e-3,
            include_nli=True,
            frequency_start=F_START,
            frequency_slot_bandwidth=SLOT,
            interferer_psd_actual=False,
            **extra,
        )

    plain = _best_time(run(), repeats=50, inner=20)
    split = _best_time(run(split_nli=True), repeats=50, inner=20)
    assert split <= MAX_SLOWDOWN * plain, f"split {split * 1e6:.1f} us vs {plain * 1e6:.1f} us"
