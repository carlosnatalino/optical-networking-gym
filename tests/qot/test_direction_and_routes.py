"""Undirected physical model, route identity of the engine caches, and the
split of the per-link NLI into self- and cross-channel interference.

* Lightpaths are bidirectional and the engine is undirected: the GSNR of a
  route is computed once, in its canonical direction (lower node index first),
  whatever the source of the request. CFM2, the only direction-dependent part
  of the model, accumulates dispersion along that direction for every channel.
* The engine and the runtime state identify a path by its links, so records
  that reuse an id (sub-paths, external planners) are evaluated correctly.
* ``LightpathNoiseBreakdown`` reports the SCI and XCI of every link.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from optical_networking_gym import (
    Modulation,
    QoTEngine,
    RequestAnalysisEngine,
    RuntimeState,
    ScenarioConfig,
    ServiceRequest,
    TopologyModel,
)
from optical_networking_gym.defaults import BUILTIN_TOPOLOGY_DIR
from optical_networking_gym.network import PathRecord
from optical_networking_gym.network.topology import _reverse_path_record
from optical_networking_gym.optical import cfm2
from optical_networking_gym.optical.kernels import qot_kernel as compiled_kernel

_TWIN_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "optical_networking_gym" / "optical" / "kernels" / "qot_kernel.py"
)


def _load_python_twin():
    spec = importlib.util.spec_from_file_location("qot_kernel_python_twin_direction", _TWIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


python_kernel = _load_python_twin()
KERNELS = pytest.mark.parametrize("kernel", [compiled_kernel, python_kernel], ids=["compiled", "python"])

NOBEL_EU = BUILTIN_TOPOLOGY_DIR / "nobel-eu.xml"
QPSK = Modulation("QPSK", 100_000.0, 2, minimum_osnr=6.72)
QAM16 = Modulation("16QAM", 100_000.0, 4, minimum_osnr=13.24)
SLOT = 12.5e9
F_START = 3e8 / 1565e-9
ALPHA = 0.2 / (2 * 10 * math.log10(math.e) * 1e3)


@pytest.fixture(scope="module")
def topology() -> TopologyModel:
    return TopologyModel.from_file(NOBEL_EU, k_paths=2, max_span_length_km=80.0)


def _config(**overrides: object) -> ScenarioConfig:
    values: dict[str, object] = dict(
        scenario_id="direction",
        topology_id="nobel-eu",
        k_paths=2,
        num_spectrum_resources=320,
        nli_include_interferers=True,
        nli_modulation_correction="cfm2",
        modulations=(QPSK, QAM16),
    )
    values.update(overrides)
    return ScenarioConfig(**values)  # type: ignore[arg-type]


def _asymmetric_path(topology: TopologyModel) -> PathRecord:
    """A long path whose links have different span counts, so the order in
    which they are travelled changes the CFM2 distances."""
    return max(
        (
            path
            for path in topology.paths
            if len({len(topology.links[link_id].spans) for link_id in path.link_ids}) > 1
        ),
        key=lambda path: path.hops,
    )


def _request(service_id: int, path: PathRecord, *, backwards: bool = False) -> ServiceRequest:
    source, destination = path.node_indices[0], path.node_indices[-1]
    if backwards:
        source, destination = destination, source
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=source,
        destination_id=destination,
        bit_rate=100,
        arrival_time=float(service_id),
        holding_time=1e6,
    )


def _provision(
    engine: QoTEngine,
    state: RuntimeState,
    service_id: int,
    path: PathRecord,
    start: int,
    *,
    backwards: bool = False,
    modulation: Modulation = QPSK,
) -> None:
    request = _request(service_id, path, backwards=backwards)
    candidate = engine.build_candidate(request, path, modulation, start, 4)
    state.apply_provision(
        request=request,
        path=path,
        service_slot_start=start,
        service_num_slots=4,
        occupied_slot_start=start,
        occupied_slot_end_exclusive=start + 5,
        modulation=modulation,
        center_frequency=candidate.center_frequency,
        bandwidth=candidate.bandwidth,
        launch_power=candidate.launch_power,
    )


def _loaded(config: ScenarioConfig, topology: TopologyModel, path: PathRecord):
    """Interferers on ``path`` travelling in both directions."""
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    for index, (start, backwards) in enumerate(((40, False), (46, True), (58, True), (70, False))):
        _provision(engine, state, 100 + index, path, start, backwards=backwards)
    return engine, state


def _osnr(engine: QoTEngine, state: RuntimeState, path: PathRecord, *, backwards: bool, modulation=QAM16) -> float:
    candidate = engine.build_candidate(_request(0, path, backwards=backwards), path, modulation, 52, 4)
    return engine.evaluate_candidate(state, candidate).osnr


# --------------------------------------------------------------------------
# Undirected model
# --------------------------------------------------------------------------


def test_topology_paths_are_canonical(topology: TopologyModel) -> None:
    # The engine evaluates every route from its lower-index endpoint, the
    # direction of the k-shortest path records, which it therefore uses as is.
    assert all(path.node_indices[0] < path.node_indices[-1] for path in topology.paths)


@pytest.mark.parametrize("correction", ["egn_xci", "cfm2", "gn"])
def test_gsnr_does_not_depend_on_the_direction(topology: TopologyModel, correction: str) -> None:
    path = _asymmetric_path(topology)
    engine, state = _loaded(_config(nli_modulation_correction=correction), topology, path)
    forwards = _osnr(engine, state, path, backwards=False)
    assert _osnr(engine, state, path, backwards=True) == forwards
    # The same route stored backwards (same id, links reversed) is canonicalised.
    flipped = _reverse_path_record(path)
    assert _osnr(engine, state, flipped, backwards=False) == forwards
    assert _osnr(engine, state, flipped, backwards=True) == forwards


def test_interferers_accumulate_dispersion_along_the_canonical_direction(topology: TopologyModel) -> None:
    path = _asymmetric_path(topology)
    config = _config()
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    _provision(engine, state, 1, path, 40, backwards=False)
    _provision(engine, state, 2, path, 60, backwards=True)
    _provision(engine, state, 3, _reverse_path_record(path), 80)
    link_id = path.link_ids[1]
    rho = engine._link_running_service_arrays(state, link_id).rho.reshape(-1, 3)
    start_km = topology.links[path.link_ids[0]].length_km
    expected = cfm2.rho_interferer(1.0, 21.3 * (engine._link_span_start_km[link_id] + start_km))
    for column in range(3):
        np.testing.assert_allclose(rho[:, column], expected, rtol=1e-3)
    np.testing.assert_array_equal(rho[:, 0], rho[:, 1])
    np.testing.assert_array_equal(rho[:, 0], rho[:, 2])


def test_every_entry_point_is_direction_independent(topology: TopologyModel) -> None:
    path = _asymmetric_path(topology)
    config = _config(roadm_add_osnr_db=33.0, nli_coherence_epsilon=0.05)
    engine, state = _loaded(config, topology, path)
    flipped = _reverse_path_record(path)
    expected = _osnr(engine, state, path, backwards=False)
    candidate = engine.build_candidate(_request(0, path, backwards=True), flipped, QAM16, 52, 4)
    assert engine.evaluate_candidate(state, candidate).osnr == expected
    assert engine.summarize_candidate(state, candidate).osnr == expected
    for route in (path, flipped):
        at = engine.summarize_candidate_at(
            state=state, service_id=0, path=route, modulation=QAM16, service_slot_start=52, service_num_slots=4
        )
        assert at.osnr == expected
        batch = engine.summarize_candidate_starts(
            state=state, service_id=0, path=route, modulation=QAM16, service_num_slots=4, candidate_starts=[52]
        )
        assert batch.osnr_margin[0] == at.osnr_margin
        breakdown = engine.noise_breakdown(
            state,
            path=route,
            service_id=0,
            center_frequency=candidate.center_frequency,
            bandwidth=candidate.bandwidth,
            launch_power=candidate.launch_power,
            modulation=QAM16,
        )
        assert breakdown.total_nsr == pytest.approx(10 ** (-expected / 10), rel=1e-12)
        assert breakdown.link_ids == path.link_ids  # canonical order
    # An established service recomputes to the same GSNR, whichever end its
    # request came from and whichever orientation its record has.
    for route, backwards in ((path, True), (flipped, False)):
        engine, state = _loaded(config, topology, path)
        _provision(engine, state, 0, route, 52, backwards=backwards, modulation=QAM16)
        assert engine.recompute_service(state, 0).osnr == expected
        assert engine.service_noise_breakdown(state, 0).gsnr_db == pytest.approx(expected, abs=1e-9)


def test_request_analysis_is_direction_independent(topology: TopologyModel) -> None:
    path = _asymmetric_path(topology)
    config = _config(k_paths=2)
    engine, state = _loaded(config, topology, path)
    analysis_engine = RequestAnalysisEngine(config, topology, engine)
    forwards = analysis_engine.build(state, _request(0, path, backwards=False))
    backwards = analysis_engine.build(state, _request(0, path, backwards=True))
    np.testing.assert_array_equal(forwards.osnr_margin_by_start, backwards.osnr_margin_by_start)
    np.testing.assert_array_equal(forwards.qot_valid_starts, backwards.qot_valid_starts)


# --------------------------------------------------------------------------
# Route identity
# --------------------------------------------------------------------------


def test_path_from_link_ids_follows_the_links(topology: TopologyModel) -> None:
    path = _asymmetric_path(topology)
    rebuilt = topology.path_from_link_ids(path.link_ids)
    assert rebuilt.link_ids == path.link_ids
    assert rebuilt.node_indices == path.node_indices
    assert rebuilt.node_names == path.node_names
    assert rebuilt.length_km == pytest.approx(path.length_km)
    assert (rebuilt.id, rebuilt.k, rebuilt.hops) == (-1, -1, path.hops)
    backwards = topology.path_from_link_ids(tuple(reversed(path.link_ids)), path_id=7)
    assert backwards.node_indices == tuple(reversed(path.node_indices))
    assert backwards.id == 7
    single = topology.path_from_link_ids([path.link_ids[0]])
    link = topology.links[path.link_ids[0]]
    assert single.node_indices == (link.source_index, link.target_index)


def test_path_from_link_ids_rejects_invalid_routes(topology: TopologyModel) -> None:
    path = _asymmetric_path(topology)
    with pytest.raises(ValueError, match="at least one link"):
        topology.path_from_link_ids([])
    with pytest.raises(ValueError, match="unknown link"):
        topology.path_from_link_ids([len(topology.links)])
    with pytest.raises(ValueError, match="do not share a node"):
        topology.path_from_link_ids([path.link_ids[0], path.link_ids[2]])


def test_engine_does_not_confuse_paths_that_share_an_id(topology: TopologyModel) -> None:
    path = _asymmetric_path(topology)
    other = next(candidate for candidate in topology.paths if candidate.link_ids != path.link_ids)
    sub = topology.path_from_link_ids(path.link_ids[:2], path_id=other.id)
    for correction in ("egn_xci", "cfm2"):
        config = _config(nli_modulation_correction=correction, roadm_express_osnr_db=35.0)
        engine, state = _loaded(config, topology, path)
        _osnr(engine, state, other, backwards=False)  # caches `other` under its id
        reused = _osnr(engine, state, sub, backwards=False)
        fresh_engine = QoTEngine(config, topology)
        assert reused == _osnr(fresh_engine, state, sub, backwards=False)
        assert reused != _osnr(engine, state, other, backwards=False)


def test_runtime_state_does_not_confuse_paths_that_share_an_id(topology: TopologyModel) -> None:
    path = _asymmetric_path(topology)
    other = next(candidate for candidate in topology.paths if not set(candidate.link_ids) & set(path.link_ids))
    sub = topology.path_from_link_ids(path.link_ids[:2], path_id=other.id)
    config = _config()
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    _provision(engine, state, 1, other, 10)
    _provision(engine, state, 2, sub, 30)
    for link_id in sub.link_ids:
        assert np.all(state.slot_allocation[link_id, 30:35] == 2)
    for link_id in other.link_ids:
        assert np.all(state.slot_allocation[link_id, 30:35] == -1)


def test_noise_breakdown_accepts_link_ids(topology: TopologyModel) -> None:
    path = _asymmetric_path(topology)
    engine, state = _loaded(_config(roadm_express_osnr_db=35.0), topology, path)
    kwargs = dict(service_id=0, center_frequency=F_START + SLOT * 54, bandwidth=4 * SLOT, launch_power=1e-3, modulation=QPSK)
    sub_links = path.link_ids[1:]
    by_links = engine.noise_breakdown(state, link_ids=sub_links, **kwargs)
    by_path = engine.noise_breakdown(state, path=topology.path_from_link_ids(sub_links), **kwargs)
    assert by_links.link_ids == sub_links
    assert by_links.total_nsr == by_path.total_nsr
    np.testing.assert_array_equal(by_links.link_nli_nsr, by_path.link_nli_nsr)
    # Links listed backwards describe the same (undirected) route.
    backwards = engine.noise_breakdown(state, link_ids=tuple(reversed(path.link_ids)), **kwargs)
    forwards = engine.noise_breakdown(state, path=path, **kwargs)
    assert backwards.link_ids == path.link_ids
    assert backwards.total_nsr == forwards.total_nsr
    np.testing.assert_array_equal(backwards.link_xci_nsr, forwards.link_xci_nsr)
    with pytest.raises(ValueError, match="exactly one"):
        engine.noise_breakdown(state, path=path, link_ids=path.link_ids, **kwargs)
    with pytest.raises(ValueError, match="exactly one"):
        engine.noise_breakdown(state, **kwargs)


# --------------------------------------------------------------------------
# SCI / XCI split
# --------------------------------------------------------------------------


@pytest.mark.parametrize("correction", ["egn_xci", "cfm2", "gn"])
@pytest.mark.parametrize("psd", ["cut", "actual"])
def test_sci_and_xci_add_up_to_the_link_nli(topology: TopologyModel, correction: str, psd: str) -> None:
    path = _asymmetric_path(topology)
    engine, state = _loaded(_config(nli_modulation_correction=correction, nli_interferer_psd=psd), topology, path)
    breakdown = engine.noise_breakdown(
        state,
        path=path,
        service_id=0,
        center_frequency=F_START + SLOT * 54,
        bandwidth=4 * SLOT,
        launch_power=1e-3,
        modulation=QPSK,
    )
    np.testing.assert_allclose(breakdown.link_sci_nsr + breakdown.link_xci_nsr, breakdown.link_nli_nsr, rtol=1e-12)
    assert np.all(breakdown.link_sci_nsr > 0.0)
    assert np.all(breakdown.link_xci_nsr > 0.0)
    if correction != "egn_xci":
        # No span is clipped: the correction is only the rounding of gsnr - ase.
        assert abs(breakdown.nli_correction_nsr) <= 1e-12 * breakdown.total_nsr


def test_xci_is_zero_without_interferers(topology: TopologyModel) -> None:
    path = _asymmetric_path(topology)
    for include in (False, True):
        config = _config(nli_include_interferers=include)
        engine = QoTEngine(config, topology)
        breakdown = engine.noise_breakdown(
            RuntimeState(config, topology),
            path=path,
            service_id=0,
            center_frequency=F_START + SLOT * 54,
            bandwidth=4 * SLOT,
            launch_power=1e-3,
            modulation=QPSK,
        )
        assert np.all(breakdown.link_xci_nsr == 0.0)
        np.testing.assert_array_equal(breakdown.link_sci_nsr, breakdown.link_nli_nsr)


def test_sci_scales_with_the_square_of_the_launch_power(topology: TopologyModel) -> None:
    # SCI NSR ~ P^2; with the interferers' actual PSD, the XCI NSR does not
    # depend on the CUT power.
    path = _asymmetric_path(topology)
    engine, state = _loaded(_config(nli_interferer_psd="actual"), topology, path)
    kwargs = dict(path=path, service_id=0, center_frequency=F_START + SLOT * 54, bandwidth=4 * SLOT, modulation=QPSK)
    low = engine.noise_breakdown(state, launch_power=1e-3, **kwargs)
    high = engine.noise_breakdown(state, launch_power=2e-3, **kwargs)
    np.testing.assert_allclose(high.link_sci_nsr, 4.0 * low.link_sci_nsr, rtol=1e-12)
    np.testing.assert_allclose(high.link_xci_nsr, low.link_xci_nsr, rtol=1e-12)


@KERNELS
def test_split_leaves_the_other_outputs_unchanged(kernel) -> None:
    inputs = _kernel_inputs(np.random.default_rng(24))
    for extra in ({}, {"cfm2": True, "cut_phi": 1.0, "running_rho": inputs["rho"]}):
        plain = _path_noise(kernel, inputs, **extra)
        split = _path_noise(kernel, inputs, split_nli=True, **extra)
        assert len(plain) == 7 and len(split) == 9
        for got, expected in zip(split[:7], plain):
            np.testing.assert_array_equal(got, expected)
        np.testing.assert_allclose(split[7] + split[8], plain[0] - plain[1], rtol=1e-12)


def test_split_matches_between_kernels() -> None:
    inputs = _kernel_inputs(np.random.default_rng(25))
    n = inputs["lengths"].shape[0]
    # Non-nominal spans: power offsets and connector losses.
    offsets = np.random.default_rng(26).uniform(-1.0, 1.0, (n, 320))
    for extra in ({}, {"cfm2": True, "cut_phi": 17 / 25, "running_rho": inputs["rho"]}):
        compiled = _path_noise(compiled_kernel, inputs, split_nli=True, power_offsets=offsets, **extra)
        python = _path_noise(python_kernel, inputs, split_nli=True, power_offsets=offsets, **extra)
        for got, expected in zip(compiled, python):
            np.testing.assert_allclose(got, expected, rtol=1e-12)


# --------------------------------------------------------------------------
# Kernel helpers (two links: 3 + 2 spans, 4 + 2 interferers)
# --------------------------------------------------------------------------

SPAN_OFFSETS = np.array([0, 3, 5], dtype=np.int32)
RUNNING_OFFSETS = np.array([0, 4, 6], dtype=np.int32)


def _kernel_inputs(rng: np.random.Generator) -> dict[str, np.ndarray]:
    slots = rng.choice(np.arange(10, 300, 6), size=6, replace=False)
    widths = rng.integers(2, 6, size=6)
    return {
        "lengths": rng.uniform(60.0, 100.0, 5),
        "ids": np.arange(100, 106, dtype=np.int32),
        "freqs": F_START + SLOT * slots + SLOT * widths / 2.0,
        "bw": SLOT * widths.astype(np.float64),
        "phi": rng.choice([1.0, 2 / 3, 17 / 25], size=6),
        "powers": 10 ** ((rng.uniform(-6, 0, size=6) - 30) / 10),
        "rho": rng.uniform(0.05, 0.95, size=3 * 4 + 2 * 2),
    }


def _path_noise(kernel, inputs: dict[str, np.ndarray], power_offsets: np.ndarray | None = None, **extra):
    n = inputs["lengths"].shape[0]
    return kernel.path_noise(
        SPAN_OFFSETS,
        inputs["lengths"],
        np.full(n, ALPHA),
        np.full(n, 10**0.55),
        np.full(n, 10**0.05),
        np.ones(n),
        np.zeros((n, 0)) if power_offsets is None else power_offsets,
        RUNNING_OFFSETS,
        inputs["ids"],
        inputs["freqs"],
        inputs["bw"],
        inputs["phi"],
        inputs["powers"],
        current_service_id=0,
        center_frequency=F_START + SLOT * 152,
        bandwidth=SLOT * 4,
        launch_power=5e-4,
        include_nli=True,
        frequency_start=F_START,
        frequency_slot_bandwidth=SLOT,
        interferer_psd_actual=True,
        nli_scale=1.1,
        extra_nsr=1e-4,
        **extra,
    )


def _summarize(kernel, inputs: dict[str, np.ndarray], **extra):
    n = inputs["lengths"].shape[0]
    return kernel.summarize_candidate_starts(
        SPAN_OFFSETS,
        inputs["lengths"],
        np.full(n, ALPHA),
        np.full(n, 10**0.55),
        RUNNING_OFFSETS,
        inputs["ids"],
        inputs["freqs"],
        inputs["bw"],
        inputs["phi"],
        np.array([0, 40, 120, 200, 290], dtype=np.int32),
        current_service_id=0,
        frequency_start=F_START,
        frequency_slot_bandwidth=SLOT,
        service_num_slots=4,
        launch_power=1e-3,
        threshold=10.0,
        include_nli=True,
        running_launch_powers=inputs["powers"],
        interferer_psd_actual=True,
        **extra,
    )
