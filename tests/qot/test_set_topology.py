"""``QoTEngine.set_topology``: runtime span updates must give exactly what a
freshly built engine computes on the updated topology, for both kernels, every
NLI modulation correction and with or without interferers."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from optical_networking_gym import (
    Modulation,
    QoTEngine,
    RuntimeState,
    ScenarioConfig,
    ServiceRequest,
    SpanUpdate,
    TopologyModel,
)
from optical_networking_gym.defaults import BUILTIN_TOPOLOGY_DIR
from optical_networking_gym.network import GainRipple, PathRecord
from optical_networking_gym.optical import qot_engine as qot_engine_module

_TWIN_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "optical_networking_gym" / "optical" / "kernels" / "qot_kernel.py"
)


def _load_python_twin():
    spec = importlib.util.spec_from_file_location("qot_kernel_python_twin_set_topology", _TWIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


python_kernel = _load_python_twin()

QPSK = Modulation("QPSK", 100_000.0, 2, minimum_osnr=6.72)
QAM16 = Modulation("16QAM", 100_000.0, 4, minimum_osnr=13.24)
F_START = 3e8 / 1565e-9
SLOT = 12.5e9
N_SLOTS = 320


@pytest.fixture(params=["compiled", "python"])
def kernel(request, monkeypatch) -> str:
    if request.param == "python":
        monkeypatch.setattr(qot_engine_module, "path_noise", python_kernel.path_noise)
        monkeypatch.setattr(
            qot_engine_module, "summarize_candidate_starts", python_kernel.summarize_candidate_starts
        )
    return request.param


@pytest.fixture(scope="module")
def topology() -> TopologyModel:
    """nobel-eu with lumped losses and gain ripple on some links, so that every
    per-link array of the engine is exercised."""
    base = TopologyModel.from_file(BUILTIN_TOPOLOGY_DIR / "nobel-eu.xml", k_paths=3, max_span_length_km=80.0)
    ripple = GainRipple.uniform(F_START, F_START + N_SLOTS * SLOT, [0.0, 0.4, -0.3, 0.2, -0.1])
    links = list(base.links)
    for link_id in range(0, len(links), 3):
        link = links[link_id]
        links[link_id] = replace(
            link,
            spans=tuple(
                replace(span, input_loss_db=0.5, output_loss_db=0.3, gain_ripple=ripple if index % 2 else None)
                for index, span in enumerate(link.spans)
            ),
        )
    return replace(base, links=tuple(links))


def _config(correction: str, interferers: bool) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="set_topology",
        topology_id="nobel-eu",
        k_paths=3,
        num_spectrum_resources=N_SLOTS,
        frequency_start=F_START,
        frequency_slot_bandwidth=SLOT,
        launch_power_dbm=-1.0,
        nli_include_interferers=interferers,
        nli_modulation_correction=correction,
        modulations=(QPSK, QAM16),
    )


def _request(service_id: int, path: PathRecord) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=path.node_indices[0],
        destination_id=path.node_indices[-1],
        bit_rate=100,
        arrival_time=float(service_id),
        holding_time=1e6,
    )


def _loaded_state(engine: QoTEngine, config: ScenarioConfig, topology: TopologyModel, paths: list[PathRecord]):
    """Lightpaths on disjoint slot ranges, so they never collide."""
    state = RuntimeState(config, topology)
    for index, path in enumerate(paths):
        start = 8 + 9 * index
        modulation = (QPSK, QAM16)[index % 2]
        request = _request(1000 + index, path)
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
    return state


def _random_updates(rng: np.random.Generator, topology: TopologyModel, count: int) -> list[SpanUpdate]:
    updates = []
    for _ in range(count):
        link_id = int(rng.integers(topology.link_count))
        span_index = int(rng.integers(len(topology.links[link_id].spans)))
        attenuation = float(rng.uniform(0.17, 0.32)) if rng.random() < 0.7 else None
        noise_figure = float(rng.uniform(4.0, 7.5)) if rng.random() < 0.7 else None
        updates.append(SpanUpdate(link_id, span_index, attenuation, noise_figure))
    return updates


def _observables(engine: QoTEngine, state: RuntimeState, paths: list[PathRecord]) -> list[object]:
    """Every QoT output of the public entry points for the given routes."""
    values: list[object] = []
    for index, path in enumerate(paths):
        modulation = (QPSK, QAM16)[index % 2]
        start = 200 + (index % 20)
        values.append(
            engine.summarize_candidate_at(
                state=state, service_id=0, path=path, modulation=modulation, service_slot_start=start, service_num_slots=4
            )
        )
        candidate = engine.build_candidate(_request(0, path), path, modulation, start, 4)
        values.append(engine.evaluate_candidate(state, candidate))
        breakdown = engine.noise_breakdown(
            state,
            path=path,
            service_id=0,
            center_frequency=candidate.center_frequency,
            bandwidth=candidate.bandwidth,
            launch_power=candidate.launch_power,
            modulation=modulation,
        )
        values.append(_breakdown_values(breakdown))
        batch = engine.summarize_candidate_starts(
            state=state,
            service_id=0,
            path=path,
            modulation=modulation,
            service_num_slots=4,
            candidate_starts=np.arange(150, 190, dtype=np.int32),
        )
        values.append(
            tuple(
                array.tobytes()
                for array in (batch.meets_threshold, batch.osnr_margin, batch.nli_share, batch.worst_link_nli_share)
            )
        )
    for service_id in sorted(state.active_services_by_id):
        values.append(engine.recompute_service(state, service_id))
        values.append(_breakdown_values(engine.service_noise_breakdown(state, service_id)))
    return values


def _breakdown_values(breakdown) -> tuple[object, ...]:
    return tuple(
        value.tobytes() if isinstance(value, np.ndarray) else value
        for value in (getattr(breakdown, name) for name in breakdown.__slots__)
    )


@pytest.mark.parametrize("interferers", [True, False], ids=["xci", "no_xci"])
@pytest.mark.parametrize("correction", ["egn_xci", "cfm2", "gn"])
def test_span_updates_match_a_fresh_engine(
    kernel: str, correction: str, interferers: bool, topology: TopologyModel
) -> None:
    rng = np.random.default_rng(1234)
    config = _config(correction, interferers)
    engine = QoTEngine(config, topology)
    all_paths = list(topology.paths)
    service_paths = [all_paths[int(i)] for i in rng.choice(len(all_paths), size=8, replace=False)]
    probe_paths = [all_paths[int(i)] for i in rng.choice(len(all_paths), size=40, replace=False)]
    state = _loaded_state(engine, config, topology, service_paths)

    # Fill the caches, then update in several rounds, evaluating in between so
    # that entries of intermediate physical layers are cached and evicted.
    _observables(engine, state, probe_paths)
    current = topology
    applied: list[SpanUpdate] = []
    for _ in range(4):
        updates = _random_updates(rng, topology, 6)
        current = current.with_span_updates(updates)
        engine.set_topology(current, changed_link_ids={update.link_id for update in updates})
        applied.extend(updates)
        _observables(engine, state, probe_paths[::2])

    fresh = QoTEngine(config, topology.with_span_updates(applied))
    assert _observables(engine, state, probe_paths) == _observables(fresh, state, probe_paths)
    # And a route that was never cached while the topology changed.
    assert _observables(engine, state, service_paths) == _observables(fresh, state, service_paths)

    # Restoring the original topology restores the original results.
    engine.set_topology(topology, changed_link_ids={update.link_id for update in applied})
    assert _observables(engine, state, probe_paths) == _observables(QoTEngine(config, topology), state, probe_paths)


def test_set_topology_without_changed_links_rereads_everything(topology: TopologyModel) -> None:
    config = _config("egn_xci", True)
    engine = QoTEngine(config, topology)
    paths = list(topology.paths[:30])
    state = RuntimeState(config, topology)
    _observables(engine, state, paths)
    aged = topology.with_span_updates(_random_updates(np.random.default_rng(7), topology, 10))
    engine.set_topology(aged)
    assert _observables(engine, state, paths) == _observables(QoTEngine(config, aged), state, paths)


def test_toggling_gain_ripple_rebuilds_every_route() -> None:
    """``set_topology`` also accepts gain-ripple changes; a topology that gains
    ripple changes the layout of every cached route."""
    flat = TopologyModel.from_file(BUILTIN_TOPOLOGY_DIR / "nobel-eu.xml", k_paths=2, max_span_length_km=80.0)
    ripple = GainRipple.uniform(F_START, F_START + N_SLOTS * SLOT, [0.0, 0.5, -0.5])
    link = flat.links[0]
    rippled = replace(
        flat,
        links=(replace(link, spans=tuple(replace(span, gain_ripple=ripple) for span in link.spans)),)
        + flat.links[1:],
    )
    config = _config("egn_xci", False)
    engine = QoTEngine(config, flat)
    state = RuntimeState(config, flat)
    paths = list(flat.paths[:40])
    _observables(engine, state, paths)
    engine.set_topology(rippled, changed_link_ids=[0])
    assert _observables(engine, state, paths) == _observables(QoTEngine(config, rippled), state, paths)
    engine.set_topology(flat, changed_link_ids=[0])
    assert _observables(engine, state, paths) == _observables(QoTEngine(config, flat), state, paths)


@pytest.mark.parametrize("change", ["attenuation", "noise_figure"])
def test_updated_span_lowers_the_gsnr_of_routes_through_it(topology: TopologyModel, change: str) -> None:
    config = _config("egn_xci", True)
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    path = max(topology.paths, key=lambda record: record.hops)
    link_id = path.link_ids[len(path.link_ids) // 2]
    others = [record for record in topology.paths if link_id not in record.link_ids][:50]
    before = _observables(engine, state, [path])
    others_before = _observables(engine, state, others)

    span = topology.links[link_id].spans[0]
    if change == "attenuation":
        update = SpanUpdate(link_id, 0, attenuation_db_per_km=span.attenuation_db_per_km + 3.0 / span.length_km)
    else:
        update = SpanUpdate(link_id, 0, noise_figure_db=span.noise_figure_db + 2.0)
    engine.set_topology(topology.with_span_updates([update]), changed_link_ids=[link_id])

    after = _observables(engine, state, [path])
    assert after[0].osnr < before[0].osnr  # summarize_candidate_at
    assert after[1].osnr < before[1].osnr  # evaluate_candidate
    assert _observables(engine, state, others) == others_before


def test_set_topology_validates_the_structure(topology: TopologyModel) -> None:
    config = _config("egn_xci", True)
    engine = QoTEngine(config, topology)
    other = TopologyModel.from_file(BUILTIN_TOPOLOGY_DIR / "nobel-eu.xml", k_paths=3, max_span_length_km=50.0)
    with pytest.raises(ValueError, match="span lengths"):
        engine.set_topology(other)
    ring = TopologyModel.from_file(BUILTIN_TOPOLOGY_DIR / "ring_4.txt", k_paths=2)
    with pytest.raises(ValueError, match="links"):
        engine.set_topology(ring)
    with pytest.raises(ValueError, match="unknown link id"):
        engine.set_topology(topology, changed_link_ids=[topology.link_count])
    assert engine.topology is topology
    assert engine.topology_version == 0
