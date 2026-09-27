"""Per-element noise breakdown, coherent NLI accumulation and node noise terms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optical_networking_gym import (
    Modulation,
    QoTEngine,
    RuntimeState,
    ScenarioConfig,
    ServiceRequest,
    TopologyModel,
)
from optical_networking_gym.defaults import BUILTIN_TOPOLOGY_DIR
from optical_networking_gym.network import PathRecord

NOBEL_EU = BUILTIN_TOPOLOGY_DIR / "nobel-eu.xml"
MODULATION = Modulation("QPSK", 100_000.0, 2, minimum_osnr=6.72)


@pytest.fixture(scope="module")
def topology() -> TopologyModel:
    return TopologyModel.from_file(NOBEL_EU, k_paths=2, max_span_length_km=80.0)


def _config(**overrides: object) -> ScenarioConfig:
    values: dict[str, object] = dict(
        scenario_id="breakdown",
        topology_id="nobel-eu",
        k_paths=2,
        num_spectrum_resources=320,
        nli_include_interferers=True,
        nli_interferer_psd="actual",
    )
    values.update(overrides)
    return ScenarioConfig(**values)  # type: ignore[arg-type]


def _long_path(topology: TopologyModel) -> PathRecord:
    return max(topology.paths, key=lambda path: path.hops)


def _request(service_id: int, path: PathRecord, topology: TopologyModel) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=topology.get_node_index(path.node_names[0]),
        destination_id=topology.get_node_index(path.node_names[-1]),
        bit_rate=100,
        arrival_time=float(service_id),
        holding_time=1e6,
        launch_power_dbm=-2.0 + (service_id % 3),
    )


def _loaded_state(
    engine: QoTEngine, config: ScenarioConfig, topology: TopologyModel, path: PathRecord
) -> RuntimeState:
    """Provision a few neighbours on the path so XCI is non-zero."""
    state = RuntimeState(config, topology)
    for index, start in enumerate((40, 48, 60, 72)):
        request = _request(100 + index, path, topology)
        candidate = engine.build_candidate(request, path, MODULATION, start, 4)
        state.apply_provision(
            request=request,
            path=path,
            service_slot_start=start,
            service_num_slots=4,
            occupied_slot_start=start,
            occupied_slot_end_exclusive=start + 5,
            modulation=MODULATION,
            center_frequency=candidate.center_frequency,
            bandwidth=candidate.bandwidth,
            launch_power=candidate.launch_power,
        )
    return state


def _breakdown(engine: QoTEngine, state: RuntimeState, topology: TopologyModel, path: PathRecord):
    request = _request(0, path, topology)
    candidate = engine.build_candidate(request, path, MODULATION, 52, 4)
    breakdown = engine.noise_breakdown(
        state,
        path=path,
        service_id=0,
        center_frequency=candidate.center_frequency,
        bandwidth=candidate.bandwidth,
        launch_power=candidate.launch_power,
    )
    return breakdown, engine.evaluate_candidate(state, candidate)


def test_breakdown_sums_exactly_to_the_gsnr(topology: TopologyModel) -> None:
    config = _config(
        nli_coherence_epsilon=0.05,
        roadm_add_osnr_db=33.0,
        roadm_drop_osnr_db=34.0,
        roadm_express_osnr_db=36.0,
        transceiver_osnr_db=25.0,
    )
    engine = QoTEngine(config, topology)
    path = _long_path(topology)
    state = _loaded_state(engine, config, topology, path)
    breakdown, result = _breakdown(engine, state, topology, path)

    parts = (
        breakdown.link_ase_nsr.sum()
        + breakdown.link_nli_nsr.sum()
        + breakdown.coherent_excess_nsr
        + breakdown.roadm_add_nsr
        + breakdown.roadm_express_nsr
        + breakdown.roadm_drop_nsr
        + breakdown.transceiver_nsr
        + breakdown.nli_correction_nsr
    )
    assert parts == pytest.approx(breakdown.total_nsr, rel=1e-12)
    assert breakdown.gsnr_db == pytest.approx(result.osnr, abs=1e-12)
    assert breakdown.link_ids == path.link_ids
    assert breakdown.roadm_express_nsr == pytest.approx((path.hops - 1) * 10 ** -3.6)
    assert breakdown.transceiver_nsr == pytest.approx(10 ** -2.5)
    assert breakdown.coherent_excess_nsr == pytest.approx(
        (breakdown.n_spans**0.05 - 1.0)
        * (breakdown.link_nli_nsr.sum() + breakdown.nli_correction_nsr),
        rel=1e-12,
    )


def test_defaults_add_no_node_or_coherent_terms(topology: TopologyModel) -> None:
    config = _config()
    engine = QoTEngine(config, topology)
    path = _long_path(topology)
    breakdown, _ = _breakdown(engine, RuntimeState(config, topology), topology, path)
    assert breakdown.coherent_excess_nsr == 0.0
    assert breakdown.roadm_add_nsr == breakdown.transceiver_nsr == 0.0


def test_coherent_accumulation_and_node_terms_lower_gsnr(topology: TopologyModel) -> None:
    path = _long_path(topology)
    gsnr = {}
    for name, overrides in {
        "base": {},
        "coherent": {"nli_coherence_epsilon": 0.1},
        "nodes": {"roadm_express_osnr_db": 30.0},
    }.items():
        config = _config(**overrides)
        engine = QoTEngine(config, topology)
        gsnr[name] = _breakdown(engine, RuntimeState(config, topology), topology, path)[0].gsnr_db
    assert gsnr["coherent"] < gsnr["base"]
    assert gsnr["nodes"] < gsnr["base"]


def test_batch_mask_path_matches_scalar_with_all_terms(topology: TopologyModel) -> None:
    config = _config(nli_coherence_epsilon=0.05, roadm_add_osnr_db=33.0, transceiver_osnr_db=25.0)
    engine = QoTEngine(config, topology)
    path = _long_path(topology)
    state = _loaded_state(engine, config, topology, path)
    starts = [0, 52, 90, 200]
    batch = engine.summarize_candidate_starts(
        state=state,
        service_id=0,
        path=path,
        modulation=MODULATION,
        service_num_slots=4,
        candidate_starts=starts,
        launch_power=1e-3,
    )
    scalar = [
        engine.summarize_candidate_at(
            state=state,
            service_id=0,
            path=path,
            modulation=MODULATION,
            service_slot_start=start,
            service_num_slots=4,
            launch_power=1e-3,
        ).osnr_margin
        for start in starts
    ]
    np.testing.assert_allclose(batch.osnr_margin, scalar, rtol=1e-10)


def test_interferers_can_be_included_without_disruption_tracking(topology: TopologyModel) -> None:
    path = _long_path(topology)
    with_xci = _config(nli_include_interferers=True)
    without_xci = _config(nli_include_interferers=False)
    results = []
    for config in (with_xci, without_xci):
        engine = QoTEngine(config, topology)
        state = _loaded_state(engine, config, topology, path)
        results.append(_breakdown(engine, state, topology, path)[0])
    assert not with_xci.measure_disruptions
    assert results[0].link_nli_nsr.sum() > results[1].link_nli_nsr.sum()


def test_epsilon_is_validated() -> None:
    with pytest.raises(ValueError):
        _config(nli_coherence_epsilon=1.5)
    with pytest.raises(ValueError):
        _config(roadm_add_osnr_db=math.inf)
