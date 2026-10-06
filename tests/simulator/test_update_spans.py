"""``Simulator.update_spans`` / ``OpticalEnv.update_spans`` (network aging) and
the request bookkeeping of counter-only resets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optical_networking_gym import (
    QoTEngine,
    RequestAnalysisEngine,
    Simulator,
    SpanUpdate,
    Status,
    TopologyModel,
    build_scenario,
)
from optical_networking_gym.contracts.enums import MaskMode
from optical_networking_gym.defaults import resolve_topology
from optical_networking_gym.envs import OpticalEnv
from optical_networking_gym.runtime.action_codec import decode_action


@pytest.fixture(scope="module")
def topology() -> TopologyModel:
    config = build_scenario("jocn_benchmark")
    return TopologyModel.from_file(
        resolve_topology(config.topology_id),
        topology_id=config.topology_id,
        k_paths=config.k_paths,
        max_span_length_km=config.max_span_length_km,
    )


def _simulator(topology: TopologyModel, *, episode_length: int = 50, **overrides: object) -> Simulator:
    values: dict[str, object] = dict(episode_length=episode_length, load=140.0)
    values.update(overrides)
    config = build_scenario("jocn_benchmark", **values)
    return Simulator(config, topology, episode_length=episode_length)


def _first_valid_action(simulator: Simulator) -> int:
    mask = simulator.get_trace_action_mask()
    valid = np.flatnonzero(mask[:-1])
    return int(valid[0]) if valid.size else simulator.total_actions - 1


def _link_updates(topology: TopologyModel, link_ids, **values: float) -> list[SpanUpdate]:
    return [
        SpanUpdate(link_id, span_index, **values)
        for link_id in link_ids
        for span_index in range(len(topology.links[link_id].spans))
    ]


def _current_gsnr(simulator: Simulator) -> np.ndarray:
    assert simulator.current_analysis is not None
    return np.asarray(simulator.current_analysis.gsnr_db_by_start)


# --------------------------------------------------------------------------
# Propagation and exactness
# --------------------------------------------------------------------------


def test_update_propagates_to_every_helper(topology: TopologyModel) -> None:
    simulator = _simulator(topology)
    simulator.reset(seed=1)
    assert simulator.physical_layer_version == 0
    assert simulator.base_topology is topology

    aged = simulator.update_spans([SpanUpdate(2, 0, attenuation_db_per_km=0.25)])

    assert aged is simulator.topology
    assert aged is not topology
    assert aged.links[2].spans[0].attenuation_db_per_km == 0.25
    assert simulator.physical_layer_version == 1
    assert simulator.base_topology is topology
    for helper in (
        simulator.qot_engine,
        simulator.analysis_engine,
        simulator.action_mask_builder,
        simulator.observation_builder,
        simulator.reward_function,
        simulator.state,
        simulator.traffic_model,
    ):
        assert helper.topology is aged, type(helper).__name__

    assert simulator.update_spans([]) is aged
    assert simulator.physical_layer_version == 1


def test_updates_match_a_fresh_engine_in_the_same_state(topology: TopologyModel) -> None:
    simulator = _simulator(topology, episode_length=200)
    simulator.reset(seed=3)
    rng = np.random.default_rng(5)
    applied: list[SpanUpdate] = []
    for step in range(120):
        simulator.step(_first_valid_action(simulator))
        if step % 15 == 7:
            updates = []
            for _ in range(4):
                link_id = int(rng.integers(topology.link_count))
                span_index = int(rng.integers(len(topology.links[link_id].spans)))
                updates.append(
                    SpanUpdate(link_id, span_index, float(rng.uniform(0.18, 0.3)), float(rng.uniform(4.0, 7.0)))
                )
            simulator.update_spans(updates, refresh_active_services=bool(step % 2))
            applied.extend(updates)
    state = simulator.state
    assert state is not None and len(state.active_services_by_id) > 10

    aged = topology.with_span_updates(applied)
    fresh_engine = QoTEngine(simulator.config, aged)
    for service_id in sorted(state.active_services_by_id):
        assert simulator.qot_engine.recompute_service(state, service_id) == fresh_engine.recompute_service(
            state, service_id
        )
        ours = simulator.qot_engine.service_noise_breakdown(state, service_id)
        theirs = fresh_engine.service_noise_breakdown(state, service_id)
        for name in ours.__slots__:
            assert np.array_equal(getattr(ours, name), getattr(theirs, name)), name

    request = simulator.current_request
    assert request is not None
    fresh_analysis = RequestAnalysisEngine(simulator.config, aged, fresh_engine).build(state, request)
    analysis = simulator.current_analysis
    assert analysis is not None
    for name in ("gsnr_db_by_start", "osnr_margin_by_start", "qot_valid_starts", "resource_valid_starts"):
        assert np.array_equal(getattr(analysis, name), getattr(fresh_analysis, name), equal_nan=True), name


def test_analysis_cache_never_serves_an_older_physical_layer(topology: TopologyModel) -> None:
    simulator = _simulator(topology)
    simulator.reset(seed=2)
    request = simulator.current_request
    assert request is not None and simulator.state is not None
    before = simulator.analysis_engine.build(simulator.state, request)
    path = before.paths[0]
    # Bypass the simulator: changing the engine alone must invalidate analyses.
    aged = topology.with_span_updates(_link_updates(topology, path.link_ids, attenuation_db_per_km=0.4))
    simulator.qot_engine.set_topology(aged, changed_link_ids=path.link_ids)
    after = simulator.analysis_engine.build(simulator.state, request)
    assert after is not before
    assert np.nanmax(after.gsnr_db_by_start[0]) < np.nanmax(before.gsnr_db_by_start[0])


# --------------------------------------------------------------------------
# Pending request
# --------------------------------------------------------------------------


def test_pending_request_is_reanalysed(topology: TopologyModel) -> None:
    config = build_scenario(
        "jocn_benchmark", episode_length=10, load=140.0, mask_mode=MaskMode.RESOURCE_AND_QOT
    )
    env = OpticalEnv(config, topology, episode_length=10)
    hook_calls: list[object] = []
    env.simulator.request_analysed_callback = hook_calls.append
    _, info = env.reset(seed=0)
    mask = info["mask"]
    analysis = env.simulator.current_analysis
    assert analysis is not None
    valid = [int(action) for action in np.flatnonzero(mask[:-1])]
    on_path_0 = [action for action in valid if decode_action(config, action).path_index == 0]
    assert on_path_0
    assert len(hook_calls) == 1

    path = analysis.paths[0]
    env.update_spans(_link_updates(topology, path.link_ids, attenuation_db_per_km=1.0))

    new_mask = env.action_masks()
    assert new_mask is not None
    assert int(new_mask[:-1].sum()) < len(valid)
    assert not any(new_mask[action] for action in on_path_0)
    assert env.simulator.current_analysis is not analysis
    assert len(hook_calls) == 1  # the hook is not called again

    _, _, _, _, info = env.step(on_path_0[0])
    assert info["status"] == Status.BLOCKED_QOT.value


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        SpanUpdate(10_000, 0, attenuation_db_per_km=0.25),
        SpanUpdate(0, 50, attenuation_db_per_km=0.25),
        SpanUpdate(0, 0, attenuation_db_per_km=-1.0),
        SpanUpdate(0, 0, noise_figure_db=0.0),
        SpanUpdate(0, 0, noise_figure_db=math.nan),
        SpanUpdate(0, 0, attenuation_db_per_km=math.inf),
    ],
)
def test_invalid_updates_change_nothing(topology: TopologyModel, bad: SpanUpdate) -> None:
    simulator = _simulator(topology)
    simulator.reset(seed=4)
    for _ in range(10):
        simulator.step(_first_valid_action(simulator))
    state = simulator.state
    assert state is not None
    snapshot = (
        simulator.topology,
        simulator.physical_layer_version,
        simulator.current_analysis,
        simulator.current_mask,
        simulator.current_observation,
        dict(state.service_qot_by_id),
        dict(simulator.qot_engine._path_summary_static_cache),
        simulator.qot_engine.topology_version,
    )
    with pytest.raises(ValueError):
        simulator.update_spans([SpanUpdate(1, 0, attenuation_db_per_km=0.3), bad], refresh_active_services=True)
    assert (
        simulator.topology,
        simulator.physical_layer_version,
        simulator.current_analysis,
        simulator.current_mask,
        simulator.current_observation,
        dict(state.service_qot_by_id),
        dict(simulator.qot_engine._path_summary_static_cache),
        simulator.qot_engine.topology_version,
    ) == snapshot


# --------------------------------------------------------------------------
# Lifetime
# --------------------------------------------------------------------------


def test_lifetime_across_resets(topology: TopologyModel) -> None:
    simulator = _simulator(topology, episode_length=5)
    simulator.reset(seed=9)
    original_gsnr = _current_gsnr(simulator).copy()
    path = simulator.current_analysis.paths[0]
    aged = simulator.update_spans(_link_updates(topology, path.link_ids, noise_figure_db=8.0))
    aged_gsnr = _current_gsnr(simulator).copy()
    assert np.nanmax(aged_gsnr[0]) < np.nanmax(original_gsnr[0])

    for _ in range(5):
        simulator.step(simulator.total_actions - 1)
    simulator.reset(options={"only_episode_counters": True})
    assert simulator.topology is aged
    assert simulator.qot_engine.topology is aged
    assert simulator.physical_layer_version == 1

    simulator.reset(seed=9, options={"keep_physical_layer": True})
    assert simulator.topology is aged
    assert simulator.state is not None and simulator.state.topology is aged
    assert simulator.physical_layer_version == 1
    assert np.array_equal(_current_gsnr(simulator), aged_gsnr, equal_nan=True)

    simulator.reset(seed=9)
    assert simulator.topology is topology
    assert simulator.qot_engine.topology is topology
    assert simulator.state is not None and simulator.state.topology is topology
    assert simulator.physical_layer_version == 0
    assert np.array_equal(_current_gsnr(simulator), original_gsnr, equal_nan=True)


# --------------------------------------------------------------------------
# Active services
# --------------------------------------------------------------------------


def _loaded(topology: TopologyModel, **overrides: object) -> Simulator:
    simulator = _simulator(topology, episode_length=200, **overrides)
    simulator.reset(seed=11)
    for _ in range(40):
        simulator.step(_first_valid_action(simulator))
    assert simulator.state is not None and simulator.state.active_services_by_id
    return simulator


def _lowest_margin_service(simulator: Simulator) -> int:
    state = simulator.state
    assert state is not None
    return min(
        state.active_services_by_id,
        key=lambda service_id: state.active_services_by_id[service_id].osnr
        - state.active_services_by_id[service_id].modulation.minimum_osnr,
    )


def test_active_services_keep_their_qot_by_default(topology: TopologyModel) -> None:
    simulator = _loaded(topology)
    state = simulator.state
    assert state is not None
    service_id = _lowest_margin_service(simulator)
    service = state.active_services_by_id[service_id]
    stored = dict(state.service_qot_by_id)
    simulator.update_spans(_link_updates(topology, service.path.link_ids, attenuation_db_per_km=0.3))
    assert state.service_qot_by_id == stored


def test_refresh_updates_the_qot_of_services_on_changed_links(topology: TopologyModel) -> None:
    simulator = _loaded(topology)
    state = simulator.state
    assert state is not None
    service_id = _lowest_margin_service(simulator)
    link_ids = state.active_services_by_id[service_id].path.link_ids
    on_links = {sid for link_id in link_ids for sid in state.link_active_service_ids[link_id]}
    stored = dict(state.service_qot_by_id)

    simulator.update_spans(
        _link_updates(topology, link_ids, attenuation_db_per_km=0.3), refresh_active_services=True
    )

    assert service_id in state.active_services_by_id  # measure_disruptions is off
    for sid, qot in state.service_qot_by_id.items():
        if sid in on_links:
            assert qot[0] < stored[sid][0]
            assert qot == (lambda u: (u.osnr, u.ase, u.nli))(simulator.qot_engine.recompute_service(state, sid))
        else:
            assert qot == stored[sid]


@pytest.mark.parametrize("drop", [False, True], ids=["disrupt", "drop"])
def test_aging_disrupts_or_drops_a_low_margin_service(topology: TopologyModel, drop: bool) -> None:
    simulator = _loaded(topology, measure_disruptions=True, drop_on_disruption=drop)
    state, statistics = simulator.state, simulator.statistics
    assert state is not None and statistics is not None
    service_id = _lowest_margin_service(simulator)
    link_ids = state.active_services_by_id[service_id].path.link_ids
    disrupted_before = statistics.disrupted_services
    dropped_before = statistics.services_dropped_qot

    simulator.update_spans(
        _link_updates(topology, link_ids, attenuation_db_per_km=0.6), refresh_active_services=True
    )

    assert statistics.disrupted_services > disrupted_before
    if drop:
        assert statistics.services_dropped_qot > dropped_before
        assert service_id not in state.active_services_by_id
        assert service_id in state.disrupted_services_by_id
    else:
        assert statistics.services_dropped_qot == dropped_before
        assert state.active_services_by_id[service_id].is_disrupted
    # The pending request sees the post-disruption state.
    request = simulator.current_request
    assert request is not None
    fresh = RequestAnalysisEngine(
        simulator.config, simulator.topology, QoTEngine(simulator.config, simulator.topology)
    ).build(state, request)
    assert np.array_equal(_current_gsnr(simulator), fresh.gsnr_db_by_start, equal_nan=True)


# --------------------------------------------------------------------------
# Counter-only reset after a terminated episode (A.3)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["reject", "accept"])
def test_consecutive_episodes_process_each_request_once(topology: TopologyModel, policy: str) -> None:
    simulator = _simulator(topology, episode_length=5, mask_mode=MaskMode.RESOURCE_ONLY)
    simulator.reset(seed=0)
    analysed: list[int] = []
    simulator.request_analysed_callback = lambda analysis: analysed.append(analysis.request.request_index)
    processed: list[int] = []
    for episode in range(3):
        if episode:
            simulator.reset(options={"only_episode_counters": True})
        terminated = False
        while not terminated:
            assert simulator.current_request is not None
            processed.append(simulator.current_request.request_index)
            action = simulator.total_actions - 1 if policy == "reject" else _first_valid_action(simulator)
            _, _, terminated, _, info = simulator.step(action)
        assert info["episode_services_processed"] == 5
    assert processed == list(range(15))
    assert analysed == list(range(1, 15))  # request 0 was analysed before the hook was set


def test_update_after_termination_applies_to_the_next_request(topology: TopologyModel) -> None:
    simulator = _simulator(topology, episode_length=3)
    simulator.reset(seed=0)
    for _ in range(3):
        _, _, terminated, _, _ = simulator.step(simulator.total_actions - 1)
    assert terminated
    stale = simulator.current_analysis
    path = stale.paths[0]
    simulator.update_spans(_link_updates(topology, path.link_ids, attenuation_db_per_km=0.3))
    assert simulator.current_analysis is stale  # processed request: not re-analysed

    simulator.reset(options={"only_episode_counters": True})
    request = simulator.current_request
    assert request is not None and request.request_index == 3
    state = simulator.state
    assert state is not None
    fresh = RequestAnalysisEngine(
        simulator.config, simulator.topology, QoTEngine(simulator.config, simulator.topology)
    ).build(state, request)
    assert np.array_equal(_current_gsnr(simulator), fresh.gsnr_db_by_start, equal_nan=True)
