"""Per-request launch power: configuration, seeded sampling, and QoT usage."""

from __future__ import annotations

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
from optical_networking_gym.runtime.traffic_model import TrafficModel

RING_4 = BUILTIN_TOPOLOGY_DIR / "ring_4.txt"


def _config(**overrides: object) -> ScenarioConfig:
    values: dict[str, object] = dict(
        scenario_id="power_ring_4",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=64,
        seed=11,
        load=10.0,
        mean_holding_time=10.0,
    )
    values.update(overrides)
    return ScenarioConfig(**values)  # type: ignore[arg-type]


def _requests(config: ScenarioConfig, count: int = 50) -> list[ServiceRequest]:
    topology = TopologyModel.from_file(RING_4, topology_id="ring_4", k_paths=2)
    model = TrafficModel(config, topology)
    return [model.next_request() for _ in range(count)]


def test_config_validates_launch_power_options() -> None:
    with pytest.raises(ValueError):
        _config(nli_interferer_psd="bogus")
    with pytest.raises(ValueError):
        _config(launch_power_dbm_choices=())
    with pytest.raises(ValueError):
        _config(launch_power_seed=-1)
    config = _config(launch_power_dbm_choices=[-6, -4])
    assert config.launch_power_dbm_choices == (-6.0, -4.0)


def test_requests_have_no_power_by_default() -> None:
    assert all(request.launch_power_dbm is None for request in _requests(_config()))


def test_power_sampling_is_seeded_and_leaves_traffic_unchanged() -> None:
    choices = (-6.0, -4.0, -2.0)
    plain = _requests(_config())
    first = _requests(_config(launch_power_dbm_choices=choices, launch_power_seed=5))
    second = _requests(_config(launch_power_dbm_choices=choices, launch_power_seed=5))
    other = _requests(_config(launch_power_dbm_choices=choices, launch_power_seed=6))
    powers = [request.launch_power_dbm for request in first]
    assert set(powers) == set(choices)
    assert powers == [request.launch_power_dbm for request in second]
    assert powers != [request.launch_power_dbm for request in other]
    # The traffic RNG stream is untouched by the power stream.
    for with_power, without in zip(first, plain):
        assert (with_power.source_id, with_power.destination_id) == (
            without.source_id,
            without.destination_id,
        )
        assert with_power.arrival_time == without.arrival_time


def test_request_power_must_be_finite() -> None:
    with pytest.raises(ValueError):
        ServiceRequest(
            request_index=0,
            service_id=0,
            source_id=0,
            destination_id=1,
            bit_rate=100,
            arrival_time=0.0,
            holding_time=1.0,
            launch_power_dbm=float("inf"),
        )


def test_engine_uses_request_power_for_candidates() -> None:
    config = _config()
    topology = TopologyModel.from_file(RING_4, topology_id="ring_4", k_paths=2)
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]
    modulation = Modulation("QPSK", 10_000.0, 2, minimum_osnr=6.72)
    base = dict(
        request_index=0,
        service_id=0,
        source_id=0,
        destination_id=2,
        bit_rate=100,
        arrival_time=0.0,
        holding_time=1.0,
    )
    low = engine.build_candidate(
        ServiceRequest(**base, launch_power_dbm=-6.0), path, modulation, 0, 4
    )
    default = engine.build_candidate(ServiceRequest(**base), path, modulation, 0, 4)
    assert low.launch_power == pytest.approx(10 ** (-0.6) * 1e-3)  # -6 dBm in W
    assert default.launch_power == pytest.approx(1e-3)
    # With ASE-dominated short links, lower power means lower OSNR.
    assert (
        engine.evaluate_candidate(state, low).osnr
        < engine.evaluate_candidate(state, default).osnr
    )
