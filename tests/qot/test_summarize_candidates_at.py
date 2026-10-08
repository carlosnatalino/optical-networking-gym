from __future__ import annotations

import numpy as np
import pytest

from optical_networking_gym import MaskMode, make_env
from optical_networking_gym.heuristics.runtime_heuristics import select_first_fit_action


def _loaded_simulator(**overrides: object):
    env = make_env(
        scenario="jocn_benchmark",
        episode_length=400,
        overrides={
            "mask_mode": MaskMode.RESOURCE_ONLY,
            "enable_observation": False,
            "enable_action_mask": False,
            "analysis_detail": "resources",
            **overrides,
        },
    )
    env.reset(seed=13)
    for _ in range(300):
        env.step(select_first_fit_action(env))
    simulator = env.unwrapped.simulator
    assert len(simulator.state.active_services_by_id) > 100
    return simulator


@pytest.mark.parametrize("correction", ["egn_xci", "cfm2", "gn"])
@pytest.mark.parametrize("interferer_psd", ["cut", "actual"])
def test_batched_summaries_match_single_summaries(correction: str, interferer_psd: str) -> None:
    simulator = _loaded_simulator(nli_modulation_correction=correction, nli_interferer_psd=interferer_psd)
    config = simulator.config
    qot_engine = simulator.qot_engine
    state = simulator.state
    rng = np.random.default_rng(0)
    active_ids = sorted(state.active_services_by_id)
    paths = simulator.topology.paths

    for trial in range(40):
        path = paths[int(rng.integers(len(paths)))]
        # A new service and, sometimes, an established one (excluded from its own XCI).
        service_id = int(active_ids[int(rng.integers(len(active_ids)))]) if trial % 4 == 0 else 10**6
        launch_power = None if trial % 3 else 10 ** (float(rng.uniform(-3.0, 3.0)) / 10.0) * 1e-3
        candidates = []
        for _ in range(int(rng.integers(1, 8))):
            modulation = config.modulations[int(rng.integers(len(config.modulations)))]
            num_slots = int(rng.integers(1, 12))
            start = int(rng.integers(0, config.num_spectrum_resources - num_slots + 1))
            candidates.append((modulation, start, num_slots))

        batched = qot_engine.summarize_candidates_at(
            state=state,
            service_id=service_id,
            path=path,
            candidates=candidates,
            launch_power=launch_power,
        )
        single = [
            qot_engine.summarize_candidate_at(
                state=state,
                service_id=service_id,
                path=path,
                modulation=modulation,
                service_slot_start=start,
                service_num_slots=num_slots,
                launch_power=launch_power,
            )
            for modulation, start, num_slots in candidates
        ]
        # Dataclass equality compares every float exactly.
        assert batched == single


def test_empty_candidates() -> None:
    simulator = _loaded_simulator()
    assert (
        simulator.qot_engine.summarize_candidates_at(
            state=simulator.state,
            service_id=0,
            path=simulator.topology.paths[0],
            candidates=[],
        )
        == []
    )
