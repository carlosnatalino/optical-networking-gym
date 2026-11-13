from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from optical_networking_gym import MaskMode, build_scenario, make_env
from optical_networking_gym.heuristics.runtime_heuristics import select_first_fit_action


_ZERO_FILLED = (
    "fragmentation_damage_num_blocks_by_start",
    "fragmentation_damage_largest_block_by_start",
    "path_route_cuts_norm_by_path",
    "path_route_rss_by_path",
)
_SHARED = (
    "resource_valid_starts",
    "qot_valid_starts",
    "osnr_margin_by_start",
    "nli_share_by_start",
    "worst_link_nli_share_by_start",
    "required_slots_by_path_mod",
    "action_mask",
)


def _run(detail: str, arrivals: int, **overrides: object) -> list[dict[str, object]]:
    env = make_env(
        scenario="jocn_benchmark",
        episode_length=arrivals,
        overrides={"enable_observation": False, "analysis_detail": detail, **overrides},
    )
    simulator = env.unwrapped.simulator
    env.reset(seed=5)
    records: list[dict[str, object]] = []
    for _ in range(arrivals):
        analysis = simulator.current_analysis
        record: dict[str, object] = {
            "modulation_indices": analysis.modulation_indices,
            "paths": tuple(path.id for path in analysis.paths),
            "mean_link_entropy": analysis.mean_link_entropy,
        }
        for name in _SHARED + _ZERO_FILLED:
            value = getattr(analysis, name)
            record[name] = None if value is None else np.array(value, copy=True)
        _, reward, terminated, truncated, info = env.step(select_first_fit_action(env))
        record["reward"] = reward
        record["info"] = {key: value for key, value in info.items() if key != "mask"}
        records.append(record)
        if terminated or truncated:
            break
    return records


def _assert_arrays_equal(left: object, right: object) -> None:
    if right is None:
        assert left is None
        return
    assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mask_mode": MaskMode.RESOURCE_ONLY, "enable_action_mask": False},
        {"mask_mode": MaskMode.RESOURCE_ONLY},
        {"mask_mode": MaskMode.RESOURCE_AND_QOT},
        {"mask_mode": MaskMode.RESOURCE_AND_QOT, "qot_constraint": "DIST"},
    ],
    ids=["resource_only", "resource_only_mask", "resource_and_qot", "resource_and_qot_dist"],
)
def test_resources_detail_keeps_the_resource_and_qot_results(overrides: dict[str, object]) -> None:
    full = _run("full", 200, **overrides)
    lean = _run("resources", 200, **overrides)

    assert len(lean) == len(full)
    for lean_step, full_step in zip(lean, full):
        assert lean_step["modulation_indices"] == full_step["modulation_indices"]
        assert lean_step["paths"] == full_step["paths"]
        for name in _SHARED:
            _assert_arrays_equal(lean_step[name], full_step[name])
        # Skipped outputs are zero-filled with the same shapes and dtypes.
        assert lean_step["mean_link_entropy"] == 0.0
        for name in _ZERO_FILLED:
            lean_array, full_array = lean_step[name], full_step[name]
            assert isinstance(lean_array, np.ndarray) and isinstance(full_array, np.ndarray)
            assert lean_array.dtype == full_array.dtype
            assert lean_array.shape == full_array.shape
            assert not lean_array.any()

        lean_info, full_info = lean_step["info"], full_step["info"]
        assert isinstance(lean_info, dict) and isinstance(full_info, dict)
        for key in ("accepted", "status", "osnr", "osnr_req", "chosen_path_index", "chosen_slot"):
            assert lean_info[key] == full_info[key]
        for key in (
            "fragmentation_shannon_entropy",
            "fragmentation_route_cuts",
            "fragmentation_route_rss",
            "reward_fragmentation_penalty",
        ):
            assert lean_info[key] == 0.0
    # The fragmentation terms are computed in the full analysis.
    assert any(step["info"]["fragmentation_route_cuts"] != 0.0 for step in full)  # type: ignore[index]


def test_inspection_builds_the_full_analysis() -> None:
    env = make_env(
        scenario="jocn_benchmark",
        episode_length=40,
        overrides={"enable_observation": False, "analysis_detail": "resources"},
    )
    simulator = env.unwrapped.simulator
    env.reset(seed=5)
    for _ in range(20):
        env.step(select_first_fit_action(env))

    lean = simulator.current_analysis
    full = simulator.analysis_engine.build(simulator.state, simulator.current_request, include_inspection=True)

    assert lean is not full
    assert lean.inspection is None
    assert full.inspection is not None
    assert full.fragmentation_damage_num_blocks_by_start.any()
    np.testing.assert_array_equal(lean.resource_valid_starts, full.resource_valid_starts)
    np.testing.assert_array_equal(lean.qot_valid_starts, full.qot_valid_starts)


def test_analysis_detail_validation_and_structure_key() -> None:
    config = build_scenario("ring4_quickstart", enable_observation=False)
    assert config.analysis_detail == "full"
    lean = replace(config, analysis_detail="resources")
    assert lean.runtime_structure_key() != config.runtime_structure_key()

    with pytest.raises(ValueError, match="analysis_detail must be one of"):
        replace(config, analysis_detail="lean")
    with pytest.raises(ValueError, match="requires enable_observation=False"):
        build_scenario("ring4_quickstart", analysis_detail="resources")
