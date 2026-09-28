"""Name-based heuristic dispatcher (heuristics.select_heuristic_action)."""

from __future__ import annotations

import pytest

from optical_networking_gym import HEURISTIC_NAMES, make_env, select_heuristic_action
from optical_networking_gym.utils.experiment_utils import select_policy_action


@pytest.fixture()
def env_and_info():
    env = make_env("nobel-eu", "QPSK,8QAM,16QAM", seed=5, load=200.0, episode_length=30, k_paths=2)
    _, info = env.reset(seed=5)
    return env, info


def test_every_name_selects_a_valid_action(env_and_info) -> None:
    env, info = env_and_info
    assert {"ksp-ff-bm", "ls-bm-ksp", "ksp-lb-bm", "bm-ksp-lb", "first_fit", "random"} <= set(HEURISTIC_NAMES)
    for name in HEURISTIC_NAMES:
        action = select_heuristic_action(name, env, info)
        assert 0 <= action < env.action_space.n


def test_names_are_case_insensitive_and_match_the_legacy_entry_point(env_and_info) -> None:
    env, info = env_and_info
    for name in ("KSP-FF-BM", "LS-BM-KSP", " ksp-lb-bm ", "strategy_3", "load-balancing"):
        assert select_heuristic_action(name, env, info) == select_policy_action(name, env, info)
    # The mask-based first fit reads the mask from the env when info has none.
    assert select_heuristic_action("first_fit", env) == select_heuristic_action("first_fit", env, info)


def test_unknown_names_are_rejected(env_and_info) -> None:
    env, info = env_and_info
    with pytest.raises(ValueError, match="unsupported policy_name"):
        select_heuristic_action("no-such-heuristic", env, info)
