from __future__ import annotations

from optical_networking_gym.envs.optical_env import OpticalEnv
from optical_networking_gym.heuristics.runtime_heuristics import (
    select_first_fit_action as select_first_fit_runtime_action,
    select_first_fit_decision,
)
from optical_networking_gym.runtime.simulator import Simulator

def shortest_available_path_first_fit_best_modulation(
    env: OpticalEnv | Simulator,
) -> tuple[int, bool, bool]:
    return select_first_fit_decision(env)


def select_first_fit_action_from_env(env: OpticalEnv | Simulator) -> int:
    return select_first_fit_runtime_action(env)


__all__ = [
    "select_first_fit_action_from_env",
    "shortest_available_path_first_fit_best_modulation",
]
