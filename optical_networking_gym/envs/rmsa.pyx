from typing import Any, Literal, Sequence, SupportsFloat

import cython
import gymnasium as gym
import networkx as nx
import numpy as np


class RMSAEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Interface defined by: https://gymnasium.farama.org/api/env/
    """

    topology: nx.Graph
    episode_length: cython.int
    load: cython.float
    mean_service_holding_time: cython.float = 10800.0
    num_spectrum_resources: cython.int = 100,
    bit_rate_selection: str = Literal["continuous", "discrete"]
    bit_rates: tuple[int | float] = (0, 40, 100)
    bit_rate_probabilities: np.ndarray | None = None
    node_request_probabilities: np.ndarray | None = None
    bit_rate_lower_bound: cython.float = 25.0
    bit_rate_higher_bound: cython.float = 100.0
    allow_rejection: bool = False
    reset: bool = True
    channel_width: cython.float = 12.5

    def __init__(self) -> None:
        super().__init__()
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Conciously not testing for the dimensions of the ndarray to speed up things.
        It will fail in the first execution anyways.
        """
        return super().step(action)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def close(self):
        return super().close()
