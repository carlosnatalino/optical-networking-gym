from typing import Any, Literal, Sequence, SupportsFloat

cimport cython
cimport numpy as cnp
cnp.import_array()
import random

import gymnasium as gym
from gymnasium.utils import seeding
import networkx as nx
import numpy as np

from optical_networking_gym.utils import rle

cdef class RMSAEnv:
    """
    Interface defined by: https://gymnasium.farama.org/api/env/

    Inspired by: https://github.com/gursky1/cygym/blob/master/cygym/acrobot.pyx
    """

    cdef:
        float load
        int episode_length
        float mean_service_holding_time
        int num_spectrum_resources
        float channel_width
        cnp.ndarray _spectrum_use
        cnp.int32_t[:, :] spectrum_use
        cnp.ndarray _spectrum_allocation
        cnp.int64_t[:, :] spectrum_allocation
        readonly object observation_space
        readonly object action_space
        object _np_random
        int _np_random_seed
        float bit_rate_lower_bound
        float bit_rate_higher_bound
        bint allow_rejection

    topology: cython.declare(nx.Graph, visibility="readonly")
    bit_rate_selection: cython.declare(Literal["continuous", "discrete"], visitility="readonly")
    bit_rates: cython.declare(tuple[int | float], visitility="readonly")

    def __cinit__(
        self,
        topology: nx.Graph,
        num_spectrum_resources: cython.int,

    ) -> None:
        self.topology = topology
        self.num_spectrum_resources = num_spectrum_resources
        self.bit_rates = (0, 40, 100)
        self._spectrum_use = np.full(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            fill_value=-1,
            dtype=np.int32,
        )
        self.spectrum_use = self._spectrum_use
        self._spectrum_allocation = np.full(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            fill_value=-1,
            dtype=np.int64,
        )
        self.spectrum_allocation = self._spectrum_allocation

        cdef Py_ssize_t row, column
        row = random.randint(0, self.topology.number_of_edges())
        column = random.randint(0, self.num_spectrum_resources)
        self.spectrum_use[row, column] = 0
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Conciously not testing for the dimensions of the ndarray to speed up things.
        It will fail in the first execution anyways.
        """
        return super().step(action)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
        return None

    def close(self):
        return super().close()
