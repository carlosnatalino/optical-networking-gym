import time
from typing import Optional, Tuple, Any, SupportsFloat

import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
import numpy as np

from optical_networking_gym.envs.qrmsa import QRMSAEnv
from optical_networking_gym.utils import rle

from optical_networking_gym.heuristics.heuristics import (
    shortest_available_path_first_fit_best_modulation,
    shortest_available_path_lowest_spectrum_best_modulation,
    best_modulation_load_balancing,
    load_balancing_best_modulation,
)

register(
    id='QRMSAEnvWrapper-v0',
    entry_point='optical_networking_gym.wrappers.qrmsa_gym:QRMSAEnvWrapper',
)

class QRMSAEnvWrapper(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, *args, **kwargs):
        super(QRMSAEnvWrapper, self).__init__()
        self.env = QRMSAEnv(*args, **kwargs)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self._np_random, _ = seeding.np_random(kwargs.get("seed", None))
        self.num_spectrum_resources = kwargs.get("num_spectrum_resources", 320)
        self.bit_rates = kwargs.get("bit_rates", (10, 40, 100))
        self.channel_width = kwargs.get("channel_width", 12.5)
        self.seed_value = kwargs.get("seed", 10)

        self._last_mask = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed_ = seeding.np_random(seed)
            self.env.input_seed = seed_

        obs, info = self.env.reset(seed=seed, options=options)
        if "mask" in info:
            self._last_mask = info["mask"]
        return obs, info

    def step(self, action: Any):
        obs, reward, done, truncated, info = self.env.step(action)
        if "mask" in info:
            self._last_mask = info["mask"]
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
        else:
            pass

    def close(self):
        if hasattr(self.env, 'close'):
            return self.env.close()
        else:
            pass


    def action_masks(self):
        return self._last_mask if self._last_mask is not None else None
    
    def get_spectrum_use_services(self):
        return self.env.get_spectrum_use_services()

    def get_available_slots(self, route):
        return self.env.get_available_slots(route)

    def get_number_slots(self, service, modulation):
        return self.env.get_number_slots(service, modulation)

    def get_available_blocks(self, idp, num_slots, j):
        return self.env.get_available_blocks(idp, num_slots, j)

