import time
from typing import Optional, Tuple

from datetime import datetime
from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from optical_networking_gym.envs.qrmsa import QRMSAEnv
from optical_networking_gym.utils import rle

from optical_networking_gym.heuristics.heuristics import (
    shortest_available_path_first_fit_best_modulation,
    shortest_available_path_lowest_spectrum_best_modulation,
    best_modulation_load_balancing,
    load_balancing_best_modulation,
)

from gymnasium.envs.registration import register

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
        self._np_random, _ = seeding.np_random(None)
        self.num_spectrum_resources =kwargs["num_spectrum_resources"]
        self.bit_rates = kwargs["bit_rates"]
        self.channel_width = kwargs.get("channel_width",12.5)
        self.seed = kwargs.get("seed", 10)
        

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            self.env.input_seed = seed

        obs = self.env.reset(seed=seed, options=options)
        return obs

    def step(self, action: Any):
        print(action)
        print("type:",type(action))
        return self.env.step(action)

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

    def get_spectrum_use_services(self):
        return self.env.get_spectrum_use_services()
    
    def get_available_slots(self, route):
        return self.env.get_available_slots(route)
    
    def get_number_slots(self, service, modulation):
        return self.env.get_number_slots(service, modulation)
    
    def get_available_blocks(self, idp, num_slots, j):
        return self.env.get_available_blocks(idp, num_slots, j)
