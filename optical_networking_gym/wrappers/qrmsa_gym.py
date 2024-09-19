from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.utils import seeding
from optical_networking_gym.envs.qrmsa import QRMSAEnv

class QRMSAEnvWrapper(gym.Env):
    def __init__(self, *args, **kwargs):
        self.env = QRMSAEnv(*args, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self._np_random, _ = seeding.np_random(None)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            self.env.input_seed = seed

        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        print(action)
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
