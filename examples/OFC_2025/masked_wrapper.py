
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

class ActionMaskWrapper(gym.Wrapper):
    """
    Extrai a máscara (info["mask"]) e disponibiliza via .action_masks().
    """
    def __init__(self, env):
        super().__init__(env)
        self._last_mask = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_mask = info.get("mask", None)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._last_mask = info.get("mask", None)
        
        return obs, reward, done, truncated, info

    def action_masks(self):
        return self._last_mask
