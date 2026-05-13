from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - import guard only
    gym = None
    _GYM_IMPORT_ERROR = exc
else:
    _GYM_IMPORT_ERROR = None

from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.config.scenario import ScenarioConfig
from optical_networking_gym_v2.runtime.simulator import Simulator


if gym is not None:

    class OpticalEnv(gym.Env):
        metadata = {"render_modes": ["human"]}

        def __init__(
            self,
            config: ScenarioConfig,
            topology: TopologyModel,
            *,
            episode_length: int,
            capture_traffic_table: bool = False,
            capture_step_trace: bool = False,
        ) -> None:
            super().__init__()
            self.simulator = Simulator(
                config,
                topology,
                episode_length=episode_length,
                capture_traffic_table=capture_traffic_table,
                capture_step_trace=capture_step_trace,
            )
            self.action_space = gym.spaces.Discrete(self.simulator.total_actions)
            observation_shape = (
                (0,)
                if not config.enable_observation
                else (self.simulator.observation_builder.schema.total_size,)
            )
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=observation_shape,
                dtype=np.float32,
            )

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            return self.simulator.reset(seed=seed, options=options)

        def step(self, action: int):
            return self.simulator.step(int(action))

        def action_masks(self) -> np.ndarray | None:
            return self.simulator.action_masks()

        def heuristic_context(self):
            return self.simulator.heuristic_context()

        def get_trace_action_mask(self) -> np.ndarray:
            return self.simulator.get_trace_action_mask()

        def export_captured_traffic_table(self):
            return self.simulator.export_captured_traffic_table()

        def save_captured_traffic_table_jsonl(self, file_path: str):
            return self.simulator.save_captured_traffic_table_jsonl(file_path)

        def export_step_trace(self):
            return self.simulator.export_step_trace()

        def save_step_trace_jsonl(self, file_path: str):
            return self.simulator.save_step_trace_jsonl(file_path)

        def render(self):
            return None

        def close(self):
            return None

else:

    class OpticalEnv:  # pragma: no cover - exercised only when gymnasium is unavailable
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("gymnasium is required to use OpticalEnv") from _GYM_IMPORT_ERROR


__all__ = ["OpticalEnv"]
