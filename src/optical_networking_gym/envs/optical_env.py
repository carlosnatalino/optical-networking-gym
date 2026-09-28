from __future__ import annotations

from typing import Any

import numpy as np

import gymnasium as gym

from optical_networking_gym.contracts import StepTransition
from optical_networking_gym.network.topology import TopologyModel
from optical_networking_gym.config.scenario import ScenarioConfig
from optical_networking_gym.runtime.request_analysis import RequestAnalysis
from optical_networking_gym.runtime.simulator import Simulator


class OpticalEnv(gym.Env):
    """Routing, modulation and spectrum assignment (RMSA) environment.

    Extension points for subclasses:

    * :meth:`observation` maps the simulator's flat observation to the
      observation returned by :meth:`reset` and :meth:`step` (same contract as
      ``gymnasium.ObservationWrapper.observation``). Subclasses that return a
      different structure should also set ``self.observation_space``.
    * :meth:`on_action_applied` is called inside :meth:`step` right after the
      action is applied, while ``self.simulator.state`` still reflects the
      moment of the decision (before time advances to the next request and
      expired services are released). Use it to record quantities that are only
      meaningful at that instant, e.g. the state seen by a newly established
      lightpath.
    * :meth:`on_request_analysed` is called with the
      :class:`~optical_networking_gym.runtime.request_analysis.RequestAnalysis`
      of every new request, before the policy acts: the candidate paths,
      formats and start slots with the QoT the environment computed for them
      (e.g. to log the QoT queries of the RMSA, the unestablished lightpaths a
      QoT estimator serves in operation). It is only wired when a subclass
      overrides it, so the default costs nothing.
    """

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
        self.simulator.post_action_callback = self.on_action_applied
        if type(self).on_request_analysed is not OpticalEnv.on_request_analysed:
            self.simulator.request_analysed_callback = self.on_request_analysed
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
        # Seed gymnasium's np_random alongside the simulator's own RNG so the
        # env satisfies the Gymnasium API contract (check_env).
        super().reset(seed=seed)
        observation, info = self.simulator.reset(seed=seed, options=options)
        return self.observation(observation), info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.simulator.step(int(action))
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation: np.ndarray) -> Any:
        """Map the simulator observation to the returned observation (identity)."""
        return observation

    def on_action_applied(self, transition: StepTransition) -> None:
        """Hook called right after each action is applied (no-op by default)."""
        return None

    def on_request_analysed(self, analysis: RequestAnalysis) -> None:
        """Hook called with the analysis of each new request, before the policy
        acts (no-op by default).

        ``analysis.paths``, ``analysis.modulation_indices`` and
        ``analysis.required_slots_by_path_mod`` describe the candidates;
        ``analysis.resource_valid_starts`` marks the free start slots and
        ``analysis.gsnr_db_by_start``/``osnr_margin_by_start`` hold their QoT
        (NaN where not evaluated). The state is ``self.simulator.state``, at
        the arrival of the request.
        """
        return None

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

__all__ = ["OpticalEnv"]
