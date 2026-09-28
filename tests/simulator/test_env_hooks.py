"""Extension hooks of OpticalEnv: observation(), on_action_applied() and on_request_analysed()."""

from __future__ import annotations

from typing import Any

import numpy as np

from optical_networking_gym import make_env
from optical_networking_gym.contracts import StepTransition
from optical_networking_gym.envs import OpticalEnv
from optical_networking_gym.heuristics.runtime_heuristics import select_first_fit_action
from optical_networking_gym.runtime.request_analysis import RequestAnalysis


class _RecordingEnv(OpticalEnv):
    """Returns a dict observation and records the state at each decision."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.records: list[dict[str, Any]] = []

    def on_action_applied(self, transition: StepTransition) -> None:
        state = self.simulator.state
        assert state is not None
        service_id = transition.request.service_id
        self.records.append(
            {
                "accepted": transition.accepted,
                "service_active": service_id in state.active_services_by_id,
                "time": state.current_time,
            }
        )

    def observation(self, observation: np.ndarray) -> Any:
        return {"flat": observation, "n_records": len(self.records)}


def _make(env_cls: type[OpticalEnv]) -> OpticalEnv:
    base = make_env(
        "nobel-eu",
        "QPSK,8QAM,16QAM",
        seed=3,
        load=200.0,
        episode_length=60,
        num_spectrum_resources=80,
        k_paths=2,
    )
    return env_cls(
        base.simulator.base_config,
        base.simulator.topology,
        episode_length=60,
    )


def test_default_observation_hook_is_identity() -> None:
    env = _make(OpticalEnv)
    observation, _ = env.reset(seed=3)
    assert isinstance(observation, np.ndarray)


def test_subclass_hooks_see_the_state_at_decision_time() -> None:
    env = _make(_RecordingEnv)
    observation, _ = env.reset(seed=3)
    assert observation["n_records"] == 0
    terminated = False
    steps = 0
    while not terminated:
        action = select_first_fit_action(env)
        observation, _, terminated, _, _ = env.step(action)
        steps += 1
        assert observation["n_records"] == steps
        assert env.simulator.last_transition is not None
    assert isinstance(env, _RecordingEnv)
    accepted = [record for record in env.records if record["accepted"]]
    assert accepted, "first-fit should accept some requests"
    # At decision time every accepted service is active (not yet released).
    assert all(record["service_active"] for record in accepted)


class _QueryEnv(OpticalEnv):
    """Records every request analysis and the request each action answers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.analyses: list[RequestAnalysis] = []
        self.answered: list[int] = []

    def on_request_analysed(self, analysis: RequestAnalysis) -> None:
        # Called before the policy acts: the analysis is the current one.
        assert self.simulator.current_analysis is analysis
        self.analyses.append(analysis)

    def on_action_applied(self, transition: StepTransition) -> None:
        self.answered.append(transition.request.request_index)


def test_request_analysed_hook_is_not_wired_by_default() -> None:
    env = _make(OpticalEnv)
    assert env.simulator.request_analysed_callback is None
    assert _make(_QueryEnv).simulator.request_analysed_callback is not None


def test_request_analysed_hook_sees_every_request_before_the_policy() -> None:
    env = _make(_QueryEnv)
    env.reset(seed=3)
    assert len(env.analyses) == 1
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(select_first_fit_action(env))
    assert isinstance(env, _QueryEnv)
    # One analysis per request presented to the policy, in the same order.
    assert [analysis.request.request_index for analysis in env.analyses] == env.answered
    config = env.simulator.config
    evaluated = 0
    for analysis in env.analyses:
        gsnr = analysis.gsnr_db_by_start
        margin = analysis.osnr_margin_by_start
        assert gsnr.shape == margin.shape
        # QoT is evaluated exactly where a free block exists (NaN elsewhere).
        np.testing.assert_array_equal(np.isfinite(gsnr), analysis.resource_valid_starts)
        for offset, modulation_index in enumerate(analysis.modulation_indices):
            threshold = config.modulations[modulation_index].minimum_osnr + config.margin
            np.testing.assert_allclose(gsnr[:, offset] - threshold, margin[:, offset], atol=1e-5)
        assert analysis.launch_power_dbm == config.launch_power_dbm
        evaluated += int(np.isfinite(gsnr).sum())
    assert evaluated > 0
