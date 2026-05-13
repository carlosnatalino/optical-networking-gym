from __future__ import annotations

from optical_networking_gym_v2.api import make_env
from optical_networking_gym_v2.config import MODULATION_CATALOG, ScenarioConfig, get_modulations
from optical_networking_gym_v2.features import ActionMask, Observation
from optical_networking_gym_v2.runtime import (
    RequestAnalysis,
    RequestAnalysisEngine,
    RuntimeState,
    Simulator,
    StepInfo,
    TrafficModel,
)


def test_architecture_facades_reexport_core_modules() -> None:
    assert make_env is not None
    assert ScenarioConfig is not None
    assert MODULATION_CATALOG
    assert get_modulations("QPSK")
    assert ActionMask is not None
    assert Observation is not None
    assert RequestAnalysis is not None
    assert RequestAnalysisEngine is not None
    assert RuntimeState is not None
    assert Simulator is not None
    assert StepInfo is not None
    assert TrafficModel is not None
