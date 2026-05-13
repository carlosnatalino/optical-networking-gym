from __future__ import annotations

from dataclasses import dataclass

from optical_networking_gym_v2.contracts.action_mask import ActionSelection
from optical_networking_gym_v2.config.scenario import ScenarioConfig


@dataclass(frozen=True, slots=True)
class EncodedAction:
    path_index: int
    modulation_offset: int
    initial_slot: int


def total_actions(config: ScenarioConfig) -> int:
    return (config.k_paths * config.modulations_to_consider * config.num_spectrum_resources) + 1


def reject_action(config: ScenarioConfig) -> int:
    return total_actions(config) - 1


def encode_action(
    config: ScenarioConfig,
    *,
    path_index: int,
    modulation_offset: int,
    initial_slot: int,
) -> int:
    path_stride = config.modulations_to_consider * config.num_spectrum_resources
    return int(
        (path_index * path_stride)
        + (modulation_offset * config.num_spectrum_resources)
        + initial_slot
    )


def decode_action(config: ScenarioConfig, action: int) -> EncodedAction | None:
    if action < 0 or action >= total_actions(config):
        raise ValueError("action is outside the action space")
    if action == reject_action(config):
        return None

    path_stride = config.modulations_to_consider * config.num_spectrum_resources
    path_index = action // path_stride
    modulation_and_slot = action % path_stride
    modulation_offset = modulation_and_slot // config.num_spectrum_resources
    initial_slot = modulation_and_slot % config.num_spectrum_resources
    return EncodedAction(
        path_index=int(path_index),
        modulation_offset=int(modulation_offset),
        initial_slot=int(initial_slot),
    )


def resolve_action_selection(
    config: ScenarioConfig,
    *,
    modulation_indices: tuple[int, ...],
    action: int,
) -> ActionSelection | None:
    decoded = decode_action(config, action)
    if decoded is None:
        return None
    if decoded.modulation_offset >= len(modulation_indices):
        return None
    return ActionSelection(
        path_index=decoded.path_index,
        modulation_index=int(modulation_indices[decoded.modulation_offset]),
        initial_slot=decoded.initial_slot,
    )


__all__ = [
    "EncodedAction",
    "decode_action",
    "encode_action",
    "reject_action",
    "resolve_action_selection",
    "total_actions",
]
