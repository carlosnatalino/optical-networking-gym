"""Select an action with a provisioning heuristic given by name.

``select_heuristic_action(name, env, info)`` is the entry point used by the
example and benchmark scripts; ``utils.experiment_utils.select_policy_action``
delegates to it. Names are case-insensitive and several aliases map to the same
heuristic (e.g. the JOCN 2024 strategy numbers).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from .masked_heuristics import select_first_fit_action
from .runtime_heuristics import (
    select_disruption_aware_first_fit_action,
    select_highest_snr_first_fit_action,
    select_jocn_bm_ksp_lb_action,
    select_jocn_ksp_lb_bm_action,
    select_jocn_ls_bm_ksp_action,
    select_ksp_best_mod_last_fit_action,
    select_load_balancing_action,
    select_lowest_fragmentation_action,
    select_random_action,
)

if TYPE_CHECKING:
    from optical_networking_gym.envs.optical_env import OpticalEnv


def _masked_first_fit(env: OpticalEnv, info: Mapping[str, object]) -> int:
    mask = info.get("mask")
    if mask is None and hasattr(env, "action_masks"):
        mask = env.action_masks()
    if mask is None:
        raise RuntimeError("first-fit policy requires an action mask")
    return int(select_first_fit_action(np.asarray(mask)))


def _runtime(select: Callable[..., int]) -> Callable[[OpticalEnv, Mapping[str, object]], int]:
    return lambda env, info: int(select(env.heuristic_context()))


# Aliases in lookup order (the first match wins).
_HEURISTICS: tuple[tuple[frozenset[str], Callable[[OpticalEnv, Mapping[str, object]], int]], ...] = (
    (frozenset({"jocn_ksp_ff_bm", "ksp-ff-bm", "strategy_1", "1", "first_fit"}), _masked_first_fit),
    (frozenset({"jocn_ls_bm_ksp", "ls-bm-ksp", "strategy_2", "2"}), _runtime(select_jocn_ls_bm_ksp_action)),
    (frozenset({"jocn_bm_ksp_lb", "bm-ksp-lb", "strategy_3", "3"}), _runtime(select_jocn_bm_ksp_lb_action)),
    (frozenset({"jocn_ksp_lb_bm", "ksp-lb-bm", "strategy_4", "4"}), _runtime(select_jocn_ksp_lb_bm_action)),
    (
        frozenset({"disruption_aware_first_fit", "disruption-aware-first-fit"}),
        _runtime(select_disruption_aware_first_fit_action),
    ),
    (frozenset({"random", "random_runtime"}), _runtime(select_random_action)),
    (frozenset({"load_balancing", "load-balancing"}), _runtime(select_load_balancing_action)),
    (frozenset({"lowest_fragmentation", "lowest-fragmentation"}), _runtime(select_lowest_fragmentation_action)),
    (frozenset({"highest_snr_first_fit", "highest-snr-first-fit"}), _runtime(select_highest_snr_first_fit_action)),
    (frozenset({"ksp_best_mod_last_fit", "ksp-best-mod-last-fit"}), _runtime(select_ksp_best_mod_last_fit_action)),
)

HEURISTIC_NAMES: tuple[str, ...] = tuple(sorted(name for aliases, _ in _HEURISTICS for name in aliases))


def select_heuristic_action(name: str, env: OpticalEnv, info: Mapping[str, object] | None = None) -> int:
    """Action chosen by the heuristic ``name`` for the current request of ``env``.

    ``info`` is the info dict of the last ``reset``/``step``; the mask-based
    first fit reads the action mask from it (or from ``env.action_masks()``).

    Raises:
        ValueError: If ``name`` is not one of :data:`HEURISTIC_NAMES`.
    """
    key = name.strip().lower()
    for aliases, select in _HEURISTICS:
        if key in aliases:
            return select(env, {} if info is None else info)
    raise ValueError(f"unsupported policy_name {name!r}")


__all__ = ["HEURISTIC_NAMES", "select_heuristic_action"]
