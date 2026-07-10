from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static-analysis view of the lazy export table below. Kept lazy at
    # runtime because eager subpackage imports create import cycles
    # (e.g. contracts -> network -> runtime -> contracts and
    # optical -> envs -> runtime -> features -> optical).
    from .first_fit import (
        select_first_fit_action_from_env,
        shortest_available_path_first_fit_best_modulation,
    )
    from .first_fit_example import run_episode as run_first_fit_episode
    from .kernels import accumulate_link_noise, block_is_free, candidate_starts_array, fill_range
    from .qot_engine import QoTEngine

_EXPORTS: dict[str, tuple[str, str]] = {
    "QoTEngine": (".qot_engine", "QoTEngine"),
    "accumulate_link_noise": (".kernels", "accumulate_link_noise"),
    "block_is_free": (".kernels", "block_is_free"),
    "candidate_starts_array": (".kernels", "candidate_starts_array"),
    "fill_range": (".kernels", "fill_range"),
    "run_first_fit_episode": (".first_fit_example", "run_episode"),
    "select_first_fit_action_from_env": (".first_fit", "select_first_fit_action_from_env"),
    "shortest_available_path_first_fit_best_modulation": (".first_fit", "shortest_available_path_first_fit_best_modulation"),
}

__all__ = [
    "QoTEngine",
    "accumulate_link_noise",
    "block_is_free",
    "candidate_starts_array",
    "fill_range",
    "run_first_fit_episode",
    "select_first_fit_action_from_env",
    "shortest_available_path_first_fit_best_modulation",
]


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
