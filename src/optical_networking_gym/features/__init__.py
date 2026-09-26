from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static-analysis view of the lazy export table below. Kept lazy at
    # runtime because eager subpackage imports create import cycles
    # (e.g. contracts -> network -> runtime -> contracts and
    # optical -> envs -> runtime -> features -> optical).
    from .action_mask import ActionMask
    from .observation import Observation
    from .reward_function import RewardFunction

_EXPORTS: dict[str, tuple[str, str]] = {
    "ActionMask": (".action_mask", "ActionMask"),
    "Observation": (".observation", "Observation"),
    "RewardFunction": (".reward_function", "RewardFunction"),
}

__all__ = [
    "ActionMask",
    "Observation",
    "RewardFunction",
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
