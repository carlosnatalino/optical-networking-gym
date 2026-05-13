from __future__ import annotations

from pathlib import Path

from optical_networking_gym_v2.contracts.modulation import Modulation


MODULATION_CATALOG: dict[str, Modulation] = {
    "BPSK": Modulation("BPSK", 100_000.0, 1, minimum_osnr=12.6, inband_xt=-14.0),
    "QPSK": Modulation("QPSK", 2_000.0, 2, minimum_osnr=12.6, inband_xt=-17.0),
    "8QAM": Modulation("8QAM", 1_000.0, 3, minimum_osnr=18.6, inband_xt=-20.0),
    "16QAM": Modulation("16QAM", 500.0, 4, minimum_osnr=22.4, inband_xt=-23.0),
    "32QAM": Modulation("32QAM", 250.0, 5, minimum_osnr=26.4, inband_xt=-26.0),
    "64QAM": Modulation("64QAM", 125.0, 6, minimum_osnr=30.4, inband_xt=-29.0),
}

DEFAULT_SEED = 10
DEFAULT_LOAD = 300.0
DEFAULT_MEAN_HOLDING_TIME = 10800.0
DEFAULT_NUM_SPECTRUM_RESOURCES = 320
DEFAULT_K_PATHS = 5
DEFAULT_LAUNCH_POWER_DBM = 1.0
DEFAULT_MODULATIONS_TO_CONSIDER = 3

BUILTIN_TOPOLOGY_DIR: Path = Path(__file__).parent / "topologies"

_TOPOLOGY_DIR: Path | None = None


def set_topology_dir(path: str | Path) -> None:
    global _TOPOLOGY_DIR
    _TOPOLOGY_DIR = Path(path)


def resolve_topology(name: str) -> Path:
    search_dir = _TOPOLOGY_DIR if _TOPOLOGY_DIR is not None else BUILTIN_TOPOLOGY_DIR
    for suffix in (".xml", ".txt"):
        candidate = search_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Topology {name!r} not found in {search_dir}. Tried: {name}.xml, {name}.txt"
    )


def get_modulations(names: str | tuple[str, ...]) -> tuple[Modulation, ...]:
    if isinstance(names, str):
        parsed = tuple(name.strip() for name in names.split(",") if name.strip())
    else:
        parsed = tuple(names)
    if not parsed:
        raise ValueError("At least one modulation must be provided")
    unknown = [name for name in parsed if name not in MODULATION_CATALOG]
    if unknown:
        raise ValueError(f"Unknown modulation(s): {', '.join(unknown)}")
    return tuple(MODULATION_CATALOG[name] for name in parsed)


__all__ = [
    "DEFAULT_K_PATHS",
    "DEFAULT_LAUNCH_POWER_DBM",
    "DEFAULT_LOAD",
    "DEFAULT_MEAN_HOLDING_TIME",
    "DEFAULT_MODULATIONS_TO_CONSIDER",
    "DEFAULT_NUM_SPECTRUM_RESOURCES",
    "DEFAULT_SEED",
    "MODULATION_CATALOG",
    "get_modulations",
    "resolve_topology",
    "set_topology_dir",
]
