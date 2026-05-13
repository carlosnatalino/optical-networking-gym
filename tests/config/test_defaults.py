from __future__ import annotations

from pathlib import Path

import pytest

from optical_networking_gym_v2 import MODULATION_CATALOG, get_modulations, resolve_topology, set_topology_dir


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_DIR = PROJECT_ROOT / "examples" / "topologies"


def test_get_modulations_parses_catalog_names() -> None:
    modulations = get_modulations("QPSK, 16QAM")

    assert tuple(modulation.name for modulation in modulations) == ("QPSK", "16QAM")
    assert modulations[0] == MODULATION_CATALOG["QPSK"]


def test_resolve_topology_uses_configured_directory() -> None:
    set_topology_dir(TOPOLOGY_DIR)

    topology_path = resolve_topology("ring_4")

    assert topology_path == TOPOLOGY_DIR / "ring_4.txt"


def test_get_modulations_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown modulation"):
        get_modulations("QPSK, INVALID")
