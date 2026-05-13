from __future__ import annotations

from pathlib import Path

import pytest

from optical_networking_gym_v2 import TopologyModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RING_4_PATH = PROJECT_ROOT / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def test_topology_model_builds_ring4_from_file() -> None:
    model = TopologyModel.from_file(
        RING_4_PATH,
        topology_id="ring_4",
        k_paths=2,
        max_span_length_km=80.0,
        default_attenuation_db_per_km=0.2,
        default_noise_figure_db=4.5,
    )

    assert model.topology_id == "ring_4"
    assert model.node_count == 4
    assert model.link_count == 4
    assert model.path_count == 12
    assert model.get_node_index("1") == 0

    first_link = model.link_by_id(0)
    assert first_link.source_name == "1"
    assert first_link.target_name == "2"
    assert first_link.length_km == pytest.approx(150.0)
    assert len(first_link.spans) == 2
    assert first_link.spans[0].length_km == pytest.approx(75.0)

    paths = model.get_paths("1", "3")
    assert len(paths) == 2
    assert [path.hops for path in paths] == [2, 2]
    assert [path.length_km for path in paths] == [pytest.approx(600.0), pytest.approx(600.0)]


def test_topology_model_normalizes_undirected_queries() -> None:
    model = TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)

    forward = model.get_paths("1", "3")
    reverse = model.get_paths("3", "1")

    assert [path.id for path in forward] == [path.id for path in reverse]
    assert [path.node_names for path in reverse] == [path.node_names for path in forward]
    assert [path.node_indices for path in reverse] == [path.node_indices for path in forward]
    assert [path.link_ids for path in reverse] == [path.link_ids for path in forward]
