from __future__ import annotations

import pytest

from optical_networking_gym.topology import get_topology

from .helpers import RING_4_PATH, build_ring4_topology_v2


def test_topology_model_matches_legacy_ring4_structure() -> None:
    legacy = get_topology(
        str(RING_4_PATH),
        topology_name="ring_4",
        modulations=None,
        max_span_length=80.0,
        default_attenuation=0.2,
        default_noise_figure=4.5,
        k_paths=2,
    )
    model = build_ring4_topology_v2(
        k_paths=2,
        max_span_length_km=80.0,
        default_attenuation_db_per_km=0.2,
        default_noise_figure_db=4.5,
    )

    assert model.node_names == tuple(legacy.nodes())
    assert model.link_count == legacy.number_of_edges()

    for node_u, node_v, edge_data in legacy.edges(data=True):
        legacy_link = edge_data["link"]
        model_link = model.link_between(node_u, node_v)
        assert model_link.id == legacy_link.id
        assert model_link.length_km == pytest.approx(legacy_link.length)
        assert len(model_link.spans) == len(legacy_link.spans)

    legacy_paths = legacy.graph["ksp"]["1", "3"]
    model_paths = model.get_paths("1", "3")
    assert [path.node_names for path in model_paths] == [tuple(path.node_list) for path in legacy_paths]
    assert [path.length_km for path in model_paths] == [pytest.approx(path.length) for path in legacy_paths]
