"""``TopologyModel.with_span_updates``: runtime changes of span parameters."""

from __future__ import annotations

import math

import pytest

from optical_networking_gym import SpanUpdate, TopologyModel
from optical_networking_gym.defaults import BUILTIN_TOPOLOGY_DIR
from optical_networking_gym.network import SpanUpdate as NetworkSpanUpdate


@pytest.fixture(scope="module")
def topology() -> TopologyModel:
    return TopologyModel.from_file(BUILTIN_TOPOLOGY_DIR / "nobel-eu.xml", k_paths=2, max_span_length_km=80.0)


def test_span_update_is_exported() -> None:
    assert SpanUpdate is NetworkSpanUpdate


def test_original_is_unchanged(topology: TopologyModel) -> None:
    before = topology.links[3].spans
    aged = topology.with_span_updates([SpanUpdate(3, 0, attenuation_db_per_km=0.25, noise_figure_db=6.0)])

    assert topology.links[3].spans is before
    assert topology.links[3].spans[0].attenuation_db_per_km == 0.2
    assert topology.links[3].spans[0].noise_figure_db == 4.5
    assert aged.links[3].spans[0].attenuation_db_per_km == 0.25
    assert aged.links[3].spans[0].noise_figure_db == 6.0


def test_untouched_fields_are_identical(topology: TopologyModel) -> None:
    aged = topology.with_span_updates([SpanUpdate(5, 1, noise_figure_db=5.5)])

    for name in (
        "topology_id",
        "node_names",
        "paths",
        "path_index_by_endpoints",
        "node_index_by_name",
        "link_id_by_endpoints",
        "link_lengths_km",
    ):
        assert getattr(aged, name) is getattr(topology, name), name
    for link, original in zip(aged.links, topology.links):
        if link.id != 5:
            assert link is original
            continue
        assert (link.id, link.source_name, link.target_name, link.length_km) == (
            original.id,
            original.source_name,
            original.target_name,
            original.length_km,
        )
        for index, (span, original_span) in enumerate(zip(link.spans, original.spans)):
            if index != 1:
                assert span is original_span
                continue
            assert span.length_km == original_span.length_km
            assert span.input_loss_db == original_span.input_loss_db
            assert span.output_loss_db == original_span.output_loss_db
            assert span.gain_ripple is original_span.gain_ripple
            assert span.attenuation_db_per_km == original_span.attenuation_db_per_km  # None kept it
            assert span.noise_figure_db == 5.5


def test_updates_apply_in_order(topology: TopologyModel) -> None:
    aged = topology.with_span_updates(
        [
            SpanUpdate(0, 0, attenuation_db_per_km=0.3, noise_figure_db=6.0),
            SpanUpdate(0, 0, attenuation_db_per_km=0.21),
            SpanUpdate(0, 0, noise_figure_db=5.0),
        ]
    )
    span = aged.links[0].spans[0]
    assert (span.attenuation_db_per_km, span.noise_figure_db) == (0.21, 5.0)


def test_sequential_and_batched_updates_agree(topology: TopologyModel) -> None:
    updates = [SpanUpdate(1, 0, attenuation_db_per_km=0.22), SpanUpdate(2, 0, noise_figure_db=5.1)]
    batched = topology.with_span_updates(updates)
    sequential = topology.with_span_updates(updates[:1]).with_span_updates(updates[1:])
    assert batched.links == sequential.links


def test_empty_updates_return_the_model(topology: TopologyModel) -> None:
    assert topology.with_span_updates([]) is topology


@pytest.mark.parametrize(
    "update",
    [
        SpanUpdate(-1, 0, attenuation_db_per_km=0.2),
        SpanUpdate(10_000, 0, attenuation_db_per_km=0.2),
        SpanUpdate(0, 99, attenuation_db_per_km=0.2),
        SpanUpdate(0, -1, attenuation_db_per_km=0.2),
        SpanUpdate(0, 0, attenuation_db_per_km=0.0),
        SpanUpdate(0, 0, attenuation_db_per_km=-0.1),
        SpanUpdate(0, 0, attenuation_db_per_km=math.nan),
        SpanUpdate(0, 0, noise_figure_db=math.inf),
        SpanUpdate(0, 0, noise_figure_db=0.0),
    ],
)
def test_invalid_updates_raise(topology: TopologyModel, update: SpanUpdate) -> None:
    links = topology.links
    with pytest.raises(ValueError):
        topology.with_span_updates([SpanUpdate(1, 0, attenuation_db_per_km=0.3), update])
    assert topology.links is links
