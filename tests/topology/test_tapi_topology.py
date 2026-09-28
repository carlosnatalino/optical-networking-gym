from __future__ import annotations

import json
from pathlib import Path

import pytest

from optical_networking_gym import TopologyModel, make_env, select_first_fit_action
from optical_networking_gym import defaults
from optical_networking_gym.network.topology import SPEED_OF_LIGHT_M_PER_S, TAPI_FIBER_GROUP_INDEX


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = PROJECT_ROOT / "examples" / "topologies"
CORONET_PATH = TOPOLOGY_DIR / "coronet_tapi_topology_context.json"


def _delay_ns(length_km: float) -> int:
    return round(length_km * 1e3 / (SPEED_OF_LIGHT_M_PER_S / TAPI_FIBER_GROUP_INDEX) * 1e9)


def _node(uuid: str, name: str) -> dict[str, object]:
    return {"uuid": uuid, "name": [{"value-name": "node-name", "value": name}], "owned-node-edge-point": []}


def _link(source: str, target: str, length_km: float | None) -> dict[str, object]:
    latency = [] if length_km is None else [
        {"traffic-property-name": "propagation-delay", "total-size": _delay_ns(length_km)}
    ]
    return {
        "uuid": f"{source}->{target}",
        "name": [{"value-name": "link-name", "value": f"{source} -> {target}"}],
        "node-edge-point": [
            {"topology-uuid": "t", "node-uuid": source, "node-edge-point-uuid": f"{source}:{target}"},
            {"topology-uuid": "t", "node-uuid": target, "node-edge-point-uuid": f"{target}:{source}"},
        ],
        "direction": "UNIDIRECTIONAL",
        "latency-characteristic": latency,
    }


def _context() -> dict[str, object]:
    """Three sites, each a transceiver attached to a ROADM by zero-delay links."""
    nodes = []
    links = []
    for site in ("A", "B", "C"):
        nodes += [_node(f"trx-{site}", f"trx {site}"), _node(f"roadm-{site}", f"roadm {site}")]
        links += [_link(f"trx-{site}", f"roadm-{site}", None), _link(f"roadm-{site}", f"trx-{site}", None)]
    links += [
        _link("roadm-A", "roadm-B", 100.0),
        _link("roadm-B", "roadm-A", 100.0),
        _link("roadm-B", "roadm-C", 200.0),
        _link("roadm-C", "roadm-B", 200.0),
        _link("roadm-C", "roadm-A", 250.0),  # a single direction is enough
    ]
    topology = {"uuid": "t", "name": [{"value-name": "network-name", "value": "abc"}], "node": nodes, "link": links}
    return {"tapi-topology:topology-context": {"topology": [topology]}}


def _write(tmp_path: Path, document: object, name: str = "abc.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_tapi_topology_has_one_node_per_roadm_and_undirected_links(tmp_path: Path) -> None:
    model = TopologyModel.from_file(_write(tmp_path, _context()), k_paths=2, max_span_length_km=80.0)

    assert model.topology_id == "abc"
    assert model.node_names == ("A", "B", "C")
    assert model.link_count == 3
    assert model.link_between("A", "B").length_km == pytest.approx(100.0, abs=1e-3)
    assert model.link_between("B", "A").id == model.link_between("A", "B").id
    assert model.link_between("B", "C").length_km == pytest.approx(200.0, abs=1e-3)
    assert model.link_between("A", "C").length_km == pytest.approx(250.0, abs=1e-3)
    assert len(model.link_between("A", "B").spans) == 2
    assert model.get_paths("A", "C")[0].node_names == ("A", "C")


def test_tapi_topology_uses_standard_physical_defaults(tmp_path: Path) -> None:
    model = TopologyModel.from_file(
        _write(tmp_path, _context()),
        k_paths=1,
        default_attenuation_db_per_km=0.21,
        default_noise_figure_db=5.0,
    )

    span = model.link_between("A", "B").spans[0]
    assert span.attenuation_db_per_km == pytest.approx(0.21)
    assert span.noise_figure_db == pytest.approx(5.0)


def test_tapi_topology_averages_the_two_directions(tmp_path: Path) -> None:
    document = _context()
    links = document["tapi-topology:topology-context"]["topology"][0]["link"]
    links[-5] = _link("roadm-A", "roadm-B", 90.0)

    model = TopologyModel.from_file(_write(tmp_path, document), k_paths=1)

    assert model.link_between("A", "B").length_km == pytest.approx(95.0, abs=1e-3)


def test_tapi_topology_accepts_the_full_common_context(tmp_path: Path) -> None:
    document = {"tapi-common:context": _context()}

    model = TopologyModel.from_file(_write(tmp_path, document), k_paths=1)

    assert model.node_names == ("A", "B", "C")


def test_tapi_topology_without_topology_context_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="topology-context"):
        TopologyModel.from_file(_write(tmp_path, {"tapi-common:context": {}}), k_paths=1)


def test_tapi_topology_with_duplicate_roadm_names_is_rejected(tmp_path: Path) -> None:
    document = _context()
    nodes = document["tapi-topology:topology-context"]["topology"][0]["node"]
    nodes[3]["name"][0]["value"] = "roadm A"

    with pytest.raises(ValueError, match="two ROADMs named 'A'"):
        TopologyModel.from_file(_write(tmp_path, document), k_paths=1)


def test_coronet_tapi_context_reads_as_coronet_conus() -> None:
    model = TopologyModel.from_file(CORONET_PATH, k_paths=1)

    assert model.node_count == 75
    assert model.link_count == 99
    assert "Abilene" in model.node_names
    assert not any(name.startswith(("trx", "roadm")) for name in model.node_names)
    # Fibre lengths of the GNPy CORONET CONUS topology the context was exported from.
    assert model.link_between("Abilene", "Dallas").length_km == pytest.approx(336.951, abs=1e-3)
    assert model.link_between("Abilene", "El_Paso").length_km == pytest.approx(761.209, abs=1e-3)
    assert model.link_lengths_km.min() == pytest.approx(24.214, abs=1e-3)
    assert model.link_lengths_km.max() == pytest.approx(1221.189, abs=1e-3)


def test_make_env_resolves_a_tapi_topology_by_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(defaults, "_TOPOLOGY_DIR", None)
    _write(tmp_path, _context())

    env = make_env(
        scenario="ring4_quickstart",
        overrides={"topology_id": "abc", "topology_dir": tmp_path},
        k_paths=2,
        episode_length=5,
    )
    _, info = env.reset(seed=1)
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(select_first_fit_action(env.action_masks()))
    assert terminated or truncated
    assert env.simulator.topology.node_names == ("A", "B", "C")
