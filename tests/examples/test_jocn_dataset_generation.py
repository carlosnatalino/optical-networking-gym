"""Tests of the JOCN 2024 QoT dataset script (examples/JOCN_Benchmark_2024/generate_dataset.py)."""

from __future__ import annotations

import csv
import json
import runpy
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("h5netcdf")
pytest.importorskip("h5py")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "examples" / "JOCN_Benchmark_2024" / "generate_dataset.py"
RUN_ID = "20260102-030405"


@pytest.fixture(scope="module")
def script() -> dict[str, object]:
    return runpy.run_path(str(SCRIPT))


@pytest.fixture(scope="module")
def run_dir(script, tmp_path_factory) -> Path:
    experiment = script["DatasetExperiment"](
        arrivals=300,
        warmup=100,
        output_dir=tmp_path_factory.mktemp("datasets"),
    )
    return script["run"](experiment, now=datetime(2026, 1, 2, 3, 4, 5))


def _open(run_dir: Path, policy: str):
    (path,) = run_dir.glob(f"jocn2024_qot_nobel-eu_210erl_{policy}_seed*_{RUN_ID}.nc")
    return xr.open_dataset(path, engine="h5netcdf")


def test_run_writes_one_file_per_heuristic_and_a_summary(script, run_dir: Path) -> None:
    assert run_dir.name == RUN_ID
    policies = script["DEFAULT_POLICIES"]
    with (run_dir / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["policy"] for row in rows] == list(policies)
    for row in rows:
        assert (run_dir / row["file"]).is_file()
        assert row["policy"] in row["file"] and RUN_ID in row["file"]
        assert int(row["arrivals"]) == 300
        assert 0 < int(row["accepted"]) <= 300
        shares = sum(float(value) for key, value in row.items() if key.startswith("share_") and "gsnr" not in key)
        assert shares == pytest.approx(1.0)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["files"] == [row["file"] for row in rows]


def test_dataset_is_consistent_with_the_noise_breakdown(run_dir: Path) -> None:
    ds = _open(run_dir, "LS-BM-KSP")
    assert ds.sizes["lightpath"] == ds.attrs["lightpaths_accepted"]
    assert np.all(ds.request_index.values >= 100)  # warm-up arrivals are not recorded
    # Per-link ASE + NLI adds up to the stored GSNR and its components.
    ase = np.nansum(ds.hop_ase_nsr.values, axis=1)
    nli = np.nansum(ds.hop_nli_nsr.values, axis=1)
    np.testing.assert_allclose(ds.gsnr_db.values, -10 * np.log10(ase + nli), atol=1e-9)
    np.testing.assert_allclose(ds.snr_ase_db.values, -10 * np.log10(ase), atol=1e-9)
    np.testing.assert_allclose(ds.snr_nli_db.values, -10 * np.log10(nli), atol=1e-9)
    # Route padding and the ragged co-propagating table agree with the per-hop counts.
    valid = ds.hop_link.values >= 0
    assert np.array_equal(valid.sum(axis=1), ds.hops.values)
    counts = np.zeros(ds.hop_link.shape, dtype=np.int64)
    np.add.at(counts, (ds.copropagating_lightpath.values, ds.copropagating_hop.values), 1)
    assert np.array_equal(counts[valid], ds.hop_copropagating.values[valid])
    # Every sample meets the threshold of its modulation format.
    thresholds = ds.modulation_gsnr_threshold_db.values[ds.modulation_index.values]
    assert np.all(ds.gsnr_db.values >= thresholds)
    np.testing.assert_allclose(ds.gsnr_margin_db.values, ds.gsnr_db.values - thresholds)
    assert ds.hop_link.dtype == np.int16 and ds.modulation_index.dtype == np.int8
    assert ds.attrs["policy_paper_name"] == "BM-LS-KSP"


def test_dataset_gsnr_includes_the_copropagating_channels(run_dir: Path) -> None:
    ds = _open(run_dir, "LS-BM-KSP")
    assert ds.attrs["qot_nli_include_interferers"] == 1
    assert ds.attrs["qot_nli_interferer_psd"] == "cut"
    assert ds.attrs["qot_nli_modulation_correction"] == "egn_xci"
    assert "query" not in ds.dims and ds.attrs["qot_query_requests"] == 0
    valid = ds.hop_link.values >= 0
    sci, xci, nli = (ds[name].values[valid] for name in ("hop_sci_nsr", "hop_xci_nsr", "hop_nli_nsr"))
    np.testing.assert_allclose(sci + xci, nli, rtol=1e-12)
    # XCI appears exactly on the hops shared with other lightpaths (at 210
    # Erlang almost every hop is shared).
    shared = ds.hop_copropagating.values[valid] > 0
    assert shared.any()
    assert np.all(xci[shared] > 0.0)
    assert np.all(xci[~shared] == 0.0)


def test_dataset_is_self_contained(script, run_dir: Path) -> None:
    ds = _open(run_dir, "KSP-LB-BM")
    # The topology object is rebuilt from the file alone.
    topology = script["load_topology"](ds)
    assert topology.node_names == tuple(ds.node.values)
    assert len(topology.links) == ds.sizes["link"]
    np.testing.assert_allclose([link.length_km for link in topology.links], ds.link_length_km.values)
    assert len(topology.paths) == ds.sizes["path"]
    for record in topology.paths:
        stored = ds.path_links.values[record.id]
        assert tuple(stored[stored >= 0]) == record.link_ids
    # Each sample's route is the stored path (in either direction).
    np.testing.assert_allclose(ds.route_length_km.values, ds.path_length_km.values[ds.path_id.values])
    assert np.array_equal(ds.hops.values, ds.path_hops.values[ds.path_id.values])
    # path_id + path_reversed give the exact traversal order, and hop 0 leaves the source.
    assert set(np.unique(ds.path_reversed.values)) <= {0, 1}
    for i in range(ds.sizes["lightpath"]):
        route = ds.hop_link.values[i]
        route = route[route >= 0]
        stored = ds.path_links.values[ds.path_id.values[i]]
        stored = stored[stored >= 0]
        nodes = ds.path_nodes.values[ds.path_id.values[i]]
        nodes = nodes[nodes >= 0]
        if ds.path_reversed.values[i]:
            stored, nodes = stored[::-1], nodes[::-1]
        assert list(route) == list(stored)
        assert nodes[0] == ds.source.values[i] and nodes[-1] == ds.destination.values[i]
        first = route[0]
        assert ds.source.values[i] in (ds.link_source.values[first], ds.link_target.values[first])
    config = json.loads(ds.attrs["scenario_config_json"])
    assert config["topology_id"] == "nobel-eu" and config["load"] == 210.0
    assert np.all(np.isfinite(ds.node_x.values)) and np.all(np.isfinite(ds.node_y.values))
    assert ds.attrs["run_id"] == RUN_ID


def test_same_seed_gives_the_same_arrival_sequence_for_all_policies(run_dir: Path) -> None:
    first = _open(run_dir, "LS-BM-KSP")
    second = _open(run_dir, "KSP-LB-BM")
    common = np.intersect1d(first.request_index.values, second.request_index.values)
    lookup_first = dict(zip(first.request_index.values, first.source.values))
    lookup_second = dict(zip(second.request_index.values, second.source.values))
    assert common.size > 0
    assert all(lookup_first[i] == lookup_second[i] for i in common)


@pytest.fixture(scope="module")
def query_runs(script, tmp_path_factory) -> dict[float, Path]:
    runs = {}
    for fraction in (0.0, 1.0):
        experiment = script["DatasetExperiment"](
            arrivals=120,
            warmup=40,
            policies=("KSP-FF-BM",),
            query_fraction=fraction,
            output_dir=tmp_path_factory.mktemp(f"queries{fraction:g}"),
        )
        runs[fraction] = script["run"](experiment, now=datetime(2026, 1, 2, 3, 4, 5))
    return runs


def _open_policy(run_dir: Path, policy: str):
    (path,) = run_dir.glob(f"jocn2024_qot_nobel-eu_210erl_{policy}_seed*_{RUN_ID}.nc")
    return xr.open_dataset(path, engine="h5netcdf")


def test_query_table_holds_the_qot_the_rmsa_evaluated(query_runs: dict[float, Path]) -> None:
    ds = _open_policy(query_runs[1.0], "KSP-FF-BM")
    assert ds.attrs["experiment_query_fraction"] == 1.0
    assert ds.attrs["qot_query_requests"] == ds.attrs["arrivals_recorded"] == 120
    requests = ds.query_request_index.values
    assert requests.size > 0 and np.all(requests >= 40)
    thresholds = ds.modulation_gsnr_threshold_db.values[ds.query_modulation_index.values]
    np.testing.assert_allclose(ds.query_gsnr_margin_db.values, ds.query_gsnr_db.values - thresholds)
    np.testing.assert_array_equal(ds.query_qot_feasible.values, ds.query_gsnr_db.values >= thresholds)
    # Candidates that became lightpaths have the GSNR recorded at establishment.
    queried = {
        (int(r), int(p), int(m), int(s)): (g, f)
        for r, p, m, s, g, f in zip(
            requests,
            ds.query_path_id.values,
            ds.query_modulation_index.values,
            ds.query_slot_start.values,
            ds.query_gsnr_db.values,
            ds.query_qot_feasible.values,
        )
    }
    matches = 0
    for r, p, m, s, g in zip(
        ds.request_index.values, ds.path_id.values, ds.modulation_index.values, ds.slot_start.values, ds.gsnr_db.values
    ):
        key = (int(r), int(p), int(m), int(s))
        if key in queried:
            matches += 1
            assert queried[key][0] == pytest.approx(g, abs=1e-4)  # float32 margins
            assert queried[key][1] == 1
    assert matches > 0


def test_queries_do_not_change_the_traffic(query_runs: dict[float, Path]) -> None:
    with_queries = _open_policy(query_runs[1.0], "KSP-FF-BM")
    without = _open_policy(query_runs[0.0], "KSP-FF-BM")
    assert "query" not in without.dims
    np.testing.assert_array_equal(with_queries.request_index.values, without.request_index.values)
    np.testing.assert_array_equal(with_queries.gsnr_db.values, without.gsnr_db.values)
