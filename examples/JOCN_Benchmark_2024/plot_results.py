from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from optical_networking_gym_v2.defaults import resolve_topology
from optical_networking_gym_v2.network.topology import TopologyModel


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def _latest_run(results_dir: Path, family: str, script_stem: str) -> Path | None:
    run_root = results_dir / family / script_stem
    if not run_root.exists():
        return None
    runs = sorted((path for path in run_root.iterdir() if path.is_dir()), key=lambda path: path.name)
    return runs[-1] if runs else None


def _read_latest_summary(results_dir: Path, family: str, script_stem: str) -> pd.DataFrame | None:
    run_dir = _latest_run(results_dir, family, script_stem)
    if run_dir is None:
        return None
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return None
    return pd.read_csv(summary_path)


def _save(fig, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}.png")
    fig.savefig(figures_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_topology(*, topology_id: str, figures_dir: Path) -> None:
    topology = TopologyModel.from_file(resolve_topology(topology_id), topology_id=topology_id, k_paths=5)

    fig, ax = plt.subplots(figsize=(2.5, 4.8))
    ax.boxplot([link.length_km for link in topology.links])
    ax.set_ylabel("Link length [km]")
    ax.set_xticks([])
    ax.set_xlim([0.8, 1.2])
    _save(fig, figures_dir, f"{topology_id}_topo_link_length")

    fig, ax = plt.subplots(figsize=(2.5, 4.8))
    ax.boxplot([len(link.spans) for link in topology.links])
    ax.set_ylabel("Number of spans per link")
    ax.set_xticks([])
    ax.set_xlim([0.8, 1.2])
    _save(fig, figures_dir, f"{topology_id}_topo_link_spans")

    path_lengths = []
    for k_index in range(5):
        lengths = [path.length_km for path in topology.paths if path.k == k_index]
        if lengths:
            path_lengths.append(lengths)
    fig, ax = plt.subplots()
    ax.boxplot(path_lengths)
    ax.set_xlabel("Path index (k-shortest paths)")
    ax.set_ylabel("Path length [km]")
    _save(fig, figures_dir, f"{topology_id}_topo_path_length")

    graph = nx.Graph()
    for node in topology.node_names:
        graph.add_node(node)
    for link in topology.links:
        graph.add_edge(link.source_name, link.target_name, length=link.length_km)
    pos = nx.spring_layout(graph, seed=7)
    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    nx.draw_networkx_nodes(graph, pos=pos, node_color="black", node_size=40, ax=ax)
    nx.draw_networkx_edges(graph, pos=pos, ax=ax)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    _save(fig, figures_dir, f"{topology_id}_topo")


def plot_launch_power(*, results_dir: Path, figures_dir: Path, topology_id: str) -> None:
    summary = _read_latest_summary(results_dir, "JOCN_Benchmark_2024", "graph_launch_power")
    if summary is None:
        return
    x = summary["sweep_value"]

    fig, ax = plt.subplots()
    ax.plot(x, summary["service_blocking_rate_mean"], marker="o")
    ax.set_yscale("log")
    ax.grid(visible=True, which="major", axis="y", ls="--")
    ax.grid(visible=True, which="minor", axis="y", ls=":")
    ax.set_xlabel("Launch power [dBm]")
    ax.set_ylabel("Request blocking ratio")
    _save(fig, figures_dir, f"{topology_id}_lp_rbr")

    fig, ax = plt.subplots()
    ax.plot(x, summary["accepted_osnr_margin_mean_mean"], label="Accepted OSNR margin", marker="o")
    ax.plot(x, summary["final_osnr_margin_mean_mean"], label="Final OSNR margin", marker="s")
    ax.grid(visible=True, which="major", axis="y", ls="--")
    ax.grid(visible=True, which="minor", axis="y", ls=":")
    ax.set_xlabel("Launch power [dBm]")
    ax.set_ylabel("OSNR margin [dB]")
    ax.legend()
    _save(fig, figures_dir, f"{topology_id}_lp_osnr_margin")


def plot_load(*, results_dir: Path, figures_dir: Path, topology_id: str) -> None:
    summary = _read_latest_summary(results_dir, "JOCN_Benchmark_2024", "graph_load")
    if summary is None:
        return
    x = summary["sweep_value"]

    fig, ax = plt.subplots()
    ax.plot(x, summary["service_blocking_rate_mean"], marker="o", mec="white")
    ax.set_yscale("log")
    ax.grid(visible=True, which="major", axis="y", ls="--")
    ax.grid(visible=True, which="minor", axis="y", ls=":")
    ax.set_xlabel("Load [Erlang]")
    ax.set_ylabel("Request blocking rate")
    _save(fig, figures_dir, f"{topology_id}_load_rbr")

    fig, ax = plt.subplots()
    ax.plot(x, summary["bit_rate_blocking_rate_mean"], marker="s", mec="white")
    ax.set_yscale("log")
    ax.grid(visible=True, which="major", axis="y", ls="--")
    ax.grid(visible=True, which="minor", axis="y", ls=":")
    ax.set_xlabel("Load [Erlang]")
    ax.set_ylabel("Bit rate blocking rate")
    _save(fig, figures_dir, f"{topology_id}_load_brbr")

    fig, ax = plt.subplots()
    ax.plot(x, summary["episode_time_s_mean"], marker="^", mec="white")
    ax.grid(visible=True, which="major", axis="y", ls="--")
    ax.grid(visible=True, which="minor", axis="y", ls=":")
    ax.set_xlabel("Load [Erlang]")
    ax.set_ylabel("Episode processing time [s]")
    _save(fig, figures_dir, f"{topology_id}_load_episode_time")


def plot_heuristics(*, results_dir: Path, figures_dir: Path, topology_id: str) -> None:
    summary = _read_latest_summary(results_dir, "SBRT2026", "judge_heuristics_load_sweep")
    if summary is None:
        return

    markers = ("o", ">", "s", "<", "^", "v")
    fig, ax = plt.subplots()
    for index, (policy, group) in enumerate(summary.groupby("policy")):
        group = group.sort_values("load")
        ax.plot(
            group["load"],
            group["service_blocking_rate_mean"],
            label=policy,
            marker=markers[index % len(markers)],
            mec="white",
        )
    ax.set_yscale("log")
    ax.grid(visible=True, which="major", axis="y", ls="--")
    ax.grid(visible=True, which="minor", axis="y", ls=":")
    ax.set_xlabel("Load [Erlang]")
    ax.set_ylabel("Request blocking rate")
    ax.legend()
    _save(fig, figures_dir, f"{topology_id}_heuristics_rbr")

    fig, ax = plt.subplots()
    for index, (policy, group) in enumerate(summary.groupby("policy")):
        group = group.sort_values("load")
        ax.plot(
            group["load"],
            group["bit_rate_blocking_rate_mean"],
            label=policy,
            marker=markers[index % len(markers)],
            mec="white",
        )
    ax.set_yscale("log")
    ax.grid(visible=True, which="major", axis="y", ls="--")
    ax.grid(visible=True, which="minor", axis="y", ls=":")
    ax.set_xlabel("Load [Erlang]")
    ax.set_ylabel("Bit rate blocking rate")
    ax.legend()
    _save(fig, figures_dir, f"{topology_id}_heuristics_brbr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate JOCN benchmark figures from v2 result CSVs.")
    parser.add_argument("--topology-id", default="nobel-eu")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir is not None else results_dir / "figures"

    plot_topology(topology_id=args.topology_id, figures_dir=figures_dir)
    plot_launch_power(results_dir=results_dir, figures_dir=figures_dir, topology_id=args.topology_id)
    plot_load(results_dir=results_dir, figures_dir=figures_dir, topology_id=args.topology_id)
    plot_heuristics(results_dir=results_dir, figures_dir=figures_dir, topology_id=args.topology_id)
    print(f"JOCN figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
