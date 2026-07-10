"""Create and pickle a TopologyModel from a topology file or built-in name.

Port of the original create_topology.py example to the new API: the span
generation, k-shortest-path computation, and node indexing that the old script
performed inline now live inside ``TopologyModel.from_file``.

Usage:
    python examples/create_topology.py --topology nsfnet_chen -k 5
    python examples/create_topology.py \
        --topology src/optical_networking_gym/topologies/nsfnet_chen.txt -k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from optical_networking_gym import TopologyModel, resolve_topology

DEFAULT_K_PATHS = 5
DEFAULT_TOPOLOGY = "nsfnet_chen"
DEFAULT_MAX_SPAN_LENGTH = 80.0  # km
DEFAULT_ATTENUATION = 0.2  # dB/km
DEFAULT_NOISE_FIGURE = 4.5  # dB

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def resolve_topology_argument(topology: str) -> Path:
    """Accept either a filesystem path to a .xml/.txt file or a built-in name."""
    candidate = Path(topology)
    if candidate.suffix in {".xml", ".txt"}:
        if not candidate.exists():
            raise FileNotFoundError(f"Topology file not found: {candidate}")
        return candidate
    return resolve_topology(topology)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-k",
        "--k_paths",
        type=int,
        default=DEFAULT_K_PATHS,
        help=f"Number of k-shortest-paths to be considered (default=`{DEFAULT_K_PATHS}`)",
    )
    parser.add_argument(
        "-t",
        "--topology",
        default=DEFAULT_TOPOLOGY,
        help=(
            "Network topology to be used: a path to a .xml/.txt file or a "
            f"built-in topology name (default: `{DEFAULT_TOPOLOGY}`)"
        ),
    )
    parser.add_argument(
        "-m",
        "--max_span_length",
        type=float,
        default=DEFAULT_MAX_SPAN_LENGTH,
        help=f"Maximum span length [km] (default: {DEFAULT_MAX_SPAN_LENGTH})",
    )
    parser.add_argument(
        "-a",
        "--attenuation",
        type=float,
        default=DEFAULT_ATTENUATION,
        help=f"Fiber attenuation [dB/km] (default: {DEFAULT_ATTENUATION})",
    )
    parser.add_argument(
        "-n",
        "--noise_figure",
        type=float,
        default=DEFAULT_NOISE_FIGURE,
        help=f"Amplifier noise figure [dB] (default: {DEFAULT_NOISE_FIGURE})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory for the pickled topology (default: {RESULTS_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    topology_path = resolve_topology_argument(args.topology)
    topology = TopologyModel.from_file(
        topology_path,
        topology_id=topology_path.stem,
        k_paths=args.k_paths,
        max_span_length_km=args.max_span_length,
        default_attenuation_db_per_km=args.attenuation,
        default_noise_figure_db=args.noise_figure,
    )

    print(f"Topology: {topology.topology_id} (from {topology_path})")
    print(f"Nodes: {topology.node_count}")
    print(f"Links: {topology.link_count}")
    print(f"Paths: {topology.path_count} (k_paths={args.k_paths})")
    print("Spans per link:")
    for link in topology.links:
        span_lengths = ", ".join(f"{span.length_km:.1f} km" for span in link.spans)
        print(
            f"  link {link.id}: {link.source_name} <-> {link.target_name} "
            f"({link.length_km:.1f} km, {len(link.spans)} spans: {span_lengths})"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{topology_path.stem}_{args.k_paths}-paths.pkl"
    with output_file.open("wb") as handle:
        pickle.dump(topology, handle)
    print(f"Pickled TopologyModel written to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
