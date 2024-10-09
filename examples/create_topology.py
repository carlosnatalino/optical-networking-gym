import argparse
import pathlib
import pickle
from typing import Optional, Tuple

import networkx as nx

from optical_networking_gym.topology import (
    Span,
    Link,
    Path,
    # get_topology,
    get_k_shortest_paths,
    get_path_weight,
    get_best_modulation_format_by_length,
    read_sndlib_topology,
    read_txt_file,
)

from optical_networking_gym.topology import Modulation

# in case you do not have modulations
cur_modulations: Optional[Tuple[Modulation]] = None

# note that minimum OSNR and in-band cross-talk are optional parameters

# example of in-band cross-talk settings for different modulation formats:
# https://ieeexplore.ieee.org/abstract/document/7541954
# table III

# defining the EON parameters
# definitions according to:
# https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/K-SP-FF%20benchmark_NSFNET.py#L268
# cur_modulations = (
#     # the first (lowest efficiency) modulation format needs to have maximum length
#     # greater or equal to the longest path in the topology.
#     # Here we put 100,000 km to be on the safe side
#     Modulation(
#         name="BPSK", maximum_length=100_000, spectral_efficiency=1, minimum_osnr=12.6, inband_xt=-14
#     ),
#     Modulation(
#         name="QPSK", maximum_length=2_000, spectral_efficiency=2, minimum_osnr=12.6, inband_xt=-17
#     ),
#     Modulation(
#         name="8QAM", maximum_length=1_250, spectral_efficiency=3, minimum_osnr=18.6, inband_xt=-20
#     ),
#     Modulation(
#         name="16QAM", maximum_length=625, spectral_efficiency=4, minimum_osnr=22.4, inband_xt=-23
#     ),
# )

# other setup:
cur_modulations: Tuple[Modulation] = (
    # the first (lowest efficiency) modulation format needs to have maximum length
    # greater or equal to the longest path in the topology.
    # Here we put 100,000 km to be on the safe side
    Modulation(
        name="BPSK",
        maximum_length=100_000,
        spectral_efficiency=1,
        minimum_osnr=12.6,
        inband_xt=-14,
    ),
    Modulation(
        name="QPSK",
        maximum_length=2_000,
        spectral_efficiency=2,
        minimum_osnr=12.6,
        inband_xt=-17,
    ),
    Modulation(
        name="8QAM",
        maximum_length=1_000,
        spectral_efficiency=3,
        minimum_osnr=18.6,
        inband_xt=-20,
    ),
    Modulation(
        name="16QAM",
        maximum_length=500,
        spectral_efficiency=4,
        minimum_osnr=22.4,
        inband_xt=-23,
    ),
    Modulation(
        name="32QAM",
        maximum_length=250,
        spectral_efficiency=5,
        minimum_osnr=26.4,
        inband_xt=-26,
    ),
    Modulation(
        name="64QAM",
        maximum_length=125,
        spectral_efficiency=6,
        minimum_osnr=30.4,
        inband_xt=-29,
    ),
)


def get_topology(
    file_path: str,
    topology_name: str,
    modulations: Optional[Tuple[Modulation]] = None,
    max_span_length: float = 100,
    default_attenuation: float = 0.2,
    default_noise_figure: float = 4.5,
    k_paths: int = 5
) -> nx.Graph:
    """
    Function
    """
    k_shortest_paths = {}
    if file_path.endswith(".xml"):
        topology = read_sndlib_topology(file_path)
    elif file_path.endswith(".txt"):
        topology = read_txt_file(file_path)
    else:
        raise ValueError("Supplied topology format is unknown")

    # generating the spans
    topology.graph["has_links_object"] = True
    for node1, node2 in topology.edges():
        length = topology[node1][node2]["length"]
        num_spans = int(length // max_span_length)
        if length % num_spans != 0:
            num_spans += 1
        # print(f"{num_spans=}")
        span_length = length / num_spans
        spans = []
        for _ in range(num_spans):
            span = Span(
                length=span_length,
                attenuation=default_attenuation,
                noise_figure=default_noise_figure,
            )
            spans.append(span)

        link = Link(
            id=topology[node1][node2]["index"],
            length=length,
            node1=node1,
            node2=node2,
            spans=tuple(spans),
        )
        topology[node1][node2]["link"] = link
        # print(link)

    idp = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 < idn2:
                paths = get_k_shortest_paths(topology, n1, n2, k_paths, weight="length")
                print(n1, n2, len(paths))
                lengths = [
                    get_path_weight(topology, path, weight="length") for path in paths
                ]
                if modulations is not None:
                    selected_modulations = [
                        get_best_modulation_format_by_length(length, modulations)
                        for length in lengths
                    ]
                else:
                    selected_modulations = [None for _ in lengths]
                objs = []

                for path, length, modulation in zip(
                    paths, lengths, selected_modulations
                ):
                    objs.append(
                        Path(
                            idp,
                            tuple(path),
                            length,
                            modulation,
                        )
                    )  # <== The topology is created and a best modulation is just automatically attached.  In our new implementation, the best modulation will be variable depending on available resources and the amount of crosstalk it will cause.
                    print("\t", objs[-1])
                    idp += 1
                k_shortest_paths[n1, n2] = objs
                k_shortest_paths[n2, n1] = objs
    topology.graph["name"] = topology_name
    topology.graph["ksp"] = k_shortest_paths
    if modulations is not None:
        topology.graph["modulations"] = modulations
    topology.graph["k_paths"] = k_paths
    topology.graph["node_indices"] = []
    for idx, node in enumerate(topology.nodes()):
        topology.graph["node_indices"].append(node)
        topology.nodes[node]["index"] = idx
    return topology


if __name__ == "__main__":
    # default values
    default_k_paths = 5
    default_topology_file = "nsfnet_chen.txt"
    default_max_span_length = 100
    default_attenuation = 0.2  # dB/km
    default_noise_figure = 4.5  # dB

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        "--k_paths",
        type=int,
        default=default_k_paths,
        help=f"Number of k-shortest-paths to be considered (default=`{default_k_paths}`)",
    )
    parser.add_argument(
        "-t",
        "--topology",
        default=default_topology_file,
        help=f"Network topology file to be used (default: `{default_topology_file}`)",
    )
    parser.add_argument(
        "-m",
        "--max_span_length",
        type=float,
        default=default_max_span_length,
        help=f"Maximum span length [km] (default: {default_max_span_length})",
    )
    parser.add_argument(
        "-a",
        "--attenuation",
        type=float,
        default=default_attenuation,
        help=f"Fiber attenuation (default: {default_attenuation})",
    )
    parser.add_argument(
        "-n",
        "--noise_figure",
        type=float,
        default=default_noise_figure,
        help=f"Amplifier noise figure (default: {default_attenuation})",
    )

    args = parser.parse_args()

    topology_path = pathlib.Path(args.topology)

    topology = get_topology(
        args.topology,
        topology_path.stem.upper(),
        cur_modulations,
        args.max_span_length,
        args.attenuation,
        args.noise_figure,
        args.k_paths,
    )

    file_name = topology_path.stem + "_" + str(args.k_paths) + "-paths"
    if cur_modulations is not None:
        file_name += "_" + str(len(cur_modulations)) + "-modulations"
    file_name += ".h5"

    output_file = topology_path.parent.resolve().joinpath(file_name)
    with open(output_file, "wb") as f:
        pickle.dump(topology, f)

    print("done for", topology)
