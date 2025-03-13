from itertools import islice
import math
from typing import Optional, Union, Tuple, Sequence
import xml.dom.minidom
import os

import networkx as nx
import numpy as np


cdef class Span:
    cdef public double length
    cdef public double attenuation_db_km
    cdef public double attenuation_normalized
    cdef public double noise_figure_db
    cdef public double noise_figure_normalized

    def __init__(self, double length, double attenuation, double noise_figure):
        self.length = length
        self.attenuation_db_km = attenuation
        self.attenuation_normalized = self.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)  # dB/km => 1/m
        self.noise_figure_db = noise_figure
        self.noise_figure_normalized = 10 ** (self.noise_figure_db / 10)  # dB => norm

    def set_attenuation(self, double attenuation):
        self.attenuation_db_km = attenuation
        self.attenuation_normalized = self.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)

    def set_noise_figure(self, double noise_figure):
        self.noise_figure_db = noise_figure
        self.noise_figure_normalized = 10 ** (self.noise_figure_db / 10)

    def __repr__(self):
        return f"Span(length={self.length:.2f}, attenuation_db_km={self.attenuation_db_km}, noise_figure_db={self.noise_figure_db})"

cdef class Link:
    cdef public int id
    cdef public str node1
    cdef public str node2
    cdef public double length
    cdef public tuple spans  # Tuple of Span objects

    def __init__(self, int id, str node1, str node2, double length, tuple spans):
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.length = length
        self.spans = spans

    def __repr__(self):
        return f"Link(id={self.id}, node1={self.node1}, node2={self.node2}, length={self.length:.2f}, spans={self.spans})"

cdef class Modulation:
    cdef public str name
    cdef public double maximum_length
    cdef public int spectral_efficiency
    cdef public double minimum_osnr
    cdef public double inband_xt

    def __init__(self, str name, double maximum_length, int spectral_efficiency, double minimum_osnr=0.0, double inband_xt=0.0):
        self.name = name
        self.maximum_length = maximum_length
        self.spectral_efficiency = spectral_efficiency
        self.minimum_osnr = minimum_osnr
        self.inband_xt = inband_xt

    def __repr__(self):
        return (f"Modulation(name={self.name}, maximum_length={self.maximum_length}, "
                f"spectral_efficiency={self.spectral_efficiency}, minimum_osnr={self.minimum_osnr:.2f}, "
                f"inband_xt={self.inband_xt:.2f})")

cdef class Path:
    cdef public int id
    cdef public int k
    cdef public tuple node_list     # Tuple of strings
    cdef public tuple links         # Tuple of Link objects
    cdef public int hops
    cdef public double length
    cdef public Modulation best_modulation  # Can be None

    def __init__(self, int id, int k, tuple node_list, tuple links, int hops, double length, Modulation best_modulation=None):
        self.id = id
        self.k = k
        self.node_list = node_list
        self.links = links
        self.hops = hops
        self.length = length
        self.best_modulation = best_modulation

    def __repr__(self):
        return (f"Path(id={self.id}, k={self.k}, node_list={self.node_list}, "
                f"hops={self.hops}, length={self.length})")

    def get_node_list(self):
        return self.node_list




def get_k_shortest_paths(G: nx.Graph, source: str, target: str, k: int, weight=None):
    """
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    """
    return tuple(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def get_path_weight(graph, path, weight="length"):
    return np.sum([graph[path[i]][path[i + 1]][weight] for i in range(len(path) - 1)])


def get_best_modulation_format_by_length(
    length: float, modulations: Sequence[Modulation]
) -> Modulation:
    # sorts modulation from the most to the least spectrally efficient
    sorted_modulations = sorted(
        modulations, key=lambda x: x.spectral_efficiency, reverse=True
    )
    for i in range(len(modulations)):
        if length <= sorted_modulations[i].maximum_length:
            return sorted_modulations[i]
    raise ValueError(
        "It was not possible to find a suitable MF for a path with {} km".format(length)
    )

def calculate_geographical_distance(
        latlong1: tuple[float, ...],
        latlong2: tuple[float, ...]
) -> float:
    r = 6373.0

    lat1 = math.radians(latlong1[1])
    lon1 = math.radians(latlong1[0])
    lat2 = math.radians(latlong2[1])
    lon2 = math.radians(latlong2[0])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    length = r * c
    return length


def read_sndlib_topology(file_name: str) -> nx.Graph:
    graph: nx.Graph = nx.Graph()

    with open(file_name, "rt", encoding="utf-8") as file:
        tree = xml.dom.minidom.parse(file)
        document = tree.documentElement

        graph.graph["coordinatesType"] = document.getElementsByTagName(
            "nodes"
        )[0].getAttribute("coordinatesType")

        nodes = document.getElementsByTagName("node")
        for idn, node in enumerate(nodes):
            x = node.getElementsByTagName("x")[0]
            y = node.getElementsByTagName("y")[0]
            # print(node['id'], x.string, y.string)
            graph.add_node(
                node.getAttribute("id"),
                pos=((float(x.childNodes[0].data), float(y.childNodes[0].data))),
                id=idn,
            )
        # print('Total nodes: ', graph.number_of_nodes())
        links = document.getElementsByTagName("link")
        index = 0
        for link in links:
            source = link.getElementsByTagName("source")[0]
            target = link.getElementsByTagName("target")[0]

            if graph.has_edge(source.childNodes[0].data, target.childNodes[0].data):
                continue

            if graph.graph["coordinatesType"] == "geographical":
                length = np.around(
                    calculate_geographical_distance(
                        graph.nodes[source.childNodes[0].data]["pos"],
                        graph.nodes[target.childNodes[0].data]["pos"],
                    ),
                    3,
                )
            else:
                latlong1 = graph.nodes[source.childNodes[0].data]["pos"]
                latlong2 = graph.nodes[target.childNodes[0].data]["pos"]
                length = np.around(
                    math.sqrt(
                        (latlong1[0] - latlong2[0]) ** 2
                        + (latlong1[1] - latlong2[1]) ** 2
                    ),
                    3,
                )

            weight = 1.0
            graph.add_edge(
                source.childNodes[0].data,
                target.childNodes[0].data,
                id=link.getAttribute("id"),
                weight=weight,
                length=length,
                index=index,
            )
            index += 1
    #         print(source.childNodes[0].data, target.childNodes[0].data, index)
    # print(graph.number_of_edges())
    # exit(10)
    return graph


def read_txt_file(file_name: str) -> nx.Graph:
    graph: nx.Graph = nx.Graph()
    num_nodes: int = 0
    id_link: int = 0
    with open(file_name, "r", encoding="utf-8") as lines:
        # gets only lines that do not start with the # character
        nodes_lines = [value for value in lines if not value.startswith("#")]
        for idx, line in enumerate(nodes_lines):
            if idx == 0:
                num_nodes = int(line)
                for _id in range(1, num_nodes + 1):
                    graph.add_node(str(_id), name=str(_id))
            elif idx == 1:
                continue
            elif len(line) > 1:
                info = line.replace("\n", "").split(" ")
                graph.add_edge(
                    info[0],
                    info[1],
                    id=id_link,
                    index=id_link,
                    weight=1,
                    length=int(info[2]),
                )
                id_link += 1

    return graph


def get_topology(
    file_path: str,
    topology_name: str | None = None,
    modulations: Optional[Tuple[Modulation]] = None,
    max_span_length: float = 100,
    default_attenuation: float = 0.2,
    default_noise_figure: float = 4.5,
    k_paths: int = 5
) -> nx.Graph:
    """
    Generates a network topology with necessary attributes for simulation.

    Parameters:
    - file_path: Path to the topology file (.xml or .txt).
    - topology_name: Name assigned to the topology.
    - modulations: Optional tuple of modulation formats.
    - max_span_length: Maximum length of a span in kilometers.
    - default_attenuation: Default attenuation per span in dB/km.
    - default_noise_figure: Default noise figure in dB.
    - k_paths: Number of shortest paths to compute between node pairs.

    Returns:
    - topology: A NetworkX graph with additional attributes.
    """
    k_shortest_paths = {}
    max_length = 0
    min_length = 1e12

    # Read the topology from file
    if file_path.endswith(".xml"):
        topology = read_sndlib_topology(file_path)
    elif file_path.endswith(".txt"):
        topology = read_txt_file(file_path)
    else:
        raise ValueError("Supplied topology format is unknown")

    if topology_name is None:
        # Extract the filename from the path, ignoring folders and file extension
        topology_name = os.path.splitext(os.path.basename(file_path))[0]

    # Generating the spans and links
    topology.graph["has_links_object"] = True
    for node1, node2 in topology.edges():
        length = topology[node1][node2]["length"]
        num_spans = int(length // max_span_length) or 1
        if length % max_span_length != 0:
            num_spans += 1
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
            id=topology[node1][node2].get("index", f"{node1}-{node2}"),
            length=length,
            node1=node1,
            node2=node2,
            spans=tuple(spans),
        )
        topology[node1][node2]["link"] = link

    idp = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 < idn2:
                paths = get_k_shortest_paths(topology, n1, n2, k_paths, weight="length")
                lengths = [
                    get_path_weight(topology, path, weight="length") for path in paths
                ]

                if modulations is not None:
                    selected_modulations = [
                        get_best_modulation_format(length, modulations)
                        for length in lengths
                    ]
                else:
                    selected_modulations = [None for _ in lengths]

                objs = []
                k = 0
                for path, length, modulation in zip(paths, lengths, selected_modulations):
                    # Generate links along the path
                    path_links = []
                    for i in range(len(path) - 1):
                        node_u = path[i]
                        node_v = path[i + 1]
                        link = topology[node_u][node_v]["link"]
                        path_links.append(link)
                    path_links = tuple(path_links)
                    objs.append(
                        Path(
                            id=idp,
                            k=k,  
                            node_list=tuple(path),
                            hops=len(path) - 1,
                            length=length,
                            best_modulation=modulation,
                            links=path_links,
                        )
                    )
                    k += 1
                    idp += 1
                    max_length = max(max_length, length)
                    min_length = min(min_length, length)

                k_shortest_paths[n1, n2] = objs
                k_shortest_paths[n2, n1] = objs

    # Assign attributes to the topology graph
    topology.graph["name"] = topology_name
    topology.graph["ksp"] = k_shortest_paths
    if modulations is not None:
        topology.graph["modulations"] = modulations
    topology.graph["k_paths"] = k_paths
    topology.graph["node_indices"] = []

    # Assign indices to nodes
    for idx, node in enumerate(topology.nodes()):
        topology.graph["node_indices"].append(node)
        topology.nodes[node]["index"] = idx
    return topology


def get_best_modulation_format(
    length: float, modulations: Sequence[Modulation]
) -> Modulation:
    # sorts modulation from the most to the least spectrally efficient
    sorted_modulations = sorted(
        modulations, key=lambda x: x.spectral_efficiency, reverse=True
    )
    for i in range(len(modulations)):
        if length <= sorted_modulations[i].maximum_length:
            return sorted_modulations[i]
    raise ValueError(
        "It was not possible to find a suitable MF for a path with {} km".format(length)
    )
