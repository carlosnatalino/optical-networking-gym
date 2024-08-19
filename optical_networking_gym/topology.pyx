from itertools import islice
import math
from typing import Optional, Union, Tuple, Sequence
import xml.dom.minidom

import cython
import networkx as nx
import numpy as np


@cython.cclass
class Span:

    length = cython.declare(cython.double, visibility="readonly")
    attenuation_db_km = cython.declare(cython.double, visibility="readonly")
    attenuation_normalized = cython.declare(cython.double, visibility="readonly")
    noise_figure_db = cython.declare(cython.double, visibility="readonly")
    noise_figure_normalized = cython.declare(cython.double, visibility="readonly")

    def __init__(self, length: float, attenuation: float, noise_figure: float):
        self.length = length

        self.attenuation_db_km = attenuation
        self.attenuation_normalized = self.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)  # dB/km ===> 1/m

        self.noise_figure_db = noise_figure
        self.noise_figure_normalized = 10 ** (self.noise_figure_db / 10)  # dB ===> norm
    
    def set_attenuation(self, attenuation: float) -> None:
        self.attenuation_db_km = attenuation
        self.attenuation_normalized = self.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)  # dB/km ===> 1/m
    
    def set_noise_figure(self, noise_figure: float) -> None:
        self.noise_figure_db = noise_figure
        self.noise_figure_normalized = 10 ** (self.noise_figure_db / 10)  # dB ===> norm
    
    def __repr__(self) -> str:
        return f"Span(length={self.length:0.2f}, attenuation_db_km={self.attenuation_db_km}, noise_figure_db={self.noise_figure_db})"


@cython.cclass
class Link:
    id = cython.declare(cython.int, visibility="readonly")
    node1 = cython.declare(cython.str, visibility="readonly")
    node2 = cython.declare(cython.str, visibility="readonly")
    length = cython.declare(cython.double, visibility="readonly")
    spans = cython.declare(cython.tuple[Span], visibility="readonly")

    def __init__(self, id: cython.int, node1: cython.str, node2: cython.str, length: cython.double, spans: cython.tuple):
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.length = length
        self.spans = spans
    
    def __repr__(self) -> str:
        return f"Link(id={self.id}, node1={self.node1}, node2={self.node2}, length={self.length:0.2f}, spans={self.spans})"

@cython.cclass
class Modulation:
    name = cython.declare(cython.str, visibility="readonly")
    # maximum length in km
    maximum_length = cython.declare(cython.float, visibility="readonly")
    # number of bits per Hz per sec.
    spectral_efficiency = cython.declare(cython.int, visibility="readonly")
    # minimum OSNR that allows it to work
    minimum_osnr = cython.declare(cython.float, visibility="readonly")
    # maximum in-band cross-talk
    inband_xt = cython.declare(cython.float, visibility="readonly")

    def __init__(self, name: cython.str, maximum_length: cython.float, spectral_efficiency: cython.int, minimum_osnr: cython.float = 0.0, inband_xt: cython.float = 0.0) -> None:
        self.name = name
        self.maximum_length = maximum_length
        self.spectral_efficiency = spectral_efficiency
        self.minimum_osnr = minimum_osnr
        self.inband_xt = inband_xt
    
    def __repr__(self) -> str:
        return f"Modulation(name={self.name}, maximum_length={self.maximum_length}, spectral_efficiency={self.spectral_efficiency}, minimum_osnr={self.minimum_osnr:0.2f}, inband_xt={self.inband_xt:0.2f})"


@cython.cclass
class Path:
    path_id = cython.declare(cython.int, visibility="readonly")
    nodes = cython.declare(cython.tuple[cython.str], visibility="readonly")
    hops = cython.declare(cython.int, visibility="readonly")
    length = cython.declare(cython.float, visibility="readonly")
    best_modulation_by_distance = cython.declare(Modulation, visibility="readonly")
    
    def __init__(self, path_id: cython.int, nodes: cython.tuple[cython.str], length: cython.float, best_modulation_by_distance: Modulation) -> None:
        self.path_id = path_id
        self.nodes = nodes
        self.hops = len(nodes) - 1
        self.length = length
        self.best_modulation_by_distance = best_modulation_by_distance
    
    def __repr__(self) -> str:
        return f"Path(path_id={self.path_id}, nodes={self.nodes}, hops={self.hops}, length={self.length}, best_modulation_by_distance={self.best_modulation_by_distance})"


@cython.cclass
class Service:
    service_id = cython.declare(cython.int, visibility="readonly")
    source = cython.declare(cython.str, visibility="readonly")
    source_id = cython.declare(cython.int, visibility="readonly")
    destination = cython.declare(cython.str, visibility="readonly")
    destination_id = cython.declare(cython.int, visibility="readonly")
    arrival_time = cython.declare(cython.float, visibility="readonly")
    holding_time = cython.declare(cython.float, visibility="readonly")
    bit_rate = cython.declare(cython.float, visibility="readonly")
    path = cython.declare(Path, visibility="public")
    best_modulation = cython.declare(Modulation, visibility="public")
    service_class = cython.declare(cython.int, visibility="readonly")
    number_slots = cython.declare(cython.int, visibility="public")
    core = cython.declare(cython.int, visibility="public")
    launch_power = cython.declare(cython.float, visibility="public")
    accepted = cython.declare(cython.bint, visibility="public")

    # TODO: write the __init__ method

    def __str__(self):
        # TODO: improve the __str__ method
        msg = "{"
        msg += "" if self.bit_rate is None else f"br: {self.bit_rate}, "
        msg += "" if self.service_class is None else f"cl: {self.service_class}, "
        return f"Serv. {self.service_id} ({self.source} -> {self.destination})" + msg


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

    lat1 = math.radians(latlong1[0])
    lon1 = math.radians(latlong1[1])
    lat2 = math.radians(latlong2[0])
    lon2 = math.radians(latlong2[1])

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