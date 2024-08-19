from itertools import islice
from typing import Optional, Union, Tuple, Sequence

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
