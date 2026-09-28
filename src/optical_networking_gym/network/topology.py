from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import math
import xml.dom.minidom

import networkx as nx
import numpy as np

from .equipment import GainRipple


def _canonical_pair(node_a: str, node_b: str, node_index_by_name: dict[str, int]) -> tuple[str, str]:
    if node_index_by_name[node_a] <= node_index_by_name[node_b]:
        return (node_a, node_b)
    return (node_b, node_a)


def _path_weight(graph: nx.Graph, path: list[str], weight: str = "length") -> float:
    return float(
        np.sum([graph[path[index]][path[index + 1]][weight] for index in range(len(path) - 1)])
    )


def _read_txt_topology(file_path: Path) -> nx.Graph:
    graph = nx.Graph()
    with file_path.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    node_count = int(lines[0].strip())
    for node_id in range(1, node_count + 1):
        graph.add_node(str(node_id), name=str(node_id))
    link_id = 0
    edge_lines = lines[1:]
    if edge_lines and len(edge_lines[0].strip().split()) == 1:
        edge_lines = edge_lines[1:]
    for raw_line in edge_lines:
        line = raw_line.strip()
        if not line:
            continue
        source, target, length = line.split(" ")[:3]
        graph.add_edge(
            source,
            target,
            id=link_id,
            index=link_id,
            weight=1.0,
            length=float(length),
        )
        link_id += 1
    return graph


def _geo_distance(latlong1: tuple[float, float], latlong2: tuple[float, float]) -> float:
    radius_km = 6373.0
    lat1 = math.radians(latlong1[1])
    lon1 = math.radians(latlong1[0])
    lat2 = math.radians(latlong2[1])
    lon2 = math.radians(latlong2[0])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_km * c


def _element_text(element: xml.dom.minidom.Element) -> str:
    first_child = element.childNodes[0]
    if not isinstance(first_child, xml.dom.minidom.Text):
        raise ValueError(f"expected text content in <{element.tagName}> element")
    return str(first_child.data)


def _read_sndlib_topology(file_path: Path) -> nx.Graph:
    graph = nx.Graph()
    with file_path.open("rt", encoding="utf-8") as handle:
        tree = xml.dom.minidom.parse(handle)
    document = tree.documentElement
    if document is None:
        raise ValueError(f"invalid SNDlib topology file: {file_path}")
    coordinates_type = document.getElementsByTagName("nodes")[0].getAttribute("coordinatesType")
    graph.graph["coordinatesType"] = coordinates_type
    for node_index, node in enumerate(document.getElementsByTagName("node")):
        x = float(_element_text(node.getElementsByTagName("x")[0]))
        y = float(_element_text(node.getElementsByTagName("y")[0]))
        graph.add_node(node.getAttribute("id"), pos=(x, y), id=node_index)
    edge_index = 0
    for link in document.getElementsByTagName("link"):
        source = _element_text(link.getElementsByTagName("source")[0])
        target = _element_text(link.getElementsByTagName("target")[0])
        if graph.has_edge(source, target):
            continue
        if coordinates_type == "geographical":
            length = round(_geo_distance(graph.nodes[source]["pos"], graph.nodes[target]["pos"]), 3)
        else:
            x1, y1 = graph.nodes[source]["pos"]
            x2, y2 = graph.nodes[target]["pos"]
            length = round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 3)
        graph.add_edge(
            source,
            target,
            id=link.getAttribute("id"),
            index=edge_index,
            weight=1.0,
            length=length,
        )
        edge_index += 1
    return graph


@dataclass(frozen=True, slots=True)
class Span:
    """One fibre span followed by the amplifier that compensates its loss.

    Attributes:
        length_km: Fibre length.
        attenuation_db_per_km: Fibre loss coefficient.
        noise_figure_db: Noise figure of the amplifier at the end of the span.
        input_loss_db: Lumped loss (connectors, splices, patch panels) before
            the fibre; it lowers the power launched into the fibre.
        output_loss_db: Lumped loss after the fibre, before the amplifier.
        amplifier_type: Equipment type of the amplifier (informational).
        gain_ripple: Wavelength-dependent gain deviation of the amplifier, or
            ``None`` for a flat gain.
    """

    length_km: float
    attenuation_db_per_km: float
    noise_figure_db: float
    input_loss_db: float = 0.0
    output_loss_db: float = 0.0
    amplifier_type: str | None = None
    gain_ripple: GainRipple | None = None

    @property
    def attenuation_normalized(self) -> float:
        return float(self.attenuation_db_per_km / (2 * 10 * np.log10(np.exp(1)) * 1e3))

    @property
    def noise_figure_normalized(self) -> float:
        return float(10 ** (self.noise_figure_db / 10.0))

    @property
    def input_loss_linear(self) -> float:
        return float(10 ** (self.input_loss_db / 10.0))

    @property
    def output_loss_linear(self) -> float:
        return float(10 ** (self.output_loss_db / 10.0))

    @property
    def total_loss_db(self) -> float:
        """Fibre plus lumped losses, i.e. the gain the amplifier must provide."""
        return float(self.length_km * self.attenuation_db_per_km + self.input_loss_db + self.output_loss_db)


@dataclass(frozen=True, slots=True)
class Link:
    id: int
    source_name: str
    target_name: str
    source_index: int
    target_index: int
    length_km: float
    spans: tuple[Span, ...]


@dataclass(frozen=True, slots=True)
class PathRecord:
    id: int
    k: int
    node_names: tuple[str, ...]
    node_indices: tuple[int, ...]
    link_ids: tuple[int, ...]
    hops: int
    length_km: float


def _reverse_path_record(path_record: PathRecord) -> PathRecord:
    return PathRecord(
        id=path_record.id,
        k=path_record.k,
        node_names=tuple(reversed(path_record.node_names)),
        node_indices=tuple(reversed(path_record.node_indices)),
        link_ids=tuple(reversed(path_record.link_ids)),
        hops=path_record.hops,
        length_km=path_record.length_km,
    )


@dataclass(frozen=True, slots=True)
class TopologyModel:
    topology_id: str
    node_names: tuple[str, ...]
    links: tuple[Link, ...]
    paths: tuple[PathRecord, ...]
    path_index_by_endpoints: dict[tuple[str, str], tuple[PathRecord, ...]]
    node_index_by_name: dict[str, int]
    link_id_by_endpoints: dict[tuple[str, str], int]
    link_lengths_km: np.ndarray

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        *,
        topology_id: str | None = None,
        k_paths: int = 5,
        max_span_length_km: float = 100.0,
        default_attenuation_db_per_km: float = 0.2,
        default_noise_figure_db: float = 4.5,
    ) -> "TopologyModel":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".txt":
            graph = _read_txt_topology(path)
        elif path.suffix == ".xml":
            graph = _read_sndlib_topology(path)
        else:
            raise ValueError("Unsupported topology file format")
        resolved_topology_id = topology_id or path.stem
        node_names = tuple(graph.nodes())
        node_index_by_name = {name: index for index, name in enumerate(node_names)}

        sorted_edges = sorted(
            graph.edges(data=True),
            key=lambda edge_data: int(edge_data[2].get("index", edge_data[2]["id"])),
        )
        links: list[Link] = []
        link_id_by_endpoints: dict[tuple[str, str], int] = {}
        for source_name, target_name, edge_data in sorted_edges:
            length_km = float(edge_data["length"])
            span_count = int(length_km // max_span_length_km) or 1
            if length_km % max_span_length_km != 0:
                span_count += 1
            span_length_km = length_km / span_count
            spans = tuple(
                Span(
                    length_km=span_length_km,
                    attenuation_db_per_km=default_attenuation_db_per_km,
                    noise_figure_db=default_noise_figure_db,
                )
                for _ in range(span_count)
            )
            link = Link(
                id=int(edge_data.get("index", edge_data["id"])),
                source_name=source_name,
                target_name=target_name,
                source_index=node_index_by_name[source_name],
                target_index=node_index_by_name[target_name],
                length_km=length_km,
                spans=spans,
            )
            links.append(link)
            canonical_pair = _canonical_pair(source_name, target_name, node_index_by_name)
            link_id_by_endpoints[canonical_pair] = link.id

        paths: list[PathRecord] = []
        path_index_by_endpoints: dict[tuple[str, str], tuple[PathRecord, ...]] = {}
        next_path_id = 0
        for source_index, source_name in enumerate(node_names):
            for target_index, target_name in enumerate(node_names):
                if source_index >= target_index:
                    continue
                simple_paths = tuple(
                    islice(
                        nx.shortest_simple_paths(graph, source_name, target_name, weight="length"),
                        k_paths,
                    )
                )
                endpoint_paths: list[PathRecord] = []
                for k_index, node_path in enumerate(simple_paths):
                    path_links: list[int] = []
                    for index in range(len(node_path) - 1):
                        endpoint_key = _canonical_pair(
                            node_path[index],
                            node_path[index + 1],
                            node_index_by_name,
                        )
                        path_links.append(link_id_by_endpoints[endpoint_key])
                    path_record = PathRecord(
                        id=next_path_id,
                        k=k_index,
                        node_names=tuple(node_path),
                        node_indices=tuple(node_index_by_name[name] for name in node_path),
                        link_ids=tuple(path_links),
                        hops=len(node_path) - 1,
                        length_km=_path_weight(graph, list(node_path), weight="length"),
                    )
                    next_path_id += 1
                    paths.append(path_record)
                    endpoint_paths.append(path_record)
                frozen_paths = tuple(endpoint_paths)
                path_index_by_endpoints[(source_name, target_name)] = frozen_paths
                # Legacy V1 stores undirected K-shortest paths under both endpoint
                # keys without reversing node/link order. Keep the same canonical
                # orientation here so parity traces stay byte-for-byte comparable.
                path_index_by_endpoints[(target_name, source_name)] = frozen_paths

        link_lengths_km = np.array([link.length_km for link in links], dtype=np.float64)
        return cls(
            topology_id=resolved_topology_id,
            node_names=node_names,
            links=tuple(links),
            paths=tuple(paths),
            path_index_by_endpoints=path_index_by_endpoints,
            node_index_by_name=node_index_by_name,
            link_id_by_endpoints=link_id_by_endpoints,
            link_lengths_km=link_lengths_km,
        )

    @property
    def node_count(self) -> int:
        return len(self.node_names)

    @property
    def link_count(self) -> int:
        return len(self.links)

    @property
    def path_count(self) -> int:
        return len(self.paths)

    def get_node_index(self, node_name: str) -> int:
        return self.node_index_by_name[node_name]

    def get_paths(self, source_name: str, target_name: str) -> tuple[PathRecord, ...]:
        return self.path_index_by_endpoints[(source_name, target_name)]

    def get_paths_by_ids(self, source_id: int, target_id: int) -> tuple[PathRecord, ...]:
        return self.get_paths(self.node_names[source_id], self.node_names[target_id])

    def link_by_id(self, link_id: int) -> Link:
        return self.links[link_id]

    def link_between(self, source_name: str, target_name: str) -> Link:
        link_id = self.link_id_by_endpoints[
            _canonical_pair(source_name, target_name, self.node_index_by_name)
        ]
        return self.link_by_id(link_id)

    def path_from_link_ids(self, link_ids: Sequence[int], *, path_id: int = -1) -> PathRecord:
        """A ``PathRecord`` for an arbitrary route, e.g. a sub-path or the output
        of an external planner, given its links in the order of travel.

        The nodes follow from the links (a single link is taken from its source
        to its target). ``k`` is -1 and ``id`` is ``path_id`` (-1 by default):
        such records are not part of :attr:`paths`, and the QoT engine and the
        runtime state identify a path by its links, not by its id.

        Raises:
            ValueError: If the list is empty, a link id is unknown, or two
                consecutive links do not share a node.
        """
        links_tuple = tuple(int(link_id) for link_id in link_ids)
        if not links_tuple:
            raise ValueError("a path needs at least one link")
        for link_id in links_tuple:
            if not 0 <= link_id < len(self.links):
                raise ValueError(f"unknown link id {link_id}")
        links = [self.links[link_id] for link_id in links_tuple]
        first = links[0]
        if len(links) == 1:
            node_indices = [first.source_index, first.target_index]
        else:
            second_ends = {links[1].source_index, links[1].target_index}
            if first.target_index in second_ends:
                node_indices = [first.source_index, first.target_index]
            elif first.source_index in second_ends:
                node_indices = [first.target_index, first.source_index]
            else:
                raise ValueError(f"links {links_tuple[0]} and {links_tuple[1]} do not share a node")
        for position, link in enumerate(links[1:], start=1):
            current = node_indices[-1]
            if link.source_index == current:
                node_indices.append(link.target_index)
            elif link.target_index == current:
                node_indices.append(link.source_index)
            else:
                raise ValueError(
                    f"links {links_tuple[position - 1]} and {links_tuple[position]} do not share a node"
                )
        return PathRecord(
            id=path_id,
            k=-1,
            node_names=tuple(self.node_names[index] for index in node_indices),
            node_indices=tuple(node_indices),
            link_ids=links_tuple,
            hops=len(links_tuple),
            length_km=float(sum(link.length_km for link in links)),
        )
