# Topology

## Overview

**Optical Networking Gym** is an open-source toolkit designed for benchmarking resource assignment problems in optical networks. It provides a simulation environment where Reinforcement Learning (RL) agents can optimize routing and spectrum assignment (RSA) strategies efficiently.

## Table of Contents

1. [Classes](#classes)
    - [Span](#span)
    - [Link](#link)
    - [Modulation](#modulation)
    - [Path](#path)
2. [Utility Functions](#utility-functions)
    - [get_k_shortest_paths](#get_k_shortest_paths)
    - [get_path_weight](#get_path_weight)
    - [get_best_modulation_format_by_length](#get_best_modulation_format_by_length)
    - [calculate_geographical_distance](#calculate_geographical_distance)
    - [read_sndlib_topology](#read_sndlib_topology)
    - [read_txt_file](#read_txt_file)
    - [get_topology](#get_topology)
    - [get_best_modulation_format](#get_best_modulation_format)
3. [Dependencies](#dependencies)
4. [License and Credits](#license-and-credits)

---

## Classes

### Span

```cython
cdef class Span:
    cdef public double length
    cdef public double attenuation_db_km
    cdef public double attenuation_normalized
    cdef public double noise_figure_db
    cdef public double noise_figure_normalized

    def __init__(self, double length, double attenuation, double noise_figure):
        ...
```

- **Description:** Represents a single span within a network link.
- **Attributes:**
  - `length`: Length of the span (km).
  - `attenuation_db_km`: Attenuation (dB/km).
  - `attenuation_normalized`: Normalized attenuation (1/m).
  - `noise_figure_db`: Noise figure (dB).
  - `noise_figure_normalized`: Normalized noise figure (linear scale).
- **Methods:**
  - `set_attenuation(attenuation)`: Update attenuation.
  - `set_noise_figure(noise_figure)`: Update noise figure.
  - `__repr__()`: String representation.

---

### Link

```cython
cdef class Link:
    cdef public int id
    cdef public str node1
    cdef public str node2
    cdef public double length
    cdef public tuple spans  # Tuple of Span objects

    def __init__(self, int id, str node1, str node2, double length, tuple spans):
        ...
```

- **Description:** Represents a connection between two network nodes, comprising multiple spans.
- **Attributes:**
  - `id`: Unique identifier.
  - `node1`, `node2`: Connected node identifiers.
  - `length`: Total length (km).
  - `spans`: Tuple of `Span` objects.
- **Methods:**
  - `__repr__()`: String representation.

---

### Modulation

```cython
cdef class Modulation:
    cdef public str name
    cdef public double maximum_length
    cdef public int spectral_efficiency
    cdef public double minimum_osnr
    cdef public double inband_xt

    def __init__(self, str name, double maximum_length, int spectral_efficiency, double minimum_osnr=0.0, double inband_xt=0.0):
        ...
```

- **Description:** Defines a modulation format used in optical transmissions.
- **Attributes:**
  - `name`: Modulation name.
  - `maximum_length`: Maximum feasible transmission length (km).
  - `spectral_efficiency`: Bits per second per Hz.
  - `minimum_osnr`: Minimum required OSNR (dB).
  - `inband_xt`: In-band cross-talk factor.
- **Methods:**
  - `__repr__()`: String representation.

---

### Path

```cython
cdef class Path:
    cdef public int id
    cdef public int k
    cdef public tuple node_list     # Tuple of strings
    cdef public tuple links         # Tuple of Link objects
    cdef public int hops
    cdef public double length
    cdef public Modulation best_modulation  # Can be None

    def __init__(self, int id, int k, tuple node_list, tuple links, int hops, double length, Modulation best_modulation=None):
        ...
```

- **Description:** Represents a specific route through the network, consisting of multiple links.
- **Attributes:**
  - `id`: Unique identifier.
  - `k`: Path rank (e.g., 1st shortest).
  - `node_list`: Sequence of nodes in the path.
  - `links`: Tuple of `Link` objects in the path.
  - `hops`: Number of hops (links).
  - `length`: Total path length (km).
  - `best_modulation`: Optimal modulation format for the path.
- **Methods:**
  - `get_node_list()`: Returns the sequence of nodes.
  - `__repr__()`: String representation.

---

## Utility Functions

### get_k_shortest_paths

```python
def get_k_shortest_paths(G: nx.Graph, source: str, target: str, k: int, weight=None):
    ...
```

- **Description:** Retrieves the top `k` shortest simple paths between two nodes in the graph.
- **Parameters:**
  - `G`: NetworkX graph.
  - `source`, `target`: Node identifiers.
  - `k`: Number of paths to retrieve.
  - `weight`: Edge attribute to use as weight.
- **Returns:** Tuple of paths.

---

### get_path_weight

```python
def get_path_weight(graph, path, weight="length"):
    ...
```

- **Description:** Calculates the total weight of a given path based on a specified edge attribute.
- **Parameters:**
  - `graph`: NetworkX graph.
  - `path`: Sequence of nodes representing the path.
  - `weight`: Edge attribute to sum.
- **Returns:** Total weight (float).

---

### get_best_modulation_format_by_length

```python
def get_best_modulation_format_by_length(length: float, modulations: Sequence[Modulation]) -> Modulation:
    ...
```

- **Description:** Selects the most spectrally efficient modulation format that can support a given path length.
- **Parameters:**
  - `length`: Path length (km).
  - `modulations`: Available modulation formats.
- **Returns:** Suitable `Modulation` object.
- **Raises:** `ValueError` if no suitable modulation is found.

---

### calculate_geographical_distance

```python
def calculate_geographical_distance(latlong1: tuple[float, ...], latlong2: tuple[float, ...]) -> float:
    ...
```

- **Description:** Computes the geographical distance between two latitude-longitude coordinates using the Haversine formula.
- **Parameters:**
  - `latlong1`, `latlong2`: Tuples containing latitude and longitude.
- **Returns:** Distance in kilometers (float).

---

### read_sndlib_topology

```python
def read_sndlib_topology(file_name: str) -> nx.Graph:
    ...
```

- **Description:** Parses an SNDLib XML topology file and constructs a NetworkX graph with nodes and links.
- **Parameters:**
  - `file_name`: Path to the XML topology file.
- **Returns:** NetworkX graph with added attributes.

---

### read_txt_file

```python
def read_txt_file(file_name: str) -> nx.Graph:
    ...
```

- **Description:** Reads a custom TXT topology file and constructs a NetworkX graph.
- **Parameters:**
  - `file_name`: Path to the TXT topology file.
- **Returns:** NetworkX graph with added attributes.

---

### get_topology

```python
def get_topology(
    file_path: str,
    topology_name: str,
    modulations: Optional[Tuple[Modulation]] = None,
    max_span_length: float = 100,
    default_attenuation: float = 0.2,
    default_noise_figure: float = 4.5,
    k_paths: int = 5
) -> nx.Graph:
    ...
```

- **Description:** Generates a network topology with spans, links, and precomputed k-shortest paths.
- **Parameters:**
  - `file_path`: Path to the topology file (.xml or .txt).
  - `topology_name`: Name assigned to the topology.
  - `modulations`: Optional tuple of modulation formats.
  - `max_span_length`: Maximum span length (km).
  - `default_attenuation`: Default attenuation per span (dB/km).
  - `default_noise_figure`: Default noise figure (dB).
  - `k_paths`: Number of shortest paths to compute.
- **Returns:** NetworkX graph with added attributes and paths.

---

### get_best_modulation_format

```python
def get_best_modulation_format(length: float, modulations: Sequence[Modulation]) -> Modulation:
    ...
```

- **Description:** Selects the most spectrally efficient modulation format that can support a given path length.
- **Parameters:**
  - `length`: Path length (km).
  - `modulations`: Available modulation formats.
- **Returns:** Suitable `Modulation` object.
- **Raises:** `ValueError` if no suitable modulation is found.

---

## Dependencies

- **Cython**: For performance optimization.
- **NetworkX**: Graph operations and network topology management.
- **NumPy**: Numerical computations and array manipulations.
- **Heapq**: Priority queue management for event scheduling.
- **Math**: Mathematical functions.
- **Typing**: Type annotations for better code clarity.

Ensure all dependencies are installed, preferably within a virtual environment:

```bash
pip install cython networkx numpy
```

---

## License and Credits

**Optical Networking Gym** is licensed under the [MIT License](https://github.com/carlosnatalino/optical-networking-gym/blob/main/LICENSE). Special thanks to all contributors and the open-source community for their support and contributions.

---