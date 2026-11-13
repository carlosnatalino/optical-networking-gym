# Topology

A `TopologyModel` is the static description of the network: its nodes, the
undirected fibre links between them with their spans, and the K-shortest
paths of every node pair. It is immutable; the simulator and the QoT engine
read it, and changes to the physical layer (for example aging) produce a new
model.

```python
from optical_networking_gym import TopologyModel, resolve_topology

topology = TopologyModel.from_file(resolve_topology("nobel-eu"), k_paths=5)
print(topology.node_count, topology.link_count, topology.path_count)  # 28 41 1890
```

The classes live in `optical_networking_gym.network.topology` and are also
exported by `optical_networking_gym.network` and the top-level package.

## Table of Contents

1. [Loading a topology](#loading-a-topology)
2. [File formats](#file-formats)
    - [SNDlib (`.xml`)](#sndlib-xml)
    - [Plain text (`.txt`)](#plain-text-txt)
    - [T-API topology context (`.json`)](#t-api-topology-context-json)
3. [Built-in topologies](#built-in-topologies)
4. [Spans](#spans)
5. [Paths](#paths)
6. [Data classes](#data-classes)
    - [TopologyModel](#topologymodel)
    - [Link](#link)
    - [Span](#span)
    - [PathRecord](#pathrecord)
    - [SpanUpdate](#spanupdate)

---

## Loading a topology

```python
TopologyModel.from_file(
    file_path: str | Path,
    *,
    topology_id: str | None = None,       # defaults to the file stem
    k_paths: int = 5,
    max_span_length_km: float = 100.0,
    default_attenuation_db_per_km: float = 0.2,
    default_noise_figure_db: float = 4.5,
) -> TopologyModel
```

The format is chosen by the file extension (see [File formats](#file-formats));
any other extension raises `ValueError`, and a missing file raises
`FileNotFoundError`. None of the formats describe the physical layer, so every
span gets `default_attenuation_db_per_km` and `default_noise_figure_db`. A
heterogeneous physical layer (per-span loss, NF, lumped losses, gain ripple)
is applied afterwards with `apply_inventory`; see
[Per-span inventory](physical_layer.md#per-span-inventory).

To load a topology by name, `resolve_topology(name)` looks for `<name>.xml`,
`<name>.txt` and `<name>.json`, in that order. It searches the built-in
directory (`optical_networking_gym.defaults.BUILTIN_TOPOLOGY_DIR`), or the
directory set with `set_topology_dir(path)`. It raises `FileNotFoundError` if
none of them exists.

`make_env` and the scenario presets do this for you. The topology fields of
`ScenarioConfig` (`topology_id`, `topology_dir`, `k_paths`,
`max_span_length_km`, `default_attenuation_db_per_km`,
`default_noise_figure_db`) are passed to `resolve_topology` and `from_file`:

```python
from optical_networking_gym import make_env

env = make_env(topology_name="nsfnet_chen", k_paths=3)

# Your own file, through a preset:
env = make_env(
    scenario="nobel_eu_baseline",
    overrides={
        "topology_id": "coronet_tapi_topology_context",
        "topology_dir": "examples/topologies",
    },
)
```

`examples/create_topology.py` builds a model from a file or a built-in name
and pickles it:

```bash
python examples/create_topology.py --topology nsfnet_chen -k 5
python examples/create_topology.py \
    --topology examples/topologies/coronet_tapi_topology_context.json -k 5 -m 80
```

Its options are `-k/--k_paths`, `-t/--topology`, `-m/--max_span_length` (km,
default 80), `-a/--attenuation` (dB/km), `-n/--noise_figure` (dB) and
`-o/--output-dir`.

---

## File formats

All three formats are read into the same undirected graph: one node per site
and at most one link per node pair, with a length in km. Link ids follow the
order of the links in the file, starting at 0.

### SNDlib (`.xml`)

An [SNDlib](http://sndlib.zib.de) network in XML.

- **Nodes:** the `<node>` elements, named by their `id` attribute.
- **Links:** the `<link>` elements. A link between a pair that already has one
  (in either direction) is skipped.
- **Length:** computed from the node coordinates and rounded to 3 decimals.
  With `coordinatesType="geographical"`, it is the great-circle distance
  (haversine, Earth radius 6373 km) between the `(x, y)` = (longitude,
  latitude) positions. Otherwise it is the Euclidean distance.

### Plain text (`.txt`)

```text
# Lines starting with '#' are ignored.
4
1 2 150
2 3 450
3 4 350
4 1 250
```

- The first line is the number of nodes `N`. The nodes are named `"1"` to
  `"N"`.
- An optional second line with a single number (the link count, as in
  `nsfnet_chen.txt`) is skipped.
- Every other non-empty line is `source target length_km`, separated by
  single spaces. Extra columns are ignored.

### T-API topology context (`.json`)

A ONF T-API topology context (`tapi-topology:topology-context`), optionally
wrapped in `tapi-common:context`, as exported by
[TwinLight](https://github.com/carlosnatalino/TwinLight). Only the first
topology of the context is read.

- **Nodes:** one per ROADM, i.e. the nodes at the end of a fibre link, named
  by their `node-name` without the `roadm ` prefix (`roadm Abilene` becomes
  `Abilene`). Transceiver nodes are ignored.
- **Links:** the links with a positive `propagation-delay` latency
  characteristic. The two unidirectional links between a pair of ROADMs
  become one undirected link. Links without a delay (such as those attaching
  transceivers to their ROADM) are ignored.
- **Length:** `total-size` of the propagation delay (ns) times the speed of
  light in fibre, c / 1.47 (`TAPI_FIBER_GROUP_INDEX`). If the two directions
  differ, their mean is used. Rounded to 3 decimals.
- **Errors:** `ValueError` if the document has no topology context or no
  topology, if two ROADMs have the same name, or if a link references an
  unknown node.

`examples/topologies/coronet_tapi_topology_context.json` is CORONET CONUS in
this format (75 ROADMs, 99 links).

---

## Built-in topologies

These are found by `resolve_topology` when no topology directory is set:

| Name | Format | Nodes | Links |
|---|---|---|---|
| `germany50` | SNDlib | 50 | 88 |
| `janos-us` | SNDlib | 26 | 42 |
| `nobel-eu` | SNDlib | 28 | 41 |
| `nobel-us` | SNDlib | 14 | 21 |
| `nsfnet_chen` | text | 14 | 22 |
| `ring_4` | text | 4 | 4 |

---

## Spans

`from_file` splits each link into spans of equal length, at most
`max_span_length_km` each: a link of length `L` gets `ceil(L / max_span_length_km)`
spans. A link shorter than `max_span_length_km` is the exception and gets 2
spans (for example a 50 km link with 100 km spans gets two 25 km spans). Each
span ends with an amplifier that compensates its loss.

---

## Paths

`from_file` computes up to `k_paths` shortest simple paths (by length) for
every unordered pair of nodes, with `networkx.shortest_simple_paths`. They are
stored once per pair, ordered from the node that comes first in `node_names`
to the other one. Both `get_paths(a, b)` and `get_paths(b, a)` return the same
tuple, in that order:

```python
topology = TopologyModel.from_file(resolve_topology("ring_4"), k_paths=2)
[p.node_names for p in topology.get_paths("3", "1")]
# [('1', '2', '3'), ('1', '4', '3')]
```

The QoT engine and the runtime state identify a path by its link ids, so the
direction does not matter for them. `path_from_link_ids` builds a `PathRecord`
for any other route, for example a sub-path or the output of an external
planner.

---

## Data classes

All of them are frozen dataclasses.

### TopologyModel

| Field | Description |
|---|---|
| `topology_id: str` | Name of the topology (the file stem unless set). |
| `node_names: tuple[str, ...]` | Node names; a node's index is its position. |
| `links: tuple[Link, ...]` | Links; `links[i].id == i`. |
| `paths: tuple[PathRecord, ...]` | All K-shortest paths; `paths[i].id == i`. |
| `path_index_by_endpoints` | `(source, target) -> tuple[PathRecord, ...]`, both orientations. |
| `node_index_by_name: dict[str, int]` | Index of each node. |
| `link_id_by_endpoints` | `(node_a, node_b) -> link id`, with `node_a` the node of lower index. |
| `link_lengths_km: np.ndarray` | Length of each link, indexed by link id. |

| Method / property | Description |
|---|---|
| `from_file(...)` | Build a model from a topology file (see [Loading a topology](#loading-a-topology)). |
| `node_count`, `link_count`, `path_count` | Sizes. |
| `get_node_index(name)` | Index of a node. |
| `get_paths(source, target)` | K-shortest paths between two nodes, by name. |
| `get_paths_by_ids(source_id, target_id)` | The same, by node index. |
| `link_by_id(link_id)` | A link by id. |
| `link_between(a, b)` | The link between two nodes, in either order. `KeyError` if none. |
| `path_from_link_ids(link_ids, *, path_id=-1)` | A `PathRecord` for a route given as links in order of travel, with `k=-1`. `ValueError` if the list is empty, an id is unknown, or two consecutive links do not share a node. |
| `with_span_updates(updates)` | A new model with the loss and/or NF of some spans changed (see [SpanUpdate](#spanupdate)). |

### Link

| Field | Description |
|---|---|
| `id: int` | Link id. |
| `source_name`, `target_name: str` | End nodes, in file order. Links are undirected. |
| `source_index`, `target_index: int` | Indices of the end nodes. |
| `length_km: float` | Length. |
| `spans: tuple[Span, ...]` | Spans from `source_name` to `target_name`. |

### Span

One fibre span followed by the amplifier that compensates its loss.

| Field | Description |
|---|---|
| `length_km: float` | Fibre length. |
| `attenuation_db_per_km: float` | Fibre loss coefficient. |
| `noise_figure_db: float` | Noise figure of the amplifier at the end of the span. |
| `input_loss_db: float = 0.0` | Lumped loss before the fibre; it lowers the launch power. |
| `output_loss_db: float = 0.0` | Lumped loss after the fibre, before the amplifier. |
| `amplifier_type: str \| None = None` | Equipment type of the amplifier (informational). |
| `gain_ripple: GainRipple \| None = None` | Wavelength-dependent gain deviation, or `None` for a flat gain. |

Derived properties: `attenuation_normalized` (field attenuation in 1/m),
`noise_figure_normalized` (linear NF), `input_loss_linear`,
`output_loss_linear`, and `total_loss_db` (fibre plus lumped losses, i.e. the
gain the amplifier must provide).

`from_file` only sets the length, attenuation and NF. The other fields come
from `apply_inventory`.

### PathRecord

| Field | Description |
|---|---|
| `id: int` | Position in `TopologyModel.paths`, or `-1` (or `path_id`) for a record from `path_from_link_ids`. |
| `k: int` | Rank among the paths of its node pair, from 0; `-1` for a record from `path_from_link_ids`. |
| `node_names`, `node_indices` | Nodes in order of travel. |
| `link_ids: tuple[int, ...]` | Links in order of travel. |
| `hops: int` | Number of links. |
| `length_km: float` | Total length. |

### SpanUpdate

New fibre loss and/or amplifier NF for one span, applied by
`TopologyModel.with_span_updates` or, during a simulation, by
`Simulator.update_spans` / `OpticalEnv.update_spans`.

| Field | Description |
|---|---|
| `link_id: int` | Link of the span. |
| `span_index: int` | Position of the span in the link, from 0 (the span after `source_name`). |
| `attenuation_db_per_km: float \| None = None` | New loss; `None` keeps the current one. |
| `noise_figure_db: float \| None = None` | New NF; `None` keeps the current one. |

`with_span_updates` applies the updates in order (a later update of the same
span wins) and shares everything else with the original model. It raises
`ValueError`, without changing anything, if a link id or span index is out of
range or a value is not finite and positive. See
[Runtime physical-layer updates](physical_layer.md#runtime-physical-layer-updates-aging)
for the simulation side.
