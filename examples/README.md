# Examples

Start with the canonical scenario API:

```python
from optical_networking_gym import ScenarioConfig, iter_scenarios, make_env
from optical_networking_gym.utils.sweep_reporting import Parallelism

env = make_env(scenario="ring4_quickstart")
env = make_env(scenario="nobel_eu_baseline", load=400, margin=2.0)
env = make_env(scenario="nobel_eu_baseline", modulations="BPSK,QPSK,16QAM")

custom = ScenarioConfig(
    scenario_id="custom_ring",
    topology_id="ring_4",
    k_paths=2,
    num_spectrum_resources=24,
)
env = make_env(config=custom)
```

Build sweeps from presets:

```python
scenarios = tuple(
    iter_scenarios(
        "nobel_eu_baseline",
        axes={
            "load": (300, 400),
            "margin": (0.0, 1.0),
            "topology_id": ("ring_4", "nobel-eu"),
            "qot_constraint": ("DIST", "ASE+NLI"),
        },
    )
)
```

Standard experiment runs write:

```text
examples/results/<family>/<script-stem>/<YYYYMMDD-HHMMSS>/
  metadata.json
  episodes.csv
  summary.csv
```

Use one parallelism vocabulary:

```python
non_rl_sweep = Parallelism(workers=8, envs_per_worker=1)
single_rl_training = Parallelism(workers=1, envs_per_worker=8)
rl_sweep = Parallelism(workers=4, envs_per_worker=8)
```

`quickstart/` contains first-path examples. `SBRT2026/` contains publication
sweeps and trace scripts. Advanced examples may use `ScenarioConfig` directly
when the example is about custom configuration.

## Topologies

`TopologyModel.from_file` (and therefore `make_env` and `create_topology.py`)
reads three formats, chosen by the file extension:

| Extension | Format |
|---|---|
| `.xml` | SNDlib; link lengths from the node coordinates |
| `.txt` | node count followed by `source target length_km` lines |
| `.json` | T-API topology context (`tapi-topology:topology-context`) |

A T-API context is mapped onto the same undirected graph as the other formats:

- **Nodes.** One node per ROADM: the endpoints of the fibre links, named
  without their `roadm ` prefix (`roadm Abilene` becomes `Abilene`).
  Transceivers and the zero-delay links that attach them to their ROADM are
  ignored.
- **Links.** The two unidirectional links between a pair of ROADMs become one
  undirected link. Its length comes from the `propagation-delay`
  latency characteristic (`total-size`, in nanoseconds), divided by the speed
  of light in fibre (group index 1.47); the mean is used if the two
  directions differ. Links without a propagation delay are ignored.
- **Physical parameters.** Everything the context does not describe (span
  length, fibre attenuation, amplifier noise figure) takes the standard
  defaults of `from_file` or of the scenario (`max_span_length_km`,
  `default_attenuation_db_per_km`, `default_noise_figure_db`).
- Only the first topology of the context is read.

`topologies/coronet_tapi_topology_context.json` is the CORONET CONUS topology
(75 ROADMs, 99 links) exported by
[TwinLight](https://github.com/carlosnatalino/TwinLight). Use it with:

```bash
python examples/create_topology.py \
    --topology examples/topologies/coronet_tapi_topology_context.json -k 5
```

```python
env = make_env(
    scenario="nobel_eu_baseline",
    overrides={
        "topology_id": "coronet_tapi_topology_context",
        "topology_dir": "examples/topologies",
    },
)
```
