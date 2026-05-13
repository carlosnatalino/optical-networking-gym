# Examples

Start with the canonical scenario API:

```python
from optical_networking_gym_v2 import ScenarioConfig, iter_scenarios, make_env
from optical_networking_gym_v2.utils.sweep_reporting import Parallelism

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
