# RL Parallelism

Use the same `Parallelism` object for RL and non-RL experiments:

```python
from optical_networking_gym_v2.utils.sweep_reporting import Parallelism

non_rl_sweep = Parallelism(workers=8, envs_per_worker=1)
single_rl_training = Parallelism(workers=1, envs_per_worker=8)
rl_sweep = Parallelism(workers=4, envs_per_worker=8)
```

`workers` runs independent cases or training jobs. `envs_per_worker` is reserved for multiple environments inside one RL training job.
