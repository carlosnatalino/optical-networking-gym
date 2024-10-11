# Gymnasium Integration

## Overview

The **Wrappers** module provides a Gymnasium-compatible environment wrapper for the `QRMSAEnv` class, facilitating seamless integration with reinforcement learning (RL) algorithms. Additionally, it includes utility functions for running simulations with various heuristic strategies.

## Table of Contents

1. [QRMSAEnvWrapper Class](#qrmsaenvwrapper-class)
2. [Utility Functions](#utility-functions)
    - [run_wrapper](#run_wrapper)
    - [run_environment](#run_environment)
    - [Heuristic Strategies](#heuristic-strategies)
3. [Usage Example](#usage-example)
4. [Dependencies](#dependencies)
5. [License and Credits](#license-and-credits)

---

## QRMSAEnvWrapper Class

```python
class QRMSAEnvWrapper(gym.Env):
    ...
```

### Description

`QRMSAEnvWrapper` is a Gymnasium-compatible wrapper for the `QRMSAEnv` environment, enabling it to be used with RL algorithms seamlessly.

### Attributes

- **env** (`QRMSAEnv`): The underlying environment instance.
- **action_space** (`gym.Space`): Action space inherited from `QRMSAEnv`.
- **observation_space** (`gym.Space`): Observation space inherited from `QRMSAEnv`.
- **_np_random** (`np.random.Generator`): Random number generator for seeding.

### Methods

- **`__init__(self, *args, **kwargs)`**: Initializes the wrapper and the underlying `QRMSAEnv` instance.
  
- **`reset(self, *, seed=None, options=None)`**: Resets the environment to an initial state.
  
- **`step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]`**: Takes an action and returns the outcome.
  
- **`render(self, mode='human')`**: Renders the environment (if supported).
  
- **`close(self)`**: Closes the environment and performs cleanup.
  
- **`get_spectrum_use_services(self)`**: Retrieves spectrum usage statistics from the environment.

---

## Utility Functions

### run_wrapper

```python
def run_wrapper(args):
    ...
```

#### Description

Entrypoint function to execute the environment wrapper with specified arguments.

### run_environment

```python
def run_environment(
    n_eval_episodes,
    heuristic,
    monitor_file_name,
    topology,
    seed,
    allow_rejection,
    load,
    episode_length,
    num_spectrum_resources,
    launch_power_dbm,
    bandwidth,
    frequency_start,
    frequency_slot_bandwidth,
    bit_rate_selection,
    bit_rates,
    margin,
    file_name,
    measure_disruptions,
) -> None:
    ...
```

#### Description

Runs the simulation environment for a given number of evaluation episodes using the specified heuristic strategy. Logs performance metrics to a CSV file.

### Heuristic Strategies

These functions define different strategies for selecting actions within the environment.

#### shortest_available_path_first_fit_best_modulation

```python
def shortest_available_path_first_fit_best_modulation(env: QRMSAEnv):
    ...
```

- **Description:** Selects the shortest available path with the first-fit spectrum allocation and the best modulation format that meets OSNR requirements.

#### shortest_available_path_lowest_spectrum_best_modulation

```python
def shortest_available_path_lowest_spectrum_best_modulation(env: QRMSAEnv) -> Optional[Tuple[int, int, int]]:
    ...
```

- **Description:** Chooses the shortest available path with the lowest spectrum usage and the best modulation format.

#### best_modulation_load_balancing

```python
def best_modulation_load_balancing(env: QRMSAEnv) -> Optional[Tuple[int, int, int]]:
    ...
```

- **Description:** Balances network load by selecting the best modulation format while minimizing the load on the selected path.

#### load_balancing_best_modulation

```python
def load_balancing_best_modulation(env: QRMSAEnv) -> Optional[Tuple[int, int, int]]:
    ...
```

- **Description:** Prioritizes load balancing by choosing the best modulation format on the path with the lowest current load.

---

## Usage Example

```python
import networkx as nx
from optical_networking_gym.topology import Modulation
from optical_networking_gym.wrappers import QRMSAEnvWrapper, run_environment

# Define modulation formats
modulations = (
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
)

# Create network topology
topology = get_topology(
    file_path="topology.xml",
    topology_name="TestTopology",
    modulations=modulations,
    k_paths=5
)

# Run environment with heuristic 1
run_environment(
    n_eval_episodes=10,
    heuristic=1,
    monitor_file_name="monitor",
    topology=topology,
    seed=42,
    allow_rejection=True,
    load=10.0,
    episode_length=1000,
    num_spectrum_resources=320,
    launch_power_dbm=0.0,
    bandwidth=4e12,
    frequency_start=3e8 / 1565e-9,
    frequency_slot_bandwidth=12.5e9,
    bit_rate_selection="discrete",
    bit_rates=(10, 40, 100),
    margin=0.0,
    file_name="simulation_stats.csv",
    measure_disruptions=False,
)
```

---

## Dependencies

- **Gymnasium**: For defining and interacting with the RL environment.
- **NetworkX**: For network topology management.
- **NumPy**: Numerical operations and array handling.
- **Optical Networking Gym Modules**:
  - `qrmsa`: Core environment classes.
  - `topology`: Network topology classes and utilities.
  - `utils`: Utility functions like run-length encoding (`rle`).
  - `core.osnr`: Functions for OSNR calculations.

### Installation

Ensure all dependencies are installed, preferably within a virtual environment:

```bash
pip install gymnasium networkx numpy cython
```

---

## License and Credits

**Optical Networking Gym** is licensed under the [MIT License](https://github.com/carlosnatalino/optical-networking-gym/blob/main/LICENSE). Special thanks to all contributors and the open-source community for their support and contributions.

---