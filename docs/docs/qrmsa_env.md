# QRMSAEnv

## Overview

The `QRMSAEnv` class is a core component of the **Optical Networking Gym**, an open-source toolkit designed for benchmarking resource assignment problems in optical networks. This class represents the environment for a Reinforcement Learning (RL) agent, handling the simulation of routing and spectrum assignment (RMSA) tasks within an optical network topology.

Implemented using **Cython** for performance optimization, `QRMSAEnv` leverages typed memory views and C-level declarations to ensure efficient computation, which is crucial for large-scale network simulations.

## Table of Contents

1. [Class Definition](#class-definition)
2. [Attributes](#attributes)
3. [Initialization](#initialization)
4. [Methods](#methods)
    - [reset](#reset)
    - [observation](#observation)
    - [step](#step)
    - [_next_service](#next_service)
    - [set_load](#set_load)
    - [_get_node_pair](#get_node_pair)
    - [_get_network_compactness](#get_network_compactness)
    - [get_number_slots](#get_number_slots)
    - [is_path_free](#is_path_free)
    - [reward](#reward)
    - [_provision_path](#provision_path)
    - [_add_release](#add_release)
    - [_release_path](#release_path)
    - [_update_link_stats](#update_link_stats)
    - [get_available_slots](#get_available_slots)
    - [close](#close)
5. [Usage Example](#usage-example)
6. [Dependencies](#dependencies)
7. [License and Credits](#license-and-credits)

---

## Class Definition

```cython
cdef class QRMSAEnv:
    ...
```

The `QRMSAEnv` class inherits from a base environment class (implicitly, since it's not explicitly shown) and encapsulates the state and behavior required to simulate an optical network for RL-based RMSA tasks.

---

## Attributes

The class contains numerous attributes, each serving a specific purpose in the simulation:

- **Public Attributes:**
    - `input_seed (uint32_t)`: Seed for random number generation.
    - `load (float)`: Network load parameter.
    - `k_shortest_paths (object)`: Precomputed k-shortest paths for routing.
    - `launch_power_dbm (float)`: Launch power in dBm.
    - `launch_power (float)`: Launch power in linear scale.
    - `frequency_start (float)`: Starting frequency of the spectrum.
    - `frequency_end (float)`: Ending frequency of the spectrum.
    - `frequency_slot_bandwidth (float)`: Bandwidth of each frequency slot.
    - `margin (float)`: Margin parameter for OSNR calculations.
    - `modulations (object)`: Available modulation formats.
    - `_np_random (object)`: Numpy random number generator.
    - `_np_random_seed (int)`: Seed for the Numpy RNG.
    - `current_service (Service)`: The currently processed service.
    - `action_space (object)`: Action space for the RL agent.
    - `observation_space (object)`: Observation space for the RL agent.
    - `frequency_vector (object)`: Vector of frequencies in the spectrum.

- **Read-Only Attributes:**
    - `topology (nx.Graph)`: Network topology graph.
    - `bit_rate_selection (str)`: Strategy for bit rate selection ("continuous" or "discrete").
    - `bit_rates (tuple)`: Available bit rates.

- **Private/Internal Attributes:**
    - `episode_length (int)`: Length of an episode in the simulation.
    - `mean_service_holding_time (float)`: Average holding time of a service.
    - `num_spectrum_resources (int)`: Number of spectrum resources available.
    - `channel_width (float)`: Width of each channel.
    - `allow_rejection (bool)`: Flag to allow service rejection.
    - `bit_rate_lower_bound (float)`: Lower bound for continuous bit rate selection.
    - `bit_rate_higher_bound (float)`: Upper bound for continuous bit rate selection.
    - `bit_rate_probabilities (object)`: Probabilities for discrete bit rate selection.
    - `node_request_probabilities (object)`: Probabilities for node requests.
    - `k_paths (int)`: Number of paths considered in routing.
    - `bandwidth (float)`: Total bandwidth of the spectrum.
    - `measure_disruptions (bool)`: Flag to measure service disruptions.
    - `spectrum_use (object)`: Spectrum usage matrix.
    - `spectrum_allocation (object)`: Spectrum allocation matrix.
    - `service_id_counter (int)`: Counter for assigning unique service IDs.
    - `services_in_progress (list)`: List of services currently in progress.
    - `release_times (list)`: List of scheduled release times for services.
    - `services_processed (int)`: Total number of services processed.
    - `services_accepted (int)`: Total number of services accepted.
    - `episode_services_processed (int)`: Services processed in the current episode.
    - `episode_services_accepted (int)`: Services accepted in the current episode.
    - `bit_rate_requested (float)`: Total bit rate requested.
    - `bit_rate_provisioned (float)`: Total bit rate provisioned.
    - `episode_bit_rate_requested (float)`: Bit rate requested in the current episode.
    - `episode_bit_rate_provisioned (float)`: Bit rate provisioned in the current episode.
    - `bit_rate_requested_histogram (object)`: Histogram of requested bit rates.
    - `bit_rate_provisioned_histogram (object)`: Histogram of provisioned bit rates.
    - `slots_provisioned_histogram (object)`: Histogram of provisioned slots.
    - `episode_slots_provisioned_histogram (object)`: Slot provisioning histogram for the current episode.
    - `disrupted_services (int)`: Total number of disrupted services.
    - `episode_disrupted_services (int)`: Number of disrupted services in the current episode.
    - `disrupted_services_list (list)`: List of disrupted services.
    - `episode_actions_output (object)`: Actions output during the current episode.
    - `episode_actions_taken (object)`: Actions taken during the current episode.
    - `episode_modulation_histogram (object)`: Modulation histogram for the current episode.
    - `episode_bit_rate_requested_histogram (object)`: Bit rate requested histogram for the current episode.
    - `episode_bit_rate_provisioned_histogram (object)`: Bit rate provisioned histogram for the current episode.
    - `spectrum_slots_allocation (object)`: Spectrum slots allocation matrix.
    - `reject_action (int)`: Indicator for reject actions.
    - `actions_output (object)`: Overall actions output.
    - `actions_taken (object)`: Overall actions taken.
    - `_new_service (bool)`: Flag indicating a new service.
    - `current_time (float)`: Current simulation time.
    - `mean_service_inter_arrival_time (float)`: Mean inter-arrival time for services.
    - `rng (object)`: Random number generator instance.
    - `bit_rate_function (object)`: Function to generate bit rates.
    - `_events (list)`: Priority queue for scheduled events.
    - `file_stats (object)`: File handler for logging statistics.
    - `final_file_name (str)`: Name of the statistics file.

---

## Initialization

The `__init__` method initializes the environment with the specified parameters and sets up the necessary data structures for simulation.

### Parameters

| Parameter                      | Type                     | Default          | Description                                                                                      |
| ------------------------------ | ------------------------ | ---------------- | ------------------------------------------------------------------------------------------------ |
| `topology`                     | `nx.Graph`               | **Required**     | Network topology graph.                                                                         |
| `num_spectrum_resources`       | `int`                    | `320`            | Number of spectrum resources available.                                                         |
| `episode_length`               | `int`                    | `1000`           | Number of services per episode.                                                                  |
| `load`                         | `float`                  | `10.0`           | Network load parameter.                                                                         |
| `mean_service_holding_time`    | `float`                  | `10800.0`        | Average holding time of a service (in seconds).                                                 |
| `bit_rate_selection`           | `str`                    | `"continuous"`   | Strategy for bit rate selection (`"continuous"` or `"discrete"`).                               |
| `bit_rates`                    | `tuple`                  | `(10, 40, 100)`   | Available bit rates for discrete selection.                                                      |
| `bit_rate_probabilities`       | `object`                 | `None`           | Probabilities for each bit rate in discrete selection. Defaults to uniform distribution.        |
| `node_request_probabilities`   | `object`                 | `None`           | Probabilities for node requests. Defaults to uniform distribution if `None`.                    |
| `bit_rate_lower_bound`         | `float`                  | `25.0`           | Lower bound for continuous bit rate selection.                                                   |
| `bit_rate_higher_bound`        | `float`                  | `100.0`          | Upper bound for continuous bit rate selection.                                                   |
| `launch_power_dbm`             | `float`                  | `0.0`            | Launch power in dBm.                                                                              |
| `bandwidth`                    | `float`                  | `4e12`           | Total bandwidth of the spectrum.                                                                  |
| `frequency_start`              | `float`                  | `(3e8 / 1565e-9)`| Starting frequency of the spectrum.                                                               |
| `frequency_slot_bandwidth`     | `float`                  | `12.5e9`         | Bandwidth of each frequency slot.                                                                 |
| `margin`                       | `float`                  | `0.0`            | Margin parameter for OSNR calculations.                                                          |
| `measure_disruptions`          | `bool`                   | `False`          | Flag to measure service disruptions.                                                              |
| `seed`                         | `object`                 | `None`           | Seed for random number generation. Must be an integer if provided.                               |
| `allow_rejection`              | `bool`                   | `False`          | Flag to allow service rejection.                                                                  |
| `reset`                        | `bool`                   | `True`           | Flag to reset the environment upon initialization.                                               |
| `channel_width`                | `float`                  | `12.5`           | Width of each channel.                                                                           |
| `k_paths`                      | `int`                    | `5`              | Number of paths considered in routing.                                                           |
| `file_name`                    | `str`                    | `""`             | Name of the file to log statistics. If empty, logging is disabled.                              |

### Example Initialization

```python
import networkx as nx
from optical_networking_gym import QRMSAEnv

# Create a network topology
topology = nx.Graph()
# Add nodes and edges to the topology
# ...

# Initialize the QRMSAEnv
env = QRMSAEnv(
    topology=topology,
    num_spectrum_resources=320,
    episode_length=1000,
    load=10.0,
    mean_service_holding_time=10800.0,
    bit_rate_selection="discrete",
    bit_rates=(10, 40, 100),
    allow_rejection=True,
    seed=42,
    file_name="simulation_stats.csv"
)
```

---

## Methods

### `reset`

```cython
cpdef tuple reset(self, object seed=None, dict options=None)
```

**Description:**

Resets the environment to its initial state, preparing it for a new simulation episode. Optionally, allows setting a new seed and configuring specific reset options.

**Parameters:**

- `seed (object, optional)`: New seed for random number generation. Must be an integer if provided.
- `options (dict, optional)`: Additional options for resetting. For example, setting `only_episode_counters` to `True` will reset only the episode-specific counters.

**Returns:**

- `tuple`: A tuple containing the initial observation and an empty dictionary for additional info.

**Usage Example:**

```python
initial_observation, info = env.reset()
```

---

### `observation`

```cython
def observation(self)
```

**Description:**

Generates the current observation of the environment, which includes the available spectrum slots and the running services.

**Returns:**

- `dict`: A dictionary containing:
    - `"topology"`: Numpy array representing available slots.
    - `"running-services"`: Padded array of service IDs.

**Usage Example:**

```python
current_observation = env.observation()
```

---

### `step`

```cython
cpdef tuple[object, float, bint, bint, dict] step(self, cnp.ndarray action)
```

**Description:**

Executes a single step in the environment based on the provided action. The action typically involves selecting a route, modulation format, and initial slot for a service.

**Parameters:**

- `action (cnp.ndarray)`: Action array containing route index, modulation index, and initial slot.

**Returns:**

- `tuple`: A tuple containing:
    - `observation (dict)`: New observation after the action.
    - `reward (float)`: Reward obtained from the action.
    - `terminated (bool)`: Flag indicating if the episode has ended.
    - `truncated (bool)`: Flag indicating if the episode was truncated.
    - `info (dict)`: Additional information about the step.

**Usage Example:**

```python
action = np.array([route_idx, modulation_idx, initial_slot])
observation, reward, terminated, truncated, info = env.step(action)
```

---

### `_next_service`

```cython
cpdef _next_service(self)
```

**Description:**

Advances the simulation to the next service by scheduling its arrival based on inter-arrival times and initializing its parameters.

**Usage Example:**

```python
env._next_service()
```

---

### `set_load`

```cython
cpdef void set_load(self, float load=-1.0, float mean_service_holding_time=-1.0)
```

**Description:**

Sets or updates the network load and mean service holding time parameters.

**Parameters:**

- `load (float, optional)`: New load value. Must be positive.
- `mean_service_holding_time (float, optional)`: New mean holding time. Must be positive.

**Usage Example:**

```python
env.set_load(load=15.0, mean_service_holding_time=12000.0)
```

---

### `_get_node_pair`

```cython
cdef tuple _get_node_pair(self)
```

**Description:**

Generates a source and destination node pair based on node request probabilities.

**Returns:**

- `tuple`: A tuple containing source node, source node ID, destination node, and destination node ID.

**Usage Example:**

```python
src, src_id, dst, dst_id = env._get_node_pair()
```

---

### `_get_network_compactness`

```cython
cpdef double _get_network_compactness(self)
```

**Description:**

Calculates the network's spectrum compactness based on service allocations and slot usage.

**Returns:**

- `double`: The compactness value of the network.

**Usage Example:**

```python
compactness = env._get_network_compactness()
```

---

### `get_number_slots`

```cython
cpdef int get_number_slots(self, object service, object modulation)
```

**Description:**

Calculates the number of spectrum slots required to accommodate a service request, including guard bands.

**Parameters:**

- `service (object)`: The service request object.
- `modulation (object)`: The modulation format object.

**Returns:**

- `int`: Number of spectrum slots required.

**Usage Example:**

```python
num_slots = env.get_number_slots(service, modulation)
```

---

### `is_path_free`

```cython
cpdef bint is_path_free(self, object path, int initial_slot, int number_slots)
```

**Description:**

Checks if a specified path is free (i.e., has available spectrum slots) for the given slot range.

**Parameters:**

- `path (object)`: The path object representing the route.
- `initial_slot (int)`: Starting slot index.
- `number_slots (int)`: Number of slots required.

**Returns:**

- `bint`: `True` if the path is free, `False` otherwise.

**Usage Example:**

```python
free = env.is_path_free(path, initial_slot, number_slots)
```

---

### `reward`

```cython
cpdef double reward(self)
```

**Description:**

Calculates the reward based on the OSNR (Optical Signal-to-Noise Ratio) achieved by the service.

**Returns:**

- `double`: Reward value.

**Usage Example:**

```python
reward = env.reward()
```

---

### `_provision_path`

```cython
cpdef _provision_path(self, object path, cnp.int64_t initial_slot, int number_slots)
```

**Description:**

Allocates the specified path and spectrum slots to the current service, updating the environment's state accordingly.

**Parameters:**

- `path (object)`: The path to provision.
- `initial_slot (int)`: Starting slot index.
- `number_slots (int)`: Number of slots to allocate.

**Usage Example:**

```python
env._provision_path(path, initial_slot, number_slots)
```

---

### `_add_release`

```cython
cpdef void _add_release(self, Service service)
```

**Description:**

Schedules the release of a service after its holding time by adding it to the event queue.

**Parameters:**

- `service (Service)`: The service to be released.

**Usage Example:**

```python
env._add_release(service)
```

---

### `_release_path`

```cython
cpdef _release_path(self, Service service)
```

**Description:**

Releases the spectrum slots allocated to a service, updating the environment's state and freeing up resources.

**Parameters:**

- `service (Service)`: The service to release.

**Usage Example:**

```python
env._release_path(service)
```

---

### `_update_link_stats`

```cython
cpdef _update_link_stats(self, str node1, str node2)
```

**Description:**

Updates statistical metrics for a specific link between `node1` and `node2`, such as utilization and fragmentation.

**Parameters:**

- `node1 (str)`: Identifier of the first node.
- `node2 (str)`: Identifier of the second node.

**Usage Example:**

```python
env._update_link_stats('A', 'B')
```

---

### `get_available_slots`

```cython
cpdef cnp.ndarray get_available_slots(self, object path)
```

**Description:**

Computes the available spectrum slots along a specified path by performing an element-wise multiplication of available slots across all links in the path.

**Parameters:**

- `path (object)`: The path for which to compute available slots.

**Returns:**

- `cnp.ndarray`: Numpy array indicating available slots along the path.

**Usage Example:**

```python
available_slots = env.get_available_slots(path)
```

---

### `close`

```cython
def close(self)
```

**Description:**

Closes the environment, performing any necessary cleanup operations.

**Returns:**

- `object`: Calls the superclass's `close` method.

**Usage Example:**

```python
env.close()
```

---

## Usage Example

Below is an example of how to utilize the `QRMSAEnv` class within a simulation:

```python
import networkx as nx
import numpy as np
from optical_networking_gym import QRMSAEnv

# Create a network topology
topology = nx.Graph()
# Add nodes
topology.add_nodes_from(['A', 'B', 'C', 'D'])
# Add edges with necessary attributes
topology.add_edge('A', 'B', index=0, modulations=[Modulation(spectral_efficiency=4.0, minimum_osnr=25.0)])
topology.add_edge('B', 'C', index=1, modulations=[Modulation(spectral_efficiency=4.0, minimum_osnr=25.0)])
topology.add_edge('C', 'D', index=2, modulations=[Modulation(spectral_efficiency=4.0, minimum_osnr=25.0)])
topology.graph['ksp'] = precompute_k_shortest_paths(topology, k=5)
topology.graph['node_indices'] = list(topology.nodes())

# Initialize the environment
env = QRMSAEnv(
    topology=topology,
    num_spectrum_resources=320,
    episode_length=1000,
    load=10.0,
    mean_service_holding_time=10800.0,
    bit_rate_selection="discrete",
    bit_rates=(10, 40, 100),
    allow_rejection=True,
    seed=42,
    file_name="simulation_stats.csv"
)

# Reset the environment
observation, info = env.reset()

# Example interaction loop
for _ in range(10):
    # Sample a random action
    action = env.action_space.sample()
    # Take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}, Terminated: {terminated}")

    if terminated or truncated:
        break

# Close the environment
env.close()
```

---

## Dependencies

The `QRMSAEnv` class relies on several external libraries and modules:

- **Cython**: For performance optimization.
- **NetworkX** (`nx.Graph`): To represent and manage the network topology.
- **NumPy** (`numpy`): For numerical operations and array manipulations.
- **Gym** (`gym.spaces`): For defining action and observation spaces in the RL environment.
- **Heapq**: For managing the priority queue of events.
- **Random**: For random number generation.
- **Math**: For mathematical operations.

Ensure all dependencies are installed and properly configured in your development environment.

---

## License and Credits

**Optical Networking Gym** is licensed under the [MIT License](https://github.com/carlosnatalino/optical-networking-gym/blob/main/LICENSE). We extend our gratitude to all contributors and the open-source community for their support and contributions.

---

