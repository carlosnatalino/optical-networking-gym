import argparse
import logging
import os
import random
from typing import List, Tuple
from multiprocessing import Pool 

import numpy as np

from optical_networking_gym.wrappers.qrmsa_gym import run_wrapper
from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.envs.qrmsa import Service  # Import Service class

# Configure logging
logging.getLogger("rmsaenv").setLevel(logging.INFO)
np.set_printoptions(linewidth=np.inf)

# Initial configurations
seed = 20
random.seed(seed)

# # Monkey-patch the Service class to support ordering
# def service_lt(self, other):
#     return self.time < other.time  # Replace 'time' with the appropriate attribute

# Service.__lt__ = service_lt  # Add the __lt__ method to the Service class

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Optical Network Simulation')

    parser.add_argument(
        '-t', '--topology_file',
        type=str,
        default='nsfnet_chen.txt',
        help='Network topology file to be used (default: nsfnet_chen.txt)'
    )

    parser.add_argument(
        '-e', '--num_episodes',
        type=int,
        default=1,
        help='Number of episodes to be simulated (default: 1)'
    )

    parser.add_argument(
        '-s', '--episode_length',
        type=int,
        default=1000,
        help='Number of arrivals per episode to be generated (default: 1000)'
    )

    parser.add_argument(
        '-th', '--threads',
        type=int,
        default=2,
        help='Number of threads to be used to run the simulations (default: 2)'
    )

    return parser.parse_args()

def define_modulations() -> Tuple[Modulation, ...]:
    return (
        Modulation(
            name="BPSK",
            maximum_length=100_000,  # 100,000 km to ensure safety
            spectral_efficiency=1,
            minimum_osnr=12.6,
            inband_xt=-14,
        ),
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
        Modulation(
            name="16QAM",
            maximum_length=500,
            spectral_efficiency=4,
            minimum_osnr=22.4,
            inband_xt=-23,
        ),
        Modulation(
            name="32QAM",
            maximum_length=250,
            spectral_efficiency=5,
            minimum_osnr=26.4,
            inband_xt=-26,
        ),
        Modulation(
            name="64QAM",
            maximum_length=125,
            spectral_efficiency=6,
            minimum_osnr=30.4,
            inband_xt=-29,
        ),
    )

def get_loads(topology_name: str) -> np.ndarray:
    if topology_name == "nobel-eu.xml":
        return np.arange(50, 651, 50)
    elif topology_name == "germany50.xml":
        return np.arange(300, 801, 50)
    elif topology_name == "janos-us.xml":
        return np.arange(100, 601, 50)
    elif topology_name == "nsfnet_chen.txt":
        return np.arange(100, 601, 50)
    else:
        raise ValueError(f"Unknown topology name: {topology_name}")

def prepare_env_args(
    n_eval_episodes: int,
    topology: object,
    episode_length: int,
    launch_power: float,
    bandwidth: float,
    frequency_start: float,
    frequency_slot_bandwidth: float,
    bit_rates: tuple[int, ...],
    margin: int,
    loads: np.ndarray,
    strategies: List[int]
) -> List[Tuple]:
    env_args = []
    for current_load in loads:
        for strategy in strategies:
            sim_args = (
                n_eval_episodes,
                strategy,
                f"examples/JOCN_Benchmark_2024/results/load_episodes_{strategy}",
                topology,
                10,
                True,
                current_load,
                episode_length,
                320,
                launch_power,
                bandwidth,
                frequency_start,
                frequency_slot_bandwidth,
                "discrete",
                bit_rates,
                margin,
                f"examples/JOCN_Benchmark_2024/results/load_services_{strategy}",
                False,
            )
            env_args.append(sim_args)
    return env_args

def main() -> None:
    args = parse_arguments()

    # Assign arguments to variables
    topology_name = args.topology_file
    n_eval_episodes = args.num_episodes
    episode_length = args.episode_length
    threads = args.threads

    launch_power = -4.0

    # Define load ranges based on topology
    loads = get_loads(topology_name)

    # Define modulation formats
    cur_modulations = define_modulations()

    # Load topology
    topology = get_topology(
        os.path.join("examples", "topologies", topology_name),
        None,                # Name of the topology, adjust if necessary
        cur_modulations,         # Tuple of modulation formats
        80,                      # Maximum span length in km
        0.2,                     # Default attenuation in dB/km
        4.5,                     # Default noise figure in dB
        5                        # Number of shortest paths to compute between node pairs
    )

    bandwidth = 4e12
    frequency_start = 3e8 / 1565e-9
    frequency_end = frequency_start + bandwidth
    frequency_slot_bandwidth = 12.5e9
    bit_rates = (10, 40, 100, 400)
    margin = 0

    # Define strategies
    strategies = list(range(1, 5))

    # Prepare environment arguments
    env_args = prepare_env_args(
        n_eval_episodes,
        topology,
        episode_length,
        launch_power,
        bandwidth,
        frequency_start,
        frequency_slot_bandwidth,
        bit_rates,
        margin,
        loads,
        strategies
    )

    # Execute simulations with or without multiprocessing based on thread count
    if threads > 1:
        with Pool(processes=threads) as pool:
            results = pool.map(run_wrapper, env_args)
    else:
        results = [run_wrapper(arg) for arg in env_args]

    # Print results
    print(results)

if __name__ == "__main__":
    main()
