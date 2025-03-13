import argparse
import logging
import os
import random
from typing import Tuple

import numpy as np

from optical_networking_gym.wrappers.qrmsa_gym import run_wrapper
from optical_networking_gym.topology import Modulation, get_topology

# Configure logging
logging.getLogger("rmsaenv").setLevel(logging.INFO)
np.set_printoptions(linewidth=np.inf)

# Initial configurations
seed = 20
random.seed(seed)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Optical Network Simulation')

    parser.add_argument(
        '-t', '--topology_file',
        type=str,
        default='nsfnet_chen.txt',
        help='Network topology file to be used (default: nobel-eu)'
    )

    parser.add_argument(
        '-e', '--num_episodes',
        type=int,
        default=1000,
        help='Number of episodes to be simulated (default: 1000)'
    )

    parser.add_argument(
        '-s', '--episode_length',
        type=int,
        default=50,
        help='Number of arrivals per episode to be generated (default: 50)'
    )

    parser.add_argument(
        '-l', '--load',
        type=int,
        default=210,
        help='Load to be used in the simulation (default: 210)'
    )

    parser.add_argument(
        '-th', '--threads',
        type=int,
        default=2,
        help='Number of threads to be used for running simulations (default: 2)'
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Define modulation formats
    cur_modulations: Tuple[Modulation] = (
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

    # Default parameters
    attenuation_db_km = 0.2
    default_noise_figure_db = 4.5

    # Load topology
    topology_path = os.path.join(
        "examples", "topologies", args.topology_file
    )
    topology = get_topology(
        topology_path,           # Path to the topology file
        None,                # Name of the topology
        cur_modulations,         # Tuple of modulation formats
        80,                      # Maximum span length in km
        attenuation_db_km,       # Default attenuation in dB/km
        default_noise_figure_db, # Default noise figure in dB
        5                        # Number of shortest paths to compute between node pairs
    )

    # Simulation parameters
    threads = args.threads
    bandwidth = 4e12
    frequency_start = 3e8 / 1565e-9
    frequency_end = frequency_start + bandwidth
    frequency_slot_bandwidth = 12.5e9
    bit_rates = (10, 40, 100, 400)

    # Define launch powers
    launch_powers = np.linspace(-8, 8, 9)
    env_args = []

    # Prepare simulation arguments for each launch power
    for launch_power in launch_powers:
        sim_args = (
            args.num_episodes,
            1,
            "examples/jocn_benchmark_2024/results/lp_episodes_1",
            topology,
            10,
            True,
            args.load,
            args.episode_length,
            320,
            launch_power,
            bandwidth,
            frequency_start,
            frequency_slot_bandwidth,
            "discrete",
            bit_rates,
            0,  # margin
            "examples/jocn_benchmark_2024/results/lp_services_1",
            True,
        )
        env_args.append(sim_args)

    # Execute simulations with or without multiprocessing based on thread count
    if threads > 1:
        from multiprocessing import Pool  # Import here to avoid unnecessary import if not used
        with Pool(processes=threads) as pool:
            results = pool.map(run_wrapper, env_args)
    else:
        results = [run_wrapper(arg) for arg in env_args]

    # Print results
    print(results)

if __name__ == "__main__":
    main()
