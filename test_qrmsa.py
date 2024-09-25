from typing import Tuple

import networkx as nx
from gymnasium.utils.env_checker import check_env

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper  # Importe o wrapper

cur_modulations: Tuple[Modulation] = (
    Modulation(
        name="BPSK",
        maximum_length=100_000,  # 100,000 km to be on the safe side
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

def test_init() -> None:
    # Ajuste o caminho do arquivo de topologia conforme necessário
    topology = get_topology(
        "examples/topologies/nsfnet_chen.txt",  # Path to the topology file
        "NSFNET",  # Name of the topology
        cur_modulations,  # Tuple of modulation formats
        80,  # Maximum span length in km
        0.2,  # Default attenuation in dB/km
        4.5,  # Default noise figure in dB
        5  # Number of shortest paths to compute between node pairs
    )
    
    # Inicialize o ambiente utilizando o wrapper
    env = QRMSAEnvWrapper(
        topology=topology,
        num_spectrum_resources=320,
        episode_length=1000,
        load=10.0,
        mean_service_holding_time=10800.0,
        bit_rate_selection="continuous",
        bit_rates=(10, 40, 100),
        bit_rate_probabilities=None,
        node_request_probabilities=None,
        bit_rate_lower_bound=25.0,
        bit_rate_higher_bound=100.0,
        seed=None,
        allow_rejection=False,
        reset=True,
        channel_width=12.5,
        k_paths=5,
    )
    print("Environment initialized successfully")

def test_check_env() -> None:
    topology = get_topology(
        "examples/topologies/nsfnet_chen.txt",  # Path to the topology file
        "NSFNET",
        cur_modulations,
        80,  # Number of wavelengths per link
        0.2,  # Channel bandwidth in THz
        4.5,  # SNR gap in dB
        5,  # Number of paths per node pair
    )
    
    env = QRMSAEnvWrapper(
        topology=topology,
        num_spectrum_resources=320,
        episode_length=1000,
        load=10.0,
        mean_service_holding_time=10800.0,
        bit_rate_selection="continuous",
        bit_rates=(10, 40, 100),
        bit_rate_probabilities=None,
        node_request_probabilities=None,
        bit_rate_lower_bound=25.0,
        bit_rate_higher_bound=100.0,
        seed=None,
        allow_rejection=False,
        reset=True,
        channel_width=12.5,
        k_paths=5,
    )
    a = env.reset()
    check_env(env)
    print("Environment checked successfully")

def check_step():
    topology = get_topology(
        "examples/topologies/nsfnet_chen.txt",  # Path to the topology file
        "NSFNET",
        cur_modulations,
        80,  # Number of wavelengths per link
        0.2,  # Channel bandwidth in THz
        4.5,  # SNR gap in dB
        5,  # Number of paths per node pair
    )

    # Initialize the environment
    env = QRMSAEnvWrapper(
        topology=topology,
        num_spectrum_resources=320,
        episode_length=1000,
        load=10.0,
        mean_service_holding_time=10800.0,
        bit_rate_selection="continuous",
        bit_rates=(10, 40, 100),
        bit_rate_probabilities=None,
        node_request_probabilities=None,
        bit_rate_lower_bound=25.0,
        bit_rate_higher_bound=100.0,
        seed=None,
        allow_rejection=False,
        reset=True,
        channel_width=12.5,
        k_paths=5,
    )

    # Reset the environment and get the initial observation
    observation = env.reset()

    # Run 100 steps
    for step in range(100):
        # Sample a random action or choose based on some policy
        action = env.action_space.sample()

        # Take the step in the environment and collect the result
        observation, reward, terminated, truncated, info = env.step(action)

        # Print or log the step's result (optional)
        print(f"Step {step + 1}: Reward = {reward}, Terminated = {terminated}, Truncated = {truncated}")

        # Check if the episode is over
        if terminated or truncated:
            print(f"Environment finished at step {step + 1}. Resetting...")
            observation = env.reset()

    print("Completed 100 steps.")



# Execute os testes
if __name__ == "__main__":
    test_init()
    test_check_env()
    check_step()

