# import os
from typing import Tuple

import networkx as nx
from gymnasium.utils.env_checker import check_env

from optical_networking_gym.topology import Modulation
from optical_networking_gym.envs.rmsa import RMSAEnv


# def get_topology(name: str) -> nx.Graph:
#     G: nx.Graph = nx.Graph()

#     # Add nodes (vertices) to the graph
#     G.add_node('A')
#     G.add_node('B')
#     G.add_node('C')
#     G.add_node('D')
#     G.add_node('E')
#     G.add_node('F')

#     # Add links (edges) between nodes
#     G.add_edge('A', 'B')
#     G.add_edge('A', 'C')
#     G.add_edge('A', 'D')
#     G.add_edge('B', 'E')
#     G.add_edge('C', 'F')
#     G.add_edge('D', 'E')
#     G.add_edge('E', 'F')

#     return G
    # with open(os.path.join("examples", "topologies", name), "rb") as file:
    #     return pickle.load(file)

from optical_networking_gym.topology import get_topology


def test_init() -> None:
    cur_modulations: Tuple[Modulation] = (
        # the first (lowest efficiency) modulation format needs to have maximum length
        # greater or equal to the longest path in the topology.
        # Here we put 100,000 km to be on the safe side
        Modulation(
            name="BPSK",
            maximum_length=100_000,
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
    # topology = get_topology("germany50_gn_5-paths_6-modulations.h5")
    topology = get_topology(
        "examples/topologies/nsfnet_chen.txt",
        "NSFNET",
        cur_modulations,
        80,
        0.2,
        4.5,
        5,
    )
    _ = RMSAEnv(topology=topology, num_spectrum_resources=360)


def test_check_env() -> None:
    topology = get_topology("germany50_gn_5-paths_6-modulations.h5")
    env = RMSAEnv(topology=topology, num_spectrum_resources=360)
    # check_env(env)
