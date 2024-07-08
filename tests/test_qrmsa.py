import os
import pickle

import networkx as nx

from optical_networking_gym.envs.qrmsa import QRMSAEnv


def get_topology(name: str) -> nx.Graph:
    G = nx.Graph()

    # Add nodes (vertices) to the graph
    G.add_node('A')
    G.add_node('B')
    G.add_node('C')
    G.add_node('D')
    G.add_node('E')
    G.add_node('F')

    # Add links (edges) between nodes
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('A', 'D')
    G.add_edge('B', 'E')
    G.add_edge('C', 'F')
    G.add_edge('D', 'E')
    G.add_edge('E', 'F')

    return G
    # with open(os.path.join("examples", "topologies", name), "rb") as file:
    #     return pickle.load(file)

# from optical_networking_gym.tests.utils import get_topology

def test_init():
    topology = get_topology("germany50_gn_5-paths_6-modulations.h5")
    env = QRMSAEnv(topology=topology, num_spectrum_resources=360)
