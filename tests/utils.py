import os
import pickle

import networkx as nx


def get_topology(name: str) -> nx.Graph:
    with open(os.path.join("examples", "topologies", name), "rb") as file:
        return pickle.load(file)
