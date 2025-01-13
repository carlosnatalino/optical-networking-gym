import os
import numpy as np
import gymnasium as gym
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import time
from datetime import datetime
import random

# Importações do optical_networking_gym
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.heuristics.heuristics import (
    shortest_available_path_first_fit_best_modulation
)

# Definir modulações reduzidas
def define_modulations_reduced() -> Tuple[Modulation, ...]:
    return (
        Modulation(name="BPSK",  maximum_length=100000, spectral_efficiency=1,  minimum_osnr=12.6, inband_xt=-14),
        Modulation(name="QPSK",  maximum_length=2000,   spectral_efficiency=2,  minimum_osnr=12.6, inband_xt=-17),
    )

# Função principal de depuração
def run_debug_environment(
    n_eval_episodes: int = 10,
    monitor_file_name: str = "debug_episode_info",
    seed: int = 42
) -> None:
    """
    Executa o ambiente de depuração utilizando a heurística First Fit.

    Args:
        n_eval_episodes (int): Número de episódios para avaliar.
        monitor_file_name (str): Nome base para o arquivo de log CSV.
        seed (int): Semente para reprodução.
    """
    # Definir modulações reduzidas
    modulations = define_modulations_reduced()

    # Configurar topologia reduzida (ring-4)
    topology_name = "ring_4"

    topology_path = (
        rf"C:\Users\talle\Documents\Mestrado\optical-networking-gym\examples\topologies\{topology_name}.txt"
        # Ajuste o caminho conforme necessário
    )

    # Verificar existência do arquivo de topologia
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Topologia não encontrada em {topology_path}. Verifique o caminho.")

    topology = get_topology(
        topology_path,
        topology_name,
        modulations,
        80,
        0.2,
        4.5,
        2
    )

    # Visualizar topologia (opcional)
    plt.figure(figsize=(8, 6))
    pos = nx.circular_layout(topology)
    nx.draw(topology, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1000)
    edge_labels = nx.get_edge_attributes(topology, "length")
    nx.draw_networkx_edge_labels(topology, pos, edge_labels=edge_labels, font_color="red")
    plt.title(f"Topologia: {topology.name.upper()}")
    plt.axis("off")
    plt.show()

    # Configuração do ambiente
    episode_length = 100
    load = 100
    launch_power_dbm = 0
    bit_rates = (40, 80, 120)

    env_args = dict(
        topology=topology,
        seed=seed,
        allow_rejection=True,
        load=load,
        episode_length=episode_length,
        num_spectrum_resources=50,
        launch_power_dbm=launch_power_dbm,
        bandwidth=50 * 12.5e9,
        frequency_start=193.1e12,
        frequency_slot_bandwidth=12.5e9,
        bit_rate_selection="discrete",
        bit_rates=bit_rates,
        margin=0.0,
        file_name="",
        measure_disruptions=True,
        k_paths=2,
    )

    env = gym.make("QRMSAEnvWrapper-v0", **env_args)

    # Abrir arquivo de log
    monitor_final_name = "_".join([
        monitor_file_name,
        topology.name,
        str(env.unwrapped.env.launch_power_dbm),
        str(env.unwrapped.env.load),
        "debug",
        datetime.now().strftime("%Y%m%d-%H%M%S")
    ]) + ".csv"

    if os.path.exists(monitor_final_name):
        raise FileExistsError(f"Arquivo {monitor_final_name} já existe! Escolha outro nome ou remova o existente.")

    with open(monitor_final_name, "wt", encoding="utf-8") as file_handler:
        # Cabeçalhos do CSV
        headers = [
            "episode",
            "service_blocking_rate",
            "bit_rate_blocking_rate",
            "modulation_1.0",
            "modulation_2.0",
            "episode_disrupted_services",
            "episode_time"
        ]
        file_handler.write(",".join(headers) + "\n")

        # Executar episódios
        for ep in range(n_eval_episodes):
            observation, info = env.reset()
            done = False
            start_time = time.time()

            while not done:
                action = shortest_available_path_first_fit_best_modulation(env)

                # Lidar com ação de rejeição
                if action is None:
                    action = env.reject_action

                # Passar ação para o ambiente
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            end_time = time.time()

            # Extrair métricas
            service_blocking_rate = info.get("episode_service_blocking_rate", 0.0)
            bit_rate_blocking_rate = info.get("episode_bit_rate_blocking_rate", 0.0)
            disrupted_services = info.get("episode_disrupted_services", 0.0)
            modulation_1 = info.get("modulation_1.0", 0)
            modulation_2 = info.get("modulation_2.0", 0)

            # Escrever no CSV
            row = [
                str(ep),
                f"{service_blocking_rate:.4f}",
                f"{bit_rate_blocking_rate:.4f}",
                str(modulation_1),
                str(modulation_2),
                f"{disrupted_services:.4f}",
                f"{(end_time - start_time):.2f}"
            ]
            file_handler.write(",".join(row) + "\n")
            file_handler.flush()

            # Imprimir no console
            print(f"Ep {ep+1}/{n_eval_episodes}:")
            print(f"  Service Blocking Rate: {service_blocking_rate:.4f}")
            print(f"  Bit Rate Blocking Rate: {bit_rate_blocking_rate:.4f}")
            print(f"  Modulation 1.0 Count: {modulation_1}")
            print(f"  Modulation 2.0 Count: {modulation_2}")
            print(f"  Disrupted Services: {disrupted_services:.4f}")
            print(f"  Episode Time: {(end_time - start_time):.2f} seconds")
            print("-" * 50)

# Configuração da semente
seed = 42
random.seed(seed)
np.random.seed(seed)

# Executar depuração
run_debug_environment(
    n_eval_episodes=10,
    monitor_file_name="debug_episode_info_ring4_k2",
    seed=seed
)
