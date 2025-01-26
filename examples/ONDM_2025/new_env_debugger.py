import os
import csv
import time
import random
import numpy as np
import gymnasium as gym

from datetime import datetime
from typing import Tuple

# Importações específicas do optical-networking-gym
import optical_networking_gym.wrappers.qrmsa_gym
from optical_networking_gym.heuristics.heuristics import (
    shortest_available_path_first_fit_best_modulation,
    shortest_available_path_lowest_spectrum_best_modulation,
    best_modulation_load_balancing,
    load_balancing_best_modulation,
)
from optical_networking_gym.topology import Modulation, get_topology

# ===================================================
# 1. Definição das Modulações (mesmas do PPO)
# ===================================================
def define_modulations() -> Tuple[Modulation, ...]:
    """
    Define e retorna as modulações utilizadas no ambiente.
    """
    return (
        Modulation(name="BPSK",  maximum_length=100000, spectral_efficiency=1,  minimum_osnr=12.6, inband_xt=-14),
        Modulation(name="QPSK",  maximum_length=2000,   spectral_efficiency=2,  minimum_osnr=12.6, inband_xt=-17),
        Modulation(name="8QAM",  maximum_length=1000,   spectral_efficiency=3,  minimum_osnr=18.6, inband_xt=-20),
        Modulation(name="16QAM", maximum_length=500,    spectral_efficiency=4,  minimum_osnr=22.4, inband_xt=-23),
        Modulation(name="32QAM", maximum_length=250,    spectral_efficiency=5,  minimum_osnr=26.4, inband_xt=-26),
        Modulation(name="64QAM", maximum_length=125,    spectral_efficiency=6,  minimum_osnr=30.4, inband_xt=-29),
    )


# ===================================================
# 2. Configuração do Ambiente
# ===================================================
def create_environment():
    """
    Cria e retorna:
      - objeto de topologia,
      - dicionário de argumentos (env_args).
    """
    # Topologia (caminho usado no PPO) e modulações
    topology_name = "ring_4"  # Nome de referência
    topology_path = rf"C:\Users\talle\Documents\Mestrado\optical-networking-gym\examples\topologies\ring_4.txt"
    cur_modulations = define_modulations()

    # Cria objeto de topologia
    topology = get_topology(
        topology_path,
        topology_name,
        cur_modulations,
        80,    # Ex: Distância máxima entre amplificadores (ou outro parâmetro da topologia)
        0.2,   # Atenuação
        4.5,   # Noise figure
        2      # k_paths
    )

    # Parâmetros do ambiente (iguais ao PPO)
    seed = 10
    random.seed(seed)
    np.random.seed(seed)

    env_args = dict(
        topology=topology,
        seed=seed,
        allow_rejection=True,
        load=210,                    # Mesmo load do PPO
        episode_length=1000,         # Mesmo episode_length do PPO
        num_spectrum_resources=15,  # Mesmo número de slots do PPO
        launch_power_dbm=0,          # Mesmo launch power do PPO
        frequency_slot_bandwidth=12.5e9,
        frequency_start=3e8 / 1565e-9,
        bandwidth= 15* 12.5e9,
        bit_rate_selection="discrete",
        bit_rates=(10, 40),#,100, 400),
        margin=0,
        measure_disruptions=False,
        file_name="",  # Podemos deixar vazio, pois não estamos logando em cada step
        k_paths=1,     # Mesmo valor do PPO
    )
    return topology, env_args


# ===================================================
# 3. Função para rodar ambiente com heurística
# ===================================================
def run_first_fit_environment(
    n_eval_episodes: int,
    topology,
    env_args: dict,
    csv_output: str = "first_fit_results.csv",
):
    """
    Executa a heurística "shortest_available_path_first_fit_best_modulation"
    por n_eval_episodes, salvando métricas de cada episódio em um CSV.

    :param n_eval_episodes: número de episódios.
    :param topology: objeto de topologia (optical_networking_gym.topology.Topology).
    :param env_args: dicionário de parâmetros do ambiente.
    :param csv_output: caminho do CSV de saída.
    """
    # Seleciona a função da heurística (first-fit)
    fn_heuristic = shortest_available_path_first_fit_best_modulation
    # fn_heuristic = best_modulation_load_balancing
    
    # Cria instância do ambiente
    env = gym.make("QRMSAEnvWrapper-v0", **env_args)

    # Prepara arquivo CSV para escrita
    if os.path.exists(csv_output):
        pass
        # raise FileExistsError(f"O arquivo '{csv_output}' já existe! Escolha outro nome ou mova/renomeie o arquivo atual.")

    with open(csv_output, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Cabeçalho
        # Ex.: 
        # episode, service_blocking_rate, episode_service_blocking_rate, 
        # bit_rate_blocking_rate, episode_bit_rate_blocking_rate, 
        # modulation_1, modulation_2, ..., episode_disrupted_services, episode_time
        header = [
            "episode",
            "service_blocking_rate",
            "episode_service_blocking_rate",
            "bit_rate_blocking_rate",
            "episode_bit_rate_blocking_rate",
        ]
        # Para cada modulação, cria a coluna "modulation_<spectral_efficiency>"
        for mod in env.unwrapped.env.modulations:
            header.append(f"modulation_{mod.spectral_efficiency}")
        header += ["episode_disrupted_services", "episode_time"]
        writer.writerow(header)

        # Loop de episódios
        for ep in range(n_eval_episodes):
            osnr_count = 0
            resource_count = 0
            print(f"===== Episódio {ep} =====")
            obs, info = env.reset()
            print("================= reset =================")
            for lnk in env.unwrapped.env.topology.edges():
                index = env.unwrapped.env.topology[lnk[0]][lnk[1]]["index"]
                print(f"Link {lnk}: {env.unwrapped.env.topology.graph["available_slots"][index,:]}")
                print(f"running services:" )
                print(f"{env.unwrapped.env.topology[lnk[0]][lnk[1]]["running_services"]}")
            print("================= reset =================")
            done = False
            start_time = time.time()

            while not done:
                print(f"current service: {env.unwrapped.env.current_service}")
                action, bl_osnr, bl_resource = fn_heuristic(env.unwrapped.env)
                if bl_osnr:
                    osnr_count += 1
                if bl_resource:
                    resource_count += 1               
                obs, reward, terminated, truncated, info = env.step(action)
                print("================= step =================")
                print(f"Action: {action}, unwraped action: {env.unwrapped.env.decimal_to_array(int(action))}")
                for lnk in env.unwrapped.env.topology.edges():
                    index = env.unwrapped.env.topology[lnk[0]][lnk[1]]["index"]
                    # print(f"Link {lnk}: {env.unwrapped.env.topology.graph["available_slots"][index,:]}")
                    # print(f"running services:" )
                    # for service in env.unwrapped.env.topology[lnk[0]][lnk[1]]["running_services"]:
                    #     print(f"ID: {service.service_id}, src: {service.source}, tgt: {service.destination}, Path: {service.path}, init_slot: {service.initial_slot}, numb_slots: {service.number_slots}, BW: {service.bandwidth}, center_freq: {service.center_frequency}, mod: {service.current_modulation}, OSNR: {service.OSNR}, ASE: {service.ASE}, NLI: {service.NLI}\n")
                done = terminated or truncated

            end_time = time.time()
            ep_time = end_time - start_time
            print("Bloqueio por recursos:", resource_count)
            print("Bloqueio por OSNR:", osnr_count)


            # Coletando métricas do info
            row = [
                ep,
                info.get("service_blocking_rate", 0.0),
                info.get("episode_service_blocking_rate", 0.0),
                info.get("bit_rate_blocking_rate", 0.0),
                info.get("episode_bit_rate_blocking_rate", 0.0),
            ]
            # Adiciona contagem de modulações
            modulations = [
                info.get(f"modulation_{i}.0", 0.0) for i in range(1, 7)
            ]
            row.extend(modulations)
            # Disrupted services e tempo
            row.append(info.get("episode_disrupted_services", 0))
            row.append(f"{ep_time:.2f}")

            # Escreve a linha no CSV
            writer.writerow(row)

    print(f"\nFinalizado! Resultados salvos em: {csv_output}")


# ===================================================
# 4. Rotina Principal
# ===================================================
def main():
    # (A) Carregamos a topologia e parâmetros do ambiente (iguais ao PPO)
    topology, env_args = create_environment()

    # (B) Definimos o número de episódios para 5
    n_eval_episodes = 2

    # (C) Executamos a heurística First-Fit e salvamos em CSV
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_first_fit_environment(
        n_eval_episodes=n_eval_episodes,
        topology=topology,
        env_args=env_args,
        csv_output=f"-4dbm_newgsnr__first_fit_nobel_us_results_{time}.csv",
    )

if __name__ == "__main__":
    main()
