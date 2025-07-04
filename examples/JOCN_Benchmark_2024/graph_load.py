import argparse
import logging
import os
import random
from typing import List, Tuple
from multiprocessing import Pool

import numpy as np
import time
from datetime import datetime

from optical_networking_gym.topology import Modulation, get_topology

# ===================================================
# Função para definir as cargas com base no nome da topologia
# ===================================================
def get_loads(topology_name: str) -> np.ndarray:
    if topology_name == "nobel-eu.xml":
        return np.arange(100, 501, 100)
    elif topology_name == "germany50.xml":
        return np.arange(300, 801, 50)
    elif topology_name == "janos-us.xml":
        return np.arange(100, 601, 50)
    elif topology_name == "nsfnet_chen.txt":
        return np.arange(100, 601, 50)
    elif topology_name == "ring_4.txt":
        return np.arange(100, 601, 50)
    else:
        raise ValueError(f"Unknown topology name: {topology_name}")

# ===================================================
# Função que executa o ambiente de simulação
# (adaptada do código do launch power)
# ===================================================
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
    defragmentation,
    n_defrag_services,
    gen_observation,
) -> None:
    """
    Executa o ambiente com a heurística especificada e salva os resultados em um arquivo CSV.

    :param n_eval_episodes: Número de episódios a serem executados.
    :param heuristic: Índice da heurística a ser utilizada.
    :param monitor_file_name: Nome base para o arquivo CSV de monitoramento.
    :param topology: Objeto de topologia.
    :param seed: Semente para geração de números aleatórios.
    :param allow_rejection: Permitir rejeição de solicitações.
    :param load: Carga a ser utilizada na simulação.
    :param episode_length: Número de chegadas por episódio.
    :param num_spectrum_resources: Número de recursos de espectro.
    :param launch_power_dbm: Potência de lançamento em dBm.
    :param bandwidth: Largura de banda.
    :param frequency_start: Frequência inicial.
    :param frequency_slot_bandwidth: Largura de banda do slot de frequência.
    :param bit_rate_selection: Seleção de taxa de bits.
    :param bit_rates: Taxas de bits disponíveis.
    :param margin: Margem.
    :param file_name: Nome do arquivo para salvar serviços.
    :param measure_disruptions: Medir interrupções.
    """
    from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
    from optical_networking_gym.heuristics.heuristics import (
        shortest_available_path_first_fit_best_modulation,
        shortest_available_path_lowest_spectrum_best_modulation,
        best_modulation_load_balancing,
        load_balancing_best_modulation,
        rnd,
        heuristic_shortest_available_path_first_fit_best_modulation,
        heuristic_highest_snr,
        heuristic_lowest_fragmentation,
    )

    # Configurações do ambiente
    env_args = dict(
        topology=topology,
        seed=seed,
        allow_rejection=allow_rejection,
        load=load,
        episode_length=episode_length,
        num_spectrum_resources=num_spectrum_resources,
        launch_power_dbm=launch_power_dbm,
        bandwidth=bandwidth,
        frequency_start=frequency_start,
        frequency_slot_bandwidth=frequency_slot_bandwidth,
        bit_rate_selection=bit_rate_selection,
        bit_rates=bit_rates,
        margin=margin,
        file_name=file_name,
        measure_disruptions=measure_disruptions,
        k_paths=5, 
        modulations_to_consider=6,
        defragmentation=defragmentation,
        n_defrag_services=n_defrag_services,
        gen_observation=gen_observation,
    )

    # Seleção da heurística baseada no índice
    if heuristic == 1:
        fn_heuristic = heuristic_shortest_available_path_first_fit_best_modulation
    elif heuristic == 2:
        fn_heuristic = heuristic_highest_snr
    elif heuristic == 3:
        fn_heuristic = heuristic_lowest_fragmentation
    elif heuristic == 4:
        fn_heuristic = load_balancing_best_modulation
    else:
        raise ValueError(f"Heuristic index `{heuristic}` is not found!")

    # Criação do ambiente
    env = QRMSAEnvWrapper(**env_args)
    env.reset()

    if monitor_file_name is None:
        raise ValueError("Missing monitor file name")

    # Definição do nome final do arquivo CSV de monitoramento
    monitor_final_name = "_".join([
        monitor_file_name, 
        topology.name, 
        str(env.env.launch_power_dbm), 
        str(env.env.load) + "_nw_cnr_nobel-eu.csv"
    ])

    # Preparação do arquivo CSV
    with open(monitor_final_name, "wt", encoding="UTF-8") as file_handler:
        file_handler.write(f"# Date: {datetime.now()}\n")
        header = (
            "episode,service_blocking_rate,episode_service_blocking_rate,"
            "bit_rate_blocking_rate,episode_bit_rate_blocking_rate, episode_service_realocations, episode_defrag_cicles"
        )
        for mf in env.env.modulations:
            header += f",modulation_{mf.spectral_efficiency}"
        header += ",episode_disrupted_services,episode_time,"
        header += "mean_gsnr\n"
        file_handler.write(header)

        # Execução dos episódios
        for ep in range(n_eval_episodes):
            obs, info = env.reset()
            done = False
            start_time = time.time()
            while not done:
                action,_,_ = fn_heuristic(env)
                _, _, done, _, info = env.step(action)
            end_time = time.time()
            ep_time = end_time - start_time

            print(f"Episódio {ep} finalizado.")
            print(info)

            row = (
                f"{ep},{info['service_blocking_rate']},"
                f"{info['episode_service_blocking_rate']},"
                f"{info['bit_rate_blocking_rate']},"
                f"{info['episode_bit_rate_blocking_rate']},"
                f"{info['episode_service_realocations']},"
                f"{info['episode_defrag_cicles']}"
            )
            for mf in env.env.modulations:
                row += f",{info.get(f'modulation_{mf.spectral_efficiency}', 0.0)}"
            row += f",{info.get('episode_disrupted_services', 0)},{ep_time:.2f}"
            mean_gsnr = 0.0
            for service in env.env.topology.graph["services"]:
                mean_gsnr += service.OSNR
            mean_gsnr /= len(env.env.topology.graph["services"])
            row += f",{mean_gsnr}\n"
            file_handler.write(row)
                

    print(f"\nFinalizado! Resultados salvos em: {monitor_final_name}")

# ===================================================
# Configuração de logging, semente e argumentos de entrada
# ===================================================
logging.getLogger("rmsaenv").setLevel(logging.INFO)
np.set_printoptions(linewidth=np.inf)

seed = 50
random.seed(seed)
np.random.seed(seed)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simulação de Rede Óptica - Variação de Carga')
    parser.add_argument(
        '-t', '--topology_file',
        type=str,
        default='nobel-eu.xml',#'nobel-eu.xml',
        help='Arquivo de topologia a ser utilizado (default: nsfnet_chen.txt)'
    )
    parser.add_argument(
        '-e', '--num_episodes',
        type=int,
        default=25,
        help='Número de episódios a serem simulados (default: 5)'
    )
    parser.add_argument(
        '-s', '--episode_length',
        type=int,
        default=1000,
        help='Número de chegadas por episódio (default: 1000)'
    )
    parser.add_argument(
        '-th', '--threads',
        type=int,
        default=2,
        help='Número de threads para execução das simulações (default: 2)'
    )
    # Argumento para a heurística a ser utilizada
    parser.add_argument(
        '-hi', '--heuristic_index',
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help='Índice da heurística (1: First Fit, 2: Lowest Spectrum, 3: Load Balancing Modulation, 4: Load Balancing Best Modulation)'
    )
    # Nome base para o arquivo CSV de monitoramento
    parser.add_argument(
        '-mf', '--monitor_file_name',
        type=str,
        default='examples/jocn_benchmark_2024/results/load_episodes',
        help='Nome base para o arquivo CSV de monitoramento'
    )
    return parser.parse_args()

# ===================================================
# Função principal
# ===================================================
def main():
    args = parse_arguments()

    # Definição das modulações (valores atualizados conforme exemplo do launch power)
    cur_modulations: Tuple[Modulation, ...] = (
        Modulation(
            name="BPSK",
            maximum_length=100_000,
            spectral_efficiency=1,
            minimum_osnr=3.71,
            inband_xt=-14,
        ),
        Modulation(
            name="QPSK",
            maximum_length=2_000,
            spectral_efficiency=2,
            minimum_osnr=6.72,
            inband_xt=-17,
        ),
        Modulation(
            name="8QAM",
            maximum_length=1_000,
            spectral_efficiency=3,
            minimum_osnr=10.84,
            inband_xt=-20,
        ),
        Modulation(
            name="16QAM",
            maximum_length=500,
            spectral_efficiency=4,
            minimum_osnr=13.24,
            inband_xt=-23,
        ),
        Modulation(
            name="32QAM",
            maximum_length=250,
            spectral_efficiency=5,
            minimum_osnr=16.16,
            inband_xt=-26,
        ),
        Modulation(
            name="64QAM",
            maximum_length=125,
            spectral_efficiency=6,
            minimum_osnr=19.01,
            inband_xt=-29,
        ),
    )

    attenuation_db_km = 0.2
    default_noise_figure_db = 4.5

    # Caminho da topologia
    topology_path = "/home/backu/Documentos/ONGsimulate/defrag_ff/optical-networking-gym/examples/topologies/nobel-eu.xml"
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Arquivo de topologia '{topology_path}' não encontrado.")

    # Carregamento da topologia
    topology = get_topology(
        topology_path,      # Caminho do arquivo de topologia
        None,               # Nome da topologia (ajustável se necessário)
        cur_modulations,    # Tuple de modulações
        80,                 # Comprimento máximo de span (km)
        attenuation_db_km,  # Atenuação padrão (dB/km)
        default_noise_figure_db,  # Figura de ruído padrão (dB)
        5                   # Número de caminhos mais curtos a computar entre pares de nós
    )

    # Parâmetros de simulação
    threads = args.threads
    bandwidth = 4e12
    frequency_start = 3e8 / 1565e-9
    frequency_slot_bandwidth = 12.5e9
    bit_rates = (10, 40, 100, 400, 1000)
    margin = 0

    launch_power = 1.0

    loads = get_loads(args.topology_file)

    strategies = list(range(1, 5))

    env_args = []
    for current_load in loads:
        for strategy in [1]:
            for mensure in [[False,0], [True, 0]]:
                sim_args = (
                    args.num_episodes,              # n_eval_episodes
                    strategy,                       # heuristic_index
                    f"{args.monitor_file_name}_{strategy}",  # monitor_file_name base
                    topology,                       # topology
                    seed,                           # seed
                    True,                           # allow_rejection
                    current_load,                   # load (varia)
                    args.episode_length,            # episode_length
                    320,                            # num_spectrum_resources
                    launch_power,                   # launch_power_dbm
                    bandwidth,                      # bandwidth
                    frequency_start,                # frequency_start
                    frequency_slot_bandwidth,       # frequency_slot_bandwidth
                    "discrete",                     # bit_rate_selection
                    bit_rates,                      # bit_rates
                    margin,                         # margin
                    f"examples/jocn_benchmark_2024/results/load_services_{strategy}",  # file_name para serviços
                    False,                          # measure_disruptions
                    False,                # measure_disruptions (True/False)
                    0,                # measure_disruptions (valor)
                    False,                          # run observation
                )
                env_args.append(sim_args)

    # Execução das simulações utilizando multiprocessing se houver mais de uma thread
    print("Iniciando simulações...")
    if threads > 1:
        with Pool(processes=threads) as pool:
            pool.starmap(run_environment, env_args)
    else:
        for arg in env_args:
            run_environment(*arg)

    print("Todas as simulações foram executadas.")

if __name__ == "__main__":
    main()
