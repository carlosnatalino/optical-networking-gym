import cProfile
import pstats
import io
from datetime import datetime

import random
import numpy as np

from utils import SimulationUtils
from optical_networking_gym.heuristics.heuristics import shortest_available_path_first_fit_best_modulation
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper

def profile_step():
    # 1) Prepare ambiente igual ao seu main()
    env_args = SimulationUtils.create_environment(
        topology_name="nobel-eu",
        modulation_names="BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM",
        seed=10,
        bit_rates=(10, 40, 100, 400),
        load=300,
        num_spectrum_resources=320,
        episode_length=1000,
        modulations_to_consider=6,
        defragmentation=False,
        k_paths=3,
        gen_observation=True,
    )
    env = QRMSAEnvWrapper(**env_args)

    # 2) Profile 1000 episódios completos
    pr = cProfile.Profile()
    pr.enable()

    episode_rewards = []
    episode_lengths = []
    blocking_rates = []
    
    for episode in range(1):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            action = shortest_available_path_first_fit_best_modulation(info['mask'])
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = done or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if 'episode_service_blocking_rate' in info:
            blocking_rates.append(info['episode_service_blocking_rate'])
        
        if (episode + 1) % 100 == 0:
            print(f"Completed {episode + 1} episodes...")

    pr.disable()

    # 3) Formate e imprima relatório
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')  # ou 'tottime', 'ncalls'
    ps.print_stats(50)  # top 50 linhas

    print("\n=== Profiling de 1000 Episódios ===")
    print(s.getvalue())
    print(f"\nEstatísticas dos Episódios:")
    print(f"  Média de Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Desvio Padrão de Reward: {np.std(episode_rewards):.2f}")
    print(f"  Média de Comprimento: {np.mean(episode_lengths):.2f}")
    print(f"  Desvio Padrão de Comprimento: {np.std(episode_lengths):.2f}")
    print(f"\nInformações do Último Episódio:")
    print(info)
    if blocking_rates:
        print(f"  Média de Blocking Rate: {np.mean(blocking_rates):.4f}")

if __name__ == "__main__":
    profile_step()
