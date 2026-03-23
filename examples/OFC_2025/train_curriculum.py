#!/usr/bin/env python3
"""
Treinamento PPO com Curriculum Learning para QRMSA
Treina progressivamente aumentando a dificuldade (load)

Estratégia:
1. Load baixo (200) → modelo aprende básico
2. Load médio (275) → refina políticas  
3. Load alto (350) → desafio final

Autor: Sistema OFC 2025
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from train_n_steps import EnvConfig, TrainingConfig, train, make_vec_env
from sb3_contrib import MaskablePPO


def curriculum_training():
    """Treinamento em 3 estágios com dificuldade crescente"""
    
    print("="*80)
    print(" CURRICULUM LEARNING - TREINAMENTO PROGRESSIVO")
    print("="*80)
    
    base_exp_name = f"curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # ESTÁGIO 1: Load baixo - aprender o básico
    print("\n" + "="*80)
    print(" ESTÁGIO 1/3: Load BAIXO (200 Erlangs) - Aprendizado Básico")
    print("="*80)
    
    env_cfg_1 = EnvConfig(
        topology_name="nobel-eu",
        load=200,  # BAIXO - fácil
        episode_length=2000,
        num_spectrum_resources=400,
        k_paths=7,
        modulations_to_consider=6,  # Usar TODAS as modulações
        defragmentation=False,
        seed=42
    )
    
    train_cfg_1 = TrainingConfig(
        n_episodes=15000,  # 15k episódios
        n_envs=44,
        ppo_profile="intensive",
        experiment_name=f"{base_exp_name}_stage1_load200",
        log_freq=5000,
        verbose=1
    )
    
    model_1, run_dir_1 = train(env_cfg_1, train_cfg_1)
    print(f"\n[OK] Estágio 1 completo! Modelo salvo em: {run_dir_1}")
    
    # ESTÁGIO 2: Load médio - refinamento
    print("\n" + "="*80)
    print(" ESTÁGIO 2/3: Load MÉDIO (275 Erlangs) - Refinamento")
    print("="*80)
    
    env_cfg_2 = EnvConfig(
        topology_name="nobel-eu",
        load=275,  # MÉDIO
        episode_length=2000,
        num_spectrum_resources=400,
        k_paths=7,
        modulations_to_consider=6,
        defragmentation=False,
        seed=42
    )
    
    train_cfg_2 = TrainingConfig(
        n_episodes=20000,  # 20k episódios
        n_envs=44,
        ppo_profile="intensive",
        experiment_name=f"{base_exp_name}_stage2_load275",
        log_freq=5000,
        verbose=1
    )
    
    # Cria novo ambiente com load médio
    env_2 = make_vec_env(env_cfg_2, n_envs=train_cfg_2.n_envs)
    
    # CONTINUA do modelo anterior!
    model_1.set_env(env_2)
    
    # Treina mais
    from callbacks import create_callbacks
    from stop_after_episodes import StopAfterEpisodesCallback
    from stable_baselines3.common.callbacks import CallbackList
    
    checkpoint_dir_2 = Path(train_cfg_2.base_dir) / train_cfg_2.experiment_name / "checkpoints"
    checkpoint_dir_2.mkdir(parents=True, exist_ok=True)
    
    callbacks_2 = create_callbacks(
        save_path=str(checkpoint_dir_2),
        checkpoint_freq=5000,
        log_freq=train_cfg_2.log_freq,
        verbose=1
    )
    callbacks_2.append(StopAfterEpisodesCallback(n_episodes=train_cfg_2.n_episodes, verbose=1))
    
    print("\nContinuando treinamento com load médio...")
    model_1.learn(
        total_timesteps=100_000_000,
        callback=CallbackList(callbacks_2),
        reset_num_timesteps=False,  # NÃO reseta contadores
        tb_log_name=train_cfg_2.experiment_name,
    )
    
    model_1.save(str(checkpoint_dir_2 / "final_model"))
    env_2.close()
    
    print(f"\n[OK] Estágio 2 completo!")
    
    # ESTÁGIO 3: Load alto - desafio final
    print("\n" + "="*80)
    print(" ESTÁGIO 3/3: Load ALTO (350 Erlangs) - Desafio Final")
    print("="*80)
    
    env_cfg_3 = EnvConfig(
        topology_name="nobel-eu",
        load=350,  # ALTO - difícil
        episode_length=2000,
        num_spectrum_resources=400,
        k_paths=7,
        modulations_to_consider=6,
        defragmentation=False,
        seed=42
    )
    
    train_cfg_3 = TrainingConfig(
        n_episodes=25000,  # 25k episódios
        n_envs=44,
        ppo_profile="intensive",
        experiment_name=f"{base_exp_name}_stage3_load350",
        log_freq=5000,
        verbose=1
    )
    
    env_3 = make_vec_env(env_cfg_3, n_envs=train_cfg_3.n_envs)
    model_1.set_env(env_3)
    
    checkpoint_dir_3 = Path(train_cfg_3.base_dir) / train_cfg_3.experiment_name / "checkpoints"
    checkpoint_dir_3.mkdir(parents=True, exist_ok=True)
    
    callbacks_3 = create_callbacks(
        save_path=str(checkpoint_dir_3),
        checkpoint_freq=5000,
        log_freq=train_cfg_3.log_freq,
        verbose=1
    )
    callbacks_3.append(StopAfterEpisodesCallback(n_episodes=train_cfg_3.n_episodes, verbose=1))
    
    print("\nContinuando treinamento com load alto...")
    model_1.learn(
        total_timesteps=100_000_000,
        callback=CallbackList(callbacks_3),
        reset_num_timesteps=False,
        tb_log_name=train_cfg_3.experiment_name,
    )
    
    final_path = checkpoint_dir_3 / "final_model_curriculum"
    model_1.save(str(final_path))
    env_3.close()
    
    print("\n" + "="*80)
    print(" CURRICULUM LEARNING COMPLETO!")
    print("="*80)
    print(f"\nModelo final salvo em: {final_path}.zip")
    print("\nEstágios completados:")
    print(f"  1. Load 200 Erlangs - 15k episódios")
    print(f"  2. Load 275 Erlangs - 20k episódios")
    print(f"  3. Load 350 Erlangs - 25k episódios")
    print(f"  TOTAL: 60k episódios")
    
    return model_1, final_path


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    mp.freeze_support()
    
    model, model_path = curriculum_training()
