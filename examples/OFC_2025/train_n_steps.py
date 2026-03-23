#!/usr/bin/env python3
"""
Sistema de treinamento PPO para QRMSA - SIMPLIFICADO

Usa callbacks prontos do callbacks.py:
- BlockingRateCallback: Blocking probability por episodio
- RewardStatsCallback: Reward por episodio e media por step
- FragmentationCallback: Uso de modulacoes e metricas de defrag
- StopAfterEpisodesCallback: Para apos N episodios

Autor: Sistema de treinamento OFC 2025
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Imports do projeto
from utils import SimulationUtils
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from stop_after_episodes import StopAfterEpisodesCallback
from hyperparams import get_hyperparams
from callbacks import create_callbacks

# Imports SB3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


# ============================================================================
# CONFIGURACOES
# ============================================================================

@dataclass
class EnvConfig:
    """Configuracao do ambiente QRMSA"""
    
    # Topologia
    topology_name: str = "nobel-eu"
    
    # Modulacoes
    modulation_names: str = "BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM"
    modulations_to_consider: int = 4
    
    # Espectro
    num_spectrum_resources: int = 320
    
    # Trafico
    bit_rates: tuple = (10, 40, 100, 400)
    load: int = 350  # Erlangs - AUMENTADO
    
    # Episodio
    episode_length: int = 1000  # REDUZIDO para teste mais rápido
    
    # Roteamento
    k_paths: int = 5
    
    # Recursos
    defragmentation: bool = False
    gen_observation: bool = True
    
    # Seed
    seed: int = 42


@dataclass
class TrainingConfig:
    """Configuracao do treinamento"""
    
    # Criterio de parada (EPISODIOS, nao timesteps!)
    n_episodes: int = 100
    
    # Ambientes paralelos
    n_envs: int = 1  # Define 1 para debug; use >1 para SubprocVecEnv
    
    # Hiperparametros PPO (profile do hyperparams.py)
    ppo_profile: str = "intensive_v2_refined"  # 🔥 Recomendado: intensive_v2_refined, intensive, default, fast
    
    # Diretorios
    base_dir: str = "./training_runs"
    experiment_name: str = field(default_factory=lambda: f"qrmsa_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Frequencias de log
    log_freq: int = 1000  # A cada N steps
    
    # Verbose
    verbose: int = 0  # 0 para rodar em background sem muito output


# ============================================================================
# FUNCOES AUXILIARES
# ============================================================================

def _mask_fn(env):
    return env.action_masks()


def create_single_env(env_config: EnvConfig, seed: int = None):
    """Cria um unico ambiente QRMSA com action masking"""
    
    # Cria ambiente usando SimulationUtils
    env_args = SimulationUtils.create_environment(
        topology_name=env_config.topology_name,
        modulation_names=env_config.modulation_names,
        seed=seed if seed is not None else env_config.seed,
        bit_rates=env_config.bit_rates,
        load=env_config.load,
        num_spectrum_resources=env_config.num_spectrum_resources,
        episode_length=env_config.episode_length,
        modulations_to_consider=env_config.modulations_to_consider,
        defragmentation=env_config.defragmentation,
        k_paths=env_config.k_paths,
        gen_observation=env_config.gen_observation,
    )
    
    env = QRMSAEnvWrapper(**env_args)
    
    # Aplica action masking
    env = ActionMasker(env, _mask_fn)
    
    return env


def make_vec_env(env_config: EnvConfig, n_envs: int = 1) -> VecEnv:
    """Cria ambientes vetorizados"""
    def make_env(rank: int):
        def _init():
            return create_single_env(env_config, seed=env_config.seed + rank)
        return _init

    env_fns = [make_env(i) for i in range(n_envs)]

    if n_envs == 1:
        return DummyVecEnv(env_fns)

    import multiprocessing as mp

    available_methods = mp.get_all_start_methods()
    if "fork" in available_methods:
        start_method = "fork"
    elif "forkserver" in available_methods:
        start_method = "forkserver"
    else:
        start_method = "spawn"

    if start_method == "spawn":
        print(
            "[WARN] Multiprocessing start method 'spawn' is not compatible with "
            "QRMSAEnv parallel execution. Falling back to DummyVecEnv."
        )
        return DummyVecEnv(env_fns)

    return SubprocVecEnv(env_fns, start_method=start_method)


# ============================================================================
# FUNCAO PRINCIPAL DE TREINAMENTO
# ============================================================================

def train(
    env_config: EnvConfig = EnvConfig(),
    training_config: TrainingConfig = TrainingConfig(),
):
    """
    Funcao principal de treinamento.
    
    Args:
        env_config: Configuracao do ambiente
        training_config: Configuracao do treinamento
    
    Returns:
        Tupla (model, run_dir) com modelo treinado e diretorio do experimento
    """
    
    print("="*80)
    print(" TREINAMENTO PPO - QRMSA")
    print("="*80)
    print(f"\nExperimento: {training_config.experiment_name}")
    print(f"Criterio de parada: {training_config.n_episodes} episodios")
    print(f"Ambientes paralelos: {training_config.n_envs}")
    print(f"\nAmbiente:")
    print(f"  Topologia: {env_config.topology_name}")
    print(f"  Load: {env_config.load} Erlangs")
    print(f"  Episode length: {env_config.episode_length}")
    print(f"  Modulacoes: {env_config.modulation_names}")
    print(f"  Defragmentacao: {env_config.defragmentation}")
    
    # Cria diretorios
    run_dir = Path(training_config.base_dir).resolve() / training_config.experiment_name
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir = run_dir / "checkpoints"
    
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDiretorios:")
    print(f"  TensorBoard: {tensorboard_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    
    # Cria ambientes
    print(f"\nCriando {training_config.n_envs} ambientes...")
    env = make_vec_env(env_config, n_envs=training_config.n_envs)
    print("Ambientes criados!")
    
    # Carrega hiperparametros PPO
    print(f"\nCarregando hiperparametros PPO (profile: {training_config.ppo_profile})...")
    ppo_hyperparams = get_hyperparams(training_config.ppo_profile)
    ppo_dict = ppo_hyperparams.to_dict()
    
    # Ajusta tensorboard_log
    ppo_dict['tensorboard_log'] = str(tensorboard_dir)
    
    print(f"  Learning rate: {ppo_dict['learning_rate']}")
    print(f"  N steps: {ppo_dict['n_steps']}")
    print(f"  Batch size: {ppo_dict['batch_size']}")
    
    # Cria modelo PPO
    print("\nCriando modelo MaskablePPO...")
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        **ppo_dict
    )
    print("Modelo criado!")
    
    # Configura callbacks reaproveitando modulo central
    print(f"\nConfigurando callbacks...")
    base_callbacks = create_callbacks(
        save_path=str(checkpoint_dir),
        checkpoint_freq=2000,
        log_freq=training_config.log_freq,
        verbose=training_config.verbose
    )
    base_callbacks.append(
        StopAfterEpisodesCallback(
            n_episodes=training_config.n_episodes,
            verbose=training_config.verbose
        )
    )
    callback_list = CallbackList(base_callbacks)
    print(f"Callbacks configurados via create_callbacks + StopAfterEpisodes")
    
    # Treina
    print("\n" + "="*80)
    print(" INICIANDO TREINAMENTO")
    print("="*80 + "\n")
    
    # Usa um numero muito grande de timesteps, o callback vai parar no n_episodes
    max_timesteps = 100_000_000
    
    try:
        model.learn(
            total_timesteps=max_timesteps,
            callback=callback_list,
            progress_bar=False,
            tb_log_name=training_config.experiment_name,
        )
    except KeyboardInterrupt:
        print("\n\n[WARNING] Treinamento interrompido pelo usuario")
    
    # Salva modelo final
    final_model_path = checkpoint_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\n[OK] Modelo salvo: {final_model_path}.zip")
    
    # Fecha ambientes
    env.close()
    
    print("\n[OK] Treinamento concluido!")
    
    return model, run_dir


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # Garante compatibilidade com Windows e scripts reaproveitados
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    mp.freeze_support()
    
    print("\n" + "="*80)
    print(" CONFIGURACAO DO TREINAMENTO")
    print("="*80)
    
    # Configuracoes (MODIFIQUE AQUI conforme necessario)
    env_cfg = EnvConfig(
        topology_name="nobel-eu",
        modulation_names="BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM",
        modulations_to_consider=3,  
        num_spectrum_resources=320,
        load=350, 
        episode_length=1000, 
        k_paths=3,
        defragmentation=False,
        seed=10
    )
    
    train_cfg = TrainingConfig(
        n_episodes=100_000,  # 🔥 TREINO COMPLETO visando fine-tuning prolongado
        n_envs=44,  # 44 ambientes paralelos (máxima paralelização)
        ppo_profile="intensive_v2_refined",  # 🔥 Perfil otimizado: LR 3e-4 (linear), batch 1024, epochs 10
        base_dir="/home/talles/projects/optical-networking-gym/examples/OFC_2025/training_runs",
        experiment_name=f"qrmsa_intensive_v2_refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        log_freq=5000,  # Log a cada 5k steps
        verbose=0  # Silencioso para rodar em background
    )
    
    print("\n" + "="*80)
    print(" TREINAMENTO COMPLETO: INTENSIVE_V2_REFINED (75k episódios)")
    print("="*80)
    print(f"\n   Experimento: {train_cfg.experiment_name}")
    print(f"   🎯 Meta: {train_cfg.n_episodes:,} episódios")
    print(f"   🔧 PPO Profile: {train_cfg.ppo_profile} (OTIMIZADO)")
    print(f"   🌐 Topologia: {env_cfg.topology_name}")
    print(f"   📊 Load: {env_cfg.load} Erlangs")
    print(f"   📏 Episode length: {env_cfg.episode_length} steps")
    print(f"   🚀 Ambientes paralelos: {train_cfg.n_envs}")
    print(f"   📡 Modulações: {env_cfg.modulations_to_consider}/6 ativas")
    print(f"   📁 Diretório: {train_cfg.base_dir}")
    print(f"   📝 Callbacks: BlockingRate, Fragmentation, RewardStats, StopAfterEpisodes")
    print(f"\n   💡 INTENSIVE_V2_REFINED: LR linear 3e-4→0, Batch 1024, Epochs 10, clip_range_vf 0.2")
    print(f"   ⏱️  Tempo estimado: ~25-35 horas com 44 envs")
    print("="*80 + "\n")
    
    # Executa treinamento
    model, run_dir = train(env_cfg, train_cfg)
    
    print(f"\n[DONE] Experimento salvo em: {run_dir}")
