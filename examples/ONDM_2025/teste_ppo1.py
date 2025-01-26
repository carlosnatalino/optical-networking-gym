import csv
from datetime import datetime
import os
import random
import time
import numpy as np
import gymnasium as gym
import torch as th

# Maskable PPO (SB3-Contrib)
from sb3_contrib import MaskablePPO

# Stable Baselines3 ferramentas
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Wrapper para mascarar ações
from masked_wrapper import ActionMaskWrapper

# optical_networking_gym
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.topology import Modulation, get_topology

from typing import Tuple


# =========================================================
# 1. Definição das Modulações
# =========================================================
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


# =========================================================
# 2. Configuração do Ambiente (função dedicada)
# =========================================================
def create_environment_config():
    """
    Cria e retorna:
      - objeto de topologia,
      - dicionário de argumentos (env_args),
      - semente (seed),
      - número de ambientes paralelos (num_envs).
    """
    # Topologia e modulações
    cur_modulations = define_modulations()
    topology_name = "nobel-eu"
    topology_path = r"C:\Users\talle\Documents\Mestrado\optical-networking-gym\examples\topologies\nobel-eu.xml"
    topology = get_topology(
        topology_path,
        topology_name,
        cur_modulations,
        80,       # Número de canais? Ajuste conforme a topologia
        0.2,      # Fator de ruído (exemplo)
        4.5,      # Outra config. de ruído (exemplo)
        5         # Exemplo de config. adicional
    )

    # Parâmetros gerais do ambiente
    seed = 10
    random.seed(seed)
    np.random.seed(seed)

    episode_length = 1_000  # Cada episódio com 1000 steps
    load = 210
    launch_power = 0
    num_slots = 320
    frequency_slot_bandwidth = 12.5e9
    frequency_start = 3e8 / 1565e-9
    bandwidth = num_slots * frequency_slot_bandwidth
    bit_rates = (10, 40, 100, 400)

    env_args = dict(
        topology=topology,
        seed=seed,
        allow_rejection=True,
        load=load,
        episode_length=episode_length,
        num_spectrum_resources=num_slots,
        launch_power_dbm=launch_power,
        bandwidth=bandwidth,
        frequency_start=frequency_start,
        frequency_slot_bandwidth=frequency_slot_bandwidth,
        bit_rate_selection="discrete",
        bit_rates=bit_rates,
        margin=0,
        file_name="",
        measure_disruptions=False,
        k_paths=5,
    )

    num_envs = 4  # 14 threads/ambientes paralelos
    return topology, env_args, seed, num_envs


# =========================================================
# 3. Helper para criação de ambientes
# =========================================================
def make_env(env_id, rank, seed, env_args):
    """
    Função-fábrica que cria um ambiente de forma determinística (para SubprocVecEnv).
    """
    def _init():
        env = gym.make(env_id, **env_args)
        env.reset(seed=seed + rank)
        env = ActionMaskWrapper(env)
        env = Monitor(env)
        return env
    return _init


# =========================================================
# 4. Callback Único
# =========================================================
import csv
import os
import time
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback

class SingleCallback(BaseCallback):
    """
    Callback Único que abrange:
    - Log em CSV dos episódios (exceto mask),
    - Cálculo e registro do tempo por episódio,
    - Interrompe o treinamento após 'max_episodes',
    - Métricas personalizadas no TensorBoard,
    - Schedule do ent_coef,
    - Boost de exploração (se reward médio 'empacar').

    Observação: a cada step, múltiplos ambientes podem terminar simultaneamente
    (pois usamos VecEnv com vários ambientes).
    """

    def __init__(
        self, 
        max_episodes: int = 10_000,
        # Parâmetros do scheduler de ent_coef
        initial_ent_coef: float = 0.03,
        final_ent_coef: float = 0.01,
        schedule_timesteps: int = 1_500_000,
        # Parâmetros do ExplorationBoost
        check_interval: int = 15,
        threshold: float = 0.01,
        verbose: int = 0
    ):
        super().__init__(verbose)

        # 1) Parar treinamento após N episódios
        self.max_episodes = max_episodes
        self.num_episodes = 0

        # 2) Cálculo de tempo por episódio
        self.start_times = []

        # 3) Logging em CSV
        current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_path = f"./episode_logs/episode_info_{current_date_time}.csv"
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.fields_initialized = False
        self.header_fields = set()

        # 4) TensorBoard metrics
        self.episode_rewards = []
        self.episode_lengths = []
        # Armazenamento de métricas personalizadas (p. ex. blocking rate)
        self.custom_metrics = {}

        # 5) Scheduler do ent_coef
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.schedule_timesteps = schedule_timesteps

        # 6) Exploration Boost
        self.check_interval = check_interval
        self.threshold = threshold
        self.last_mean = None  # para comparar variação de reward
        # Reaproveitamos self.episode_rewards para esse callback

    def _on_training_start(self) -> None:
        """
        Chamado no início do treinamento.
        - Inicializa tempos para cada env,
        - Cria um CSV vazio para logs (e fecha em seguida).
        """
        num_envs = self.training_env.num_envs
        self.start_times = [time.time() for _ in range(num_envs)]

        # Garantir que o arquivo exista (cabeçalho ainda indefinido).
        with open(self.csv_path, mode='w', newline='', encoding='utf-8') as _f:
            pass

        if self.verbose > 0:
            print(f"[SingleCallback] Arquivo de log criado: {self.csv_path}")
            print(f"[SingleCallback] Treinamento iniciado com {num_envs} ambientes paralelos.")

    def _on_step(self) -> bool:
        """
        Chamado a cada step em cada ambiente. Precisamos:
        - Ver quais ambientes terminaram (done=True),
        - Registrar info no CSV (exceto mask),
        - Medir tempo de cada episódio,
        - Atualizar métricas para TensorBoard,
        - Aplicar schedule e boost de exploração do ent_coef,
        - Verificar se atingimos 'max_episodes'.
        """
        # 1) Pegamos 'dones' e 'infos' de todos os ambientes
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        num_envs = self.training_env.num_envs

        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            if done:
                # A) Contagem de episódios
                self.num_episodes += 1

                # B) Tempo de episódio
                end_time = time.time()
                exec_time = end_time - self.start_times[env_idx]
                self.start_times[env_idx] = end_time
                info["episode_time"] = round(exec_time, 3)

                # C) Extraímos as infos do episódio
                episode_info = info.get("episode", {})
                ep_reward = episode_info.get("r", 0.0)
                ep_length = episode_info.get("l", 0)
                
                # Armazena em arrays para estatísticas de TensorBoard
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Métricas personalizadas (ex: blocking rates)
                bit_rate_blocking_rate = info.get("episode_bit_rate_blocking_rate")
                service_blocking_rate = info.get("episode_service_blocking_rate")
                # Podemos adicionar qualquer outra info custom
                if bit_rate_blocking_rate is not None:
                    if "episode_bit_rate_blocking_rate" not in self.custom_metrics:
                        self.custom_metrics["episode_bit_rate_blocking_rate"] = []
                    self.custom_metrics["episode_bit_rate_blocking_rate"].append(bit_rate_blocking_rate)

                if service_blocking_rate is not None:
                    if "episode_service_blocking_rate" not in self.custom_metrics:
                        self.custom_metrics["episode_service_blocking_rate"] = []
                    self.custom_metrics["episode_service_blocking_rate"].append(service_blocking_rate)

                # D) Montar dicionário para CSV (exceto mask)
                row_dict = {}
                for k, v in info.items():
                    if k in ["mask", "service_blocking_rate", "bit_rate_blocking_rate", "disrupted_services","osnr", "osnr_req", "chosen_path_index", "chosen_slot", "TimeLimit.truncated","terminal_observation","episode","episode_bit_rate_blocking_rate","episode_service_blocking_rate"]:
                        continue
                    row_dict[k] = v

                # Garantir chaves 'episode_reward' e 'episode_length'
                row_dict["episode_reward"] = ep_reward
                row_dict["episode_length"] = ep_length

                # E) Atualizar e gravar no CSV
                if not self.fields_initialized:
                    self.header_fields = sorted(list(row_dict.keys()))
                    self.fields_initialized = True

                with open(self.csv_path, mode='a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=self.header_fields)
                    # Se for a primeira linha efetiva, escrevemos cabeçalho
                    if csv_file.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row_dict)

                if self.verbose > 0:
                    print(f"[SingleCallback] Episódio {self.num_episodes} finalizado (Ambiente {env_idx}). "
                          f"Tempo: {exec_time:.2f}s / Recompensa: {ep_reward:.2f}")

        # F) Atualizar ent_coef (Scheduler linear)
        fraction = min(1.0, self.model.num_timesteps / self.schedule_timesteps)
        current_ent_coef = self.initial_ent_coef + fraction * (self.final_ent_coef - self.initial_ent_coef)
        self.model.ent_coef = current_ent_coef
        # Podemos logar no TensorBoard
        self.logger.record("hyperparams/entropy_coefficient", self.model.ent_coef)

        # G) Boost de exploração caso reward médio 'empacar'
        # Só faz sentido se tivermos episódios suficientes
        if len(self.episode_rewards) >= self.check_interval:
            current_mean = np.mean(self.episode_rewards[-self.check_interval:])
            if self.last_mean is not None:
                if abs(current_mean - self.last_mean) < self.threshold:
                    # Aumenta ent_coef em 10%
                    self.model.ent_coef *= 1.1
                    if self.verbose > 0:
                        print(f"[SingleCallback][ExplorationBoost] Recompensa estagnada de {self.last_mean:.2f} "
                              f"para {current_mean:.2f}. Novo ent_coef = {self.model.ent_coef:.4f}")
                    # Podemos registrar também
                    self.logger.record("hyperparams/ent_coef_boost", self.model.ent_coef)
            self.last_mean = current_mean

        # H) Registrar estatísticas no TensorBoard (médias das últimas 100)
        if len(self.episode_rewards) > 0:
            mean_reward_100 = np.mean(self.episode_rewards[-100:])
            mean_length_100 = np.mean(self.episode_lengths[-100:])
            self.logger.record("metrics/mean_reward_100", mean_reward_100)
            self.logger.record("metrics/mean_length_100", mean_length_100)

        for metric_key, metric_values in self.custom_metrics.items():
            if len(metric_values) > 0:
                mean_val_100 = np.mean(metric_values[-100:])
                self.logger.record(f"metrics/{metric_key}", mean_val_100)

        # I) Verificar se atingimos o número máximo de episódios
        if self.num_episodes >= self.max_episodes:
            print(f"[SingleCallback] Atingido {self.num_episodes} episódios. Encerrando...")
            return False

        return True

    def _on_training_end(self) -> None:
        """
        Chamado ao final do treinamento.
        """
        if self.verbose > 0:
            print(f"[SingleCallback] Treinamento encerrado. Total de episódios: {self.num_episodes}.")



# =========================================================
# 5. Função principal de treinamento
# =========================================================
def main():
    # (A) Carregar/Configurar o ambiente
    topology, env_args, seed, num_envs = create_environment_config()
    env_id = "QRMSAEnvWrapper-v0"

    # Cria vetorização em Subprocessos (14 threads)
    vec_env = SubprocVecEnv([make_env(env_id, i, seed, env_args) for i in range(num_envs)])

    # (B) Definir política, rede e parâmetros de PPO com base em experiência
    # Otimizados para 10.000 episódios de 1000 steps cada => 10 milhões de steps
    # Sugerimos LR menor, batch_size robusto, etc.
    def linear_schedule(initial_value: float):
        def func(progress_remaining: float):
            return progress_remaining * initial_value
        return func

    policy_kwargs = dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]))

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=linear_schedule(1e-4),  # LR decai linearmente até 0
        n_steps=128,
        batch_size=32,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        seed=42,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_masked_tensorboard/"  # Se quiser log no TensorBoard
    )

    # (C) Criar callback único
    single_callback = SingleCallback(
        max_episodes=10_000,  # Queremos 10.000 episódios
        verbose=1
    )

    # (D) Treinamento
    # Passamos total_timesteps grande para não limitar por timesteps,
    # pois iremos parar pelos 10.000 episódios via callback.
    model.learn(
        total_timesteps=int(1e12),
        callback=single_callback
    )

    # (E) Salvar modelo final
    model.save("ppo_masked_final.zip")
    print("Treinamento finalizado. Modelo salvo em 'ppo_masked_final.zip'.")

    # (F) (Opcional) Teste rápido
    # from stable_baselines3.common.vec_env import DummyVecEnv
    # def make_test_env(env_id, seed, env_args):
    #     def _init():
    #         env = gym.make(env_id, **env_args)
    #         env.reset(seed=seed)
    #         env = ActionMaskWrapper(env)
    #         env = Monitor(env)
    #         return env
    #     return _init
    #
    # test_env = DummyVecEnv([make_test_env(env_id, seed, env_args)])
    # model = MaskablePPO.load("ppo_masked_final.zip", env=test_env)
    # obs = test_env.reset()
    # num_test_episodes = 2
    # for ep in range(num_test_episodes):
    #     done = False
    #     total_r = 0.0
    #     while not done:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done, truncated, info = test_env.step(action)
    #         done = done[0]
    #         total_r += reward[0]
    #     print(f"Episode {ep+1} reward: {total_r}")
    #     obs = test_env.reset()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    multiprocessing.freeze_support()
    main()
