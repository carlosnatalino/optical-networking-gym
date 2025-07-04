import os
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from typing import Tuple
import gymnasium as gym

# sb3_contrib: Maskable PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

# Callbacks e ferramentas do Stable Baselines
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Para TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Imports do optical_networking_gym
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.topology import Modulation, get_topology

def define_modulations() -> Tuple[Modulation, ...]:
    return (
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

cur_modulations = define_modulations()

topology_name = "nobel-eu"
topology_path = r"C:\Users\talle\Documents\Mestrado\optical-networking-gym\examples\topologies\nobel-eu.xml"

topology = get_topology(
    topology_path,
    topology_name,
    cur_modulations,
    80,
    0.2,
    4.5,
    5
)

plt.figure(figsize=(15, 10))
pos = nx.spring_layout(topology, k=0.5, seed=42)
nx.draw(topology, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=3000)
edge_labels = nx.get_edge_attributes(topology, "length")
nx.draw_networkx_edge_labels(topology, pos, edge_labels=edge_labels, font_color="red")
plt.title(str(topology.name).upper())
plt.axis("off")
plt.show()

seed = 10
random.seed(seed)

episode_length = 10000
load = 210
launch_power = 0

num_slots = 10
frequency_slot_bandwidth = 12.5e9
frequency_start = 3e8 / 1565e-9
bandwidth = num_slots * frequency_slot_bandwidth
bit_rates = (10, 40, 80, 100, 400)

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
    file_name="./results/PPO_1",
    measure_disruptions=False,
    k_paths=2,
)


class ActionMaskWrapper(gym.Wrapper):
    """
    Extrai a máscara (info["mask"]) e disponibiliza via .action_masks().
    """
    def __init__(self, env):
        super().__init__(env)
        self._last_mask = None

    def reset(self, **kwargs):
        # Aqui, a env deve retornar (obs, info) estilo gymnasium
        obs, info = self.env.reset(**kwargs)
        # Salvar a máscara se existir
        self._last_mask = info.get("mask", None)
        # Retornamos APENAS obs para o SB3
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._last_mask = info.get("mask", None)
        return obs, reward, done, truncated, info

    def action_masks(self):
        # Chamado pelo MaskablePPO
        return self._last_mask


def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func


class EntropyCoefficientScheduler(BaseCallback):
    def __init__(self, initial_ent_coef, final_ent_coef, schedule_timesteps, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.schedule_timesteps = schedule_timesteps

    def _on_step(self) -> bool:
        fraction = min(1.0, self.model.num_timesteps / self.schedule_timesteps)
        current_ent_coef = self.initial_ent_coef + fraction * (self.final_ent_coef - self.initial_ent_coef)
        self.model.ent_coef = current_ent_coef
        return True


class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos and isinstance(infos[0], dict):
            maybe_info = infos[0]
            # Se seu ambiente preenche algo como "request_blocking_rate" no info
            if "request_blocking_rate" in maybe_info:
                rbr = maybe_info["request_blocking_rate"]
                self.logger.record("metrics/request_blocking_rate", rbr)
        return True


def make_env():
    base_env = gym.make("QRMSAEnvWrapper-v0", **env_args)
    # Aplica o ActionMaskWrapper antes do Monitor
    base_env = ActionMaskWrapper(base_env)
    base_env = Monitor(base_env)
    return base_env


vec_env = DummyVecEnv([make_env])

maskable_eval_callback = MaskableEvalCallback(
    eval_env=vec_env,
    best_model_save_path="./logs_best_model/",
    log_path="./logs_eval/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

ent_scheduler = EntropyCoefficientScheduler(
    initial_ent_coef=0.05,
    final_ent_coef=0.01,
    schedule_timesteps=200_000
)

tb_logger_callback = CustomTensorboardCallback()

policy_kwargs = dict(net_arch=[256, 256, 128])

model = MaskablePPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=linear_schedule(3e-4),
    n_steps=2048,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.05,  # Será ajustado dinamicamente pelo ent_scheduler
    clip_range=0.2,
    verbose=1,
    seed=42,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./ppo_masked_tensorboard/"
)

model.learn(
    total_timesteps=300_000,
    callback=[maskable_eval_callback, ent_scheduler, tb_logger_callback]
)

model.save("ppo_masked_final.zip")
print("Treinamento concluído e modelo salvo em 'ppo_masked_final.zip'.")

# Testar o agente treinado
model = MaskablePPO.load("ppo_masked_final.zip", env=vec_env)
obs = vec_env.reset()

for ep in range(2):
    done = [False]
    total_r = 0.0
    while not done[0]:
        # Extração manual da máscara via wrapper
        # A DummyVecEnv retorna obs como [obs], info como [info]
        # mas neste caso, a wrapper populou self._last_mask internamente
        # Podemos usar .predict(obs, action_masks=mask) ou deixar que o PPO internamente chame env.action_masks()
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = vec_env.step(action)
        total_r += reward[0]
    print(f"Episódio {ep+1} finalizado, recompensa total = {total_r}")
    obs = vec_env.reset()