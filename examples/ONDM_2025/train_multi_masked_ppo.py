from datetime import datetime
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import gymnasium as gym
import torch as th

# Maskable PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

# Stable Baselines callbacks/tools
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Masked wrapper
from masked_wrapper import ActionMaskWrapper

# optical_networking_gym
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.topology import Modulation, get_topology

from typing import Tuple

# 1. Define Modulations
def define_modulations() -> Tuple[Modulation, ...]:
    return (
        Modulation(name="BPSK",  maximum_length=100000, spectral_efficiency=1,  minimum_osnr=12.6, inband_xt=-14),
        Modulation(name="QPSK",  maximum_length=2000,   spectral_efficiency=2,  minimum_osnr=12.6, inband_xt=-17),
        Modulation(name="8QAM",  maximum_length=1000,   spectral_efficiency=3,  minimum_osnr=18.6, inband_xt=-20),
        Modulation(name="16QAM", maximum_length=500,    spectral_efficiency=4,  minimum_osnr=22.4, inband_xt=-23),
        Modulation(name="32QAM", maximum_length=250,    spectral_efficiency=5,  minimum_osnr=26.4, inband_xt=-26),
        Modulation(name="64QAM", maximum_length=125,    spectral_efficiency=6,  minimum_osnr=30.4, inband_xt=-29),
    )

# 2. Custom Callbacks
class CSVLoggerCallback(BaseCallback):
    """
    Logs episode info to a CSV file.
    """
    def __init__(self, csv_path: str = f"./episode_logs/episode_info_.csv", verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.csv_file = None
        self.csv_writer = None
        self.file_initialized = False

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.csv_file = open(self.csv_path, mode='w', newline='', encoding='utf-8')
        self.csv_writer = None
        self.file_initialized = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done:
                episode_info = info.get("episode", {})
                row_dict = {
                    "episode_reward": episode_info.get("r", 0),
                    "episode_length": episode_info.get("l", 0),
                    "bit_rate_blocking_rate": info.get("episode_bit_rate_blocking_rate", 0),
                    "service_blocking_rate": info.get("episode_service_blocking_rate", 0),
                    "modulation_1.0": info.get("modulation_1.0", 0),
                    "modulation_2.0": info.get("modulation_2.0", 0),
                    "modulation_3.0": info.get("modulation_3.0", 0),
                    "modulation_4.0": info.get("modulation_4.0", 0),
                    "modulation_5.0": info.get("modulation_5.0", 0),
                    "modulation_6.0": info.get("modulation_6.0", 0),               
                }
                if not self.file_initialized:
                    import csv
                    headers = list(row_dict.keys())
                    self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=headers)
                    self.csv_writer.writeheader()
                    self.file_initialized = True
                self.csv_writer.writerow(row_dict)
                self.csv_file.flush()
        return True

    def _on_training_end(self):
        if self.csv_file:
            self.csv_file.close()


class TensorBoardInfoLoggerCallback(BaseCallback):
    """
    Logs custom episode info to TensorBoard using SB3's logger.
    """
    def __init__(self, verbose=0):
        super(TensorBoardInfoLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.custom_metrics = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done:
                ep_info = info.get("episode", {})
                ep_reward = ep_info.get("r", 0)
                ep_length = ep_info.get("l", 0)
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Armazenar métricas personalizadas
                bit_rate_blocking_rate = info.get("episode_bit_rate_blocking_rate")
                service_blocking_rate = info.get("episode_service_blocking_rate")

                if bit_rate_blocking_rate is not None:
                    if "episode_bit_rate_blocking_rate" not in self.custom_metrics:
                        self.custom_metrics["episode_bit_rate_blocking_rate"] = []
                    self.custom_metrics["episode_bit_rate_blocking_rate"].append(bit_rate_blocking_rate)

                if service_blocking_rate is not None:
                    if "episode_service_blocking_rate" not in self.custom_metrics:
                        self.custom_metrics["episode_service_blocking_rate"] = []
                    self.custom_metrics["episode_service_blocking_rate"].append(service_blocking_rate)

                # Calcular médias das últimas 100 episódios
                if len(self.episode_rewards) > 0:
                    mean_reward_100 = np.mean(self.episode_rewards[-100:])
                    mean_length_100 = np.mean(self.episode_lengths[-100:])
                    self.logger.record("metrics/mean_reward_100", mean_reward_100)
                    self.logger.record("metrics/mean_length_100", mean_length_100)

                # Logar métricas personalizadas (média das últimas 100)
                for metric_key, metric_values in self.custom_metrics.items():
                    if len(metric_values) > 0:
                        mean_val_100 = np.mean(metric_values[-100:])
                        self.logger.record(f"metrics/{metric_key}", mean_val_100)

        return True

    def _on_training_end(self):
        pass


class EntropyCoefficientScheduler(BaseCallback):
    """
    Linearly adjusts ent_coef from an initial to a final value.
    """
    def __init__(self, initial_ent_coef, final_ent_coef, schedule_timesteps, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.schedule_timesteps = schedule_timesteps

    def _on_step(self) -> bool:
        fraction = min(1.0, self.model.num_timesteps / self.schedule_timesteps)
        current_ent_coef = self.initial_ent_coef + fraction * (self.final_ent_coef - self.initial_ent_coef)
        self.model.ent_coef = current_ent_coef
        if self.verbose > 0:
            self.logger.record("entropy_coefficient", self.model.ent_coef)
        return True


class StopTrainingOnEpisodesCallback(BaseCallback):
    """
    Stops training after a certain number of episodes.
    """
    def __init__(self, max_episodes: int, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.n_episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.n_episodes += np.sum(dones)
            if self.verbose > 0 and np.sum(dones) > 0:
                print(f"Episodes completed: {self.n_episodes}/{self.max_episodes}")
        if self.n_episodes >= self.max_episodes:
            print(f"Stopping training at {self.n_episodes} episodes.")
            return False
        return True


class ExplorationBoostCallback(BaseCallback):
    """
    If the mean reward (over 'check_interval' episodes) stalls by less than 'threshold',
    increase ent_coef by 10%.
    """
    def __init__(self, check_interval=15, threshold=0.01, verbose=0):
        super().__init__(verbose)
        self.check_interval = check_interval
        self.threshold = threshold
        self.episode_rewards = []
        self.last_mean = None

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done and "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        if len(self.episode_rewards) >= self.check_interval:
            current_mean = np.mean(self.episode_rewards[-self.check_interval:])
            if self.last_mean is not None:
                if abs(current_mean - self.last_mean) < self.threshold:
                    new_ent_coef = self.model.ent_coef * 1.1
                    self.model.ent_coef = new_ent_coef
                    if self.verbose > 0:
                        print(
                            f"[ExplorationBoostCallback] Reward mean stalled "
                            f"(from {self.last_mean:.2f} to {current_mean:.2f}). "
                            f"New ent_coef = {self.model.ent_coef:.4f}."
                        )
            self.last_mean = current_mean

        return True


# 3. Helper functions for env creation
def make_env(env_id, rank, seed, env_args):
    """
    Creates an environment instance.
    """
    def _init():
        env = gym.make(env_id, **env_args)
        env.reset(seed=seed + rank)
        env = ActionMaskWrapper(env)
        env = Monitor(env)
        return env
    return _init


def make_test_env(env_id, seed, env_args):
    """
    Creates a test environment instance.
    """
    def _init():
        env = gym.make(env_id, **env_args)
        env.reset(seed=seed)
        env = ActionMaskWrapper(env)
        env = Monitor(env)
        return env
    return _init


# 4. Main training function
def main():
    # (1) Modulations
    cur_modulations = define_modulations()

    # (2) Topology
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

    # Plot the topology (optional)
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(topology, k=0.5, seed=42)
    nx.draw(topology, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=3000)
    edge_labels = nx.get_edge_attributes(topology, "length")
    nx.draw_networkx_edge_labels(topology, pos, edge_labels=edge_labels, font_color="red")
    plt.title(str(topology.name).upper())
    plt.axis("off")
    plt.show()

    # (3) Setup
    seed = 10
    random.seed(seed)
    np.random.seed(seed)

    episode_length = 1_000
    load = 210
    launch_power = 0

    num_slots = 320
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
        file_name="",
        measure_disruptions=False,
        k_paths=5,
    )

    # (4) Vectorized Environment
    num_envs = 4
    env_id = "QRMSAEnvWrapper-v0"
    vec_env = SubprocVecEnv([make_env(env_id, i, seed, env_args) for i in range(num_envs)])

    # (5) Callbacks
    eval_callback = MaskableEvalCallback(
        eval_env=vec_env,
        best_model_save_path="./logs_best_model/",
        log_path="./logs_eval/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    ent_scheduler = EntropyCoefficientScheduler(
        initial_ent_coef=0.03,
        final_ent_coef=0.01,
        schedule_timesteps=1000000,
        verbose=1
    )
    stop_training_callback = StopTrainingOnEpisodesCallback(max_episodes=10000, verbose=1)
    exploration_boost_callback = ExplorationBoostCallback(check_interval=15, threshold=0.01, verbose=1)

    def linear_schedule(initial_value: float):
        def func(progress_remaining: float):
            return progress_remaining * initial_value
        return func

    # (6) Create and train the model
    policy_kwargs = dict(
                     net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]))
    
    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=linear_schedule(3e-4),
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.03,  # will be updated by the scheduler
        clip_range=0.2,
        verbose=1,
        seed=42,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_masked_tensorboard/"
    )
    
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"./episode_logs/episode_info_{current_date_time}.csv"
    csv_logger_callback = CSVLoggerCallback(csv_path=filename, verbose=1)
    tensorboard_info_callback = TensorBoardInfoLoggerCallback(verbose=1)  

    model.learn(
        total_timesteps=int(9e200),
        callback=[
            stop_training_callback,
            eval_callback,
            ent_scheduler,
            exploration_boost_callback,
            csv_logger_callback,
            tensorboard_info_callback
        ]
    )

    model.save("ppo_masked_final.zip")
    print("Training finished. Model saved as 'ppo_masked_final.zip'.")

    # (7) Optional test
    # test_env = DummyVecEnv([make_test_env(env_id, seed, env_args)])
    # model = MaskablePPO.load("ppo_masked_final.zip", env=test_env)
    # obs = test_env.reset()
    # num_test_episodes = 2
    # for ep in range(num_test_episodes):
    #     done = [False]
    #     total_r = 0.0
    #     while not done[0]:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done, truncated, info = test_env.step(action)
    #         total_r += reward[0]
    #     print(f"Episode {ep+1} reward: {total_r}")
    #     obs = test_env.reset()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Needed on Windows
    main()
