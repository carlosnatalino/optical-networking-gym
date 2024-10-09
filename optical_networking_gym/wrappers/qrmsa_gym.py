import time
import os
from typing import Optional, Tuple

from datetime import datetime
from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from optical_networking_gym.envs.qrmsa import QRMSAEnv

from optical_networking_gym.topology import Modulation, Path
from optical_networking_gym.utils import rle
from optical_networking_gym.core.osnr import calculate_osnr


class QRMSAEnvWrapper(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        super(QRMSAEnvWrapper, self).__init__()
        self.env = QRMSAEnv(*args, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self._np_random, _ = seeding.np_random(None)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            self.env.input_seed = seed

        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        print(action)
        return self.env.step(action)

    
    def render(self, mode='human'):
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
        else:
            pass

    def close(self):
        if hasattr(self.env, 'close'):
            return self.env.close()
        else:
            pass

    def get_spectrum_use_services(self):
        return self.env.get_spectrum_use_services()
    

def run_wrapper(args):
    print(args)
    return run_environment(*args)

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
) -> None:

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
    )

    fn_heuristic = None

    if heuristic == 1:
        fn_heuristic = shortest_available_path_first_fit_best_modulation
    elif heuristic == 2:
        fn_heuristic = shortest_available_path_lowest_spectrum_best_modulation
    elif heuristic == 3:
        fn_heuristic = best_modulation_load_balancing
    elif heuristic == 4:
        fn_heuristic = load_balancing_best_modulation
    else:
        raise ValueError(f"Heuristic index `{heuristic}` is not found!")

    env = QRMSAEnvWrapper(**env_args)

    env.reset()

    if monitor_file_name is None:
        raise ValueError("Missing monitor file name")

    monitor_final_name = "_".join([monitor_file_name, topology.name, str(env.env.launch_power_dbm), str(env.env.load) + ".csv"])
    # if os.path.exists(monitor_final_name):
    #     raise ValueError(f"File `{monitor_final_name}` already exists!")
    file_handler = open(monitor_final_name, "wt", encoding="UTF-8")
    file_handler.write(f"# Date: {datetime.now()}\n")
    file_handler.write("episode,service_blocking_rate,episode_service_blocking_rate,bit_rate_blocking_rate,episode_bit_rate_blocking_rate")
    for mf in env.env.modulations:
        file_handler.write(f",modulation_{mf.spectral_efficiency}")
    file_handler.write(",episode_disrupted_services,episode_time\n")

    for ep in range(n_eval_episodes):
        env.reset(options={"only_episode_counters": True})

        # initialization
        done = False
        start_time = time.time()

        while not done:
            action = fn_heuristic(env.env)
            
            _, _, done, _, info = env.step(action)
        end_time = time.time()
        print(info)
        file_handler.write(f"{ep},{info['service_blocking_rate']},{info['episode_service_blocking_rate']},{info['bit_rate_blocking_rate']},{info['episode_bit_rate_blocking_rate']}")
        for mf in env.env.modulations:
            file_handler.write(f",{info[f'modulation_{mf.spectral_efficiency}']}")
        file_handler.write(f",{info['episode_disrupted_services']},{(end_time - start_time):.2f}\n")
    file_handler.close()

    return
def shortest_available_path_first_fit_best_modulation(
    env: QRMSAEnv,
):
    path: Path | None = None
    modulation: Modulation | None = None

    for idp, path in enumerate(env.k_shortest_paths[
        env.current_service.source,
        env.current_service.destination,
    ]):
        available_slots = env.get_available_slots(path)
        for idm, modulation in zip(range(len(env.modulations) - 1, -1, -1), reversed(env.modulations)):  # from the best to the worst
            # TODO: improve the logic to only require one extra slot
            number_slots = env.get_number_slots(env.current_service, modulation) + 2

            initial_indices, values, lengths = rle(available_slots)
            
            sufficient_indices = np.where(lengths >= number_slots + 2)

            available_indices = np.where(values == 1)

            final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.shape[0] > 0:  # there are available slots

                env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]  # first fit
                if initial_slot > 0:
                    initial_slot += 1  # guard band

                # calculate OSNR
                env.current_service.path = path
                env.current_service.initial_slot = initial_slot
                env.current_service.number_slots = number_slots
                env.current_service.center_frequency = env.frequency_start + \
                    (env.frequency_slot_bandwidth * initial_slot) + \
                    (env.frequency_slot_bandwidth * (number_slots / 2))
                env.current_service.bandwidth = env.frequency_slot_bandwidth * number_slots
                env.current_service.launch_power = env.launch_power

                osnr, _, _ = calculate_osnr(env, env.current_service)  # correct parameters, ground truth
                if osnr >= modulation.minimum_osnr + env.margin:
                    return np.array([idp, idm, initial_slot])
    return None

def shortest_available_path_lowest_spectrum_best_modulation(
    env: QRMSAEnv,
) -> Optional[Tuple[int, int, int]]:
    """
    Selects the shortest available path with the lowest spectrum using the best modulation.

    Args:
        env (QRMSAEnv): The environment instance.

    Returns:
        Optional[Tuple[int, int, int]]: A tuple containing (path_index, modulation_index, initial_slot),
                                        or None if no suitable path is found.
    """
    for idp, path in enumerate(env.k_shortest_paths[
        env.current_service.source,
        env.current_service.destination,
    ]):
        available_slots = env.get_available_slots(path)
        for idm, modulation in zip(
            range(len(env.modulations) - 1, -1, -1),
            reversed(env.modulations)
        ):
            number_slots = env.get_number_slots(env.current_service, modulation) + 2

            initial_indices, values, lengths = rle(available_slots)
            sufficient_indices = np.where(lengths >= number_slots + 2)
            available_indices = np.where(values == 1)
            final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.size > 0:
                env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]
                if initial_slot > 0:
                    initial_slot += 1  # guard band

                # Update service parameters
                env.current_service.path = path
                env.current_service.initial_slot = initial_slot
                env.current_service.number_slots = number_slots
                env.current_service.center_frequency = (
                    env.frequency_start +
                    (env.frequency_slot_bandwidth * initial_slot) +
                    (env.frequency_slot_bandwidth * (number_slots / 2))
                )
                env.current_service.bandwidth = env.frequency_slot_bandwidth * number_slots
                env.current_service.launch_power = env.launch_power

                # Calculate OSNR
                osnr, _, _ = calculate_osnr(env, env.current_service)
                if osnr >= modulation.minimum_osnr + env.margin:
                    return np.array([idp, idm, initial_slot])

    return None


def best_modulation_load_balancing(
    env: QRMSAEnv,
) -> Optional[Tuple[int, int, int]]:
    """
    Balances the load by selecting the best modulation and minimizing the load on the path.

    Args:
        env (QRMSAEnv): The environment instance.

    Returns:
        Optional[Tuple[int, int, int]]: A tuple containing (path_index, modulation_index, initial_slot),
                                        or None if no suitable path is found.
    """
    solution = None
    lowest_load = float('inf')

    for idm, modulation in zip(
        range(len(env.modulations) - 1, -1, -1),
        reversed(env.modulations)
    ):
        for idp, path in enumerate(env.k_shortest_paths[
            env.current_service.source,
            env.current_service.destination,
        ]):
            available_slots = env.get_available_slots(path)
            number_slots = env.get_number_slots(env.current_service, modulation) + 2

            initial_indices, values, lengths = rle(available_slots)
            sufficient_indices = np.where(lengths >= number_slots + 2)
            available_indices = np.where(values == 1)
            final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.size > 0:
                env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]
                if initial_slot > 0:
                    initial_slot += 1  # guard band

                # Update service parameters
                env.current_service.path = path
                env.current_service.initial_slot = initial_slot
                env.current_service.number_slots = number_slots
                env.current_service.center_frequency = (
                    env.frequency_start +
                    (env.frequency_slot_bandwidth * initial_slot) +
                    (env.frequency_slot_bandwidth * (number_slots / 2))
                )
                env.current_service.bandwidth = env.frequency_slot_bandwidth * number_slots
                env.current_service.launch_power = env.launch_power

                # Calculate OSNR
                osnr, _, _ = calculate_osnr(env, env.current_service)
                if osnr >= modulation.minimum_osnr + env.margin:
                    current_load = available_slots.sum() / np.sqrt(path.hops)
                    if current_load < lowest_load:
                        lowest_load = current_load
                        solution = np.array([idp, idm, initial_slot])

    return solution


def load_balancing_best_modulation(
    env: QRMSAEnv,
) -> Optional[Tuple[int, int, int]]:
    """
    Balances load by selecting the best modulation with the lowest load path.

    Args:
        env (QRMSAEnv): The environment instance.

    Returns:
        Optional[Tuple[int, int, int]]: A tuple containing (path_index, modulation_index, initial_slot),
                                        or None if no suitable path is found.
    """
    solution = None
    lowest_load = float('inf')

    for idp, path in enumerate(env.k_shortest_paths[
        env.current_service.source,
        env.current_service.destination,
    ]):
        available_slots = env.get_available_slots(path)
        current_load = available_slots.sum() / np.sqrt(path.hops)
        if current_load >= lowest_load:
            continue  # not a better path

        for idm, modulation in zip(
            range(len(env.modulations) - 1, -1, -1),
            reversed(env.modulations)
        ):
            number_slots = env.get_number_slots(env.current_service, modulation) + 2

            initial_indices, values, lengths = rle(available_slots)
            sufficient_indices = np.where(lengths >= number_slots + 2)
            available_indices = np.where(values == 1)
            final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.size > 0:
                env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]
                if initial_slot > 0:
                    initial_slot += 1  # guard band

                # Update service parameters
                env.current_service.path = path
                env.current_service.initial_slot = initial_slot
                env.current_service.number_slots = number_slots
                env.current_service.center_frequency = (
                    env.frequency_start +
                    (env.frequency_slot_bandwidth * initial_slot) +
                    (env.frequency_slot_bandwidth * (number_slots / 2))
                )
                env.current_service.bandwidth = env.frequency_slot_bandwidth * number_slots
                env.current_service.launch_power = env.launch_power

                # Calculate OSNR
                osnr, _, _ = calculate_osnr(env, env.current_service)
                if osnr >= modulation.minimum_osnr + env.margin and current_load < lowest_load:
                    lowest_load = current_load
                    solution = np.array([idp, idm, initial_slot])
                    break  # Move to the next path after finding a better modulation

    return solution
