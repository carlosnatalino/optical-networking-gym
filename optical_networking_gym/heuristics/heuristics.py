from typing import Optional, Tuple

import numpy as np
from optical_networking_gym.topology import Modulation, Path
from optical_networking_gym.utils import rle
from optical_networking_gym.core.osnr import calculate_osnr
from optical_networking_gym.envs.qrmsa import QRMSAEnv



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
