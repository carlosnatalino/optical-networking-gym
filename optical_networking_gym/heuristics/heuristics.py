from typing import Optional
import numpy as np
from optical_networking_gym.topology import Modulation, Path
from optical_networking_gym.utils import rle
from optical_networking_gym.core.osnr import calculate_osnr
from optical_networking_gym.envs.qrmsa import QRMSAEnv
from gymnasium import Env

from gymnasium import Env
from optical_networking_gym.envs.qrmsa import QRMSAEnv

def get_qrmsa_env(env: Env) -> QRMSAEnv:
    """
    Percorre os wrappers do ambiente até encontrar a instância base de QRMSAEnv.

    Args:
        env (gym.Env): O ambiente potencialmente envolvido em múltiplos wrappers.

    Returns:
        QRMSAEnv: A instância base de QRMSAEnv.

    Raises:
        ValueError: Se QRMSAEnv não for encontrado na cadeia de wrappers.
    """
    while not isinstance(env, QRMSAEnv):
        if hasattr(env, 'env'):
            env = env.env
        else:
            raise ValueError("QRMSAEnv não foi encontrado na cadeia de wrappers do ambiente.")
    return env


def get_action_index(env: QRMSAEnv, path_index: int, modulation_index: int, initial_slot: int) -> int:
    """
    Converte (path_index, modulation_index, initial_slot) em um índice de ação inteiro.
    
    Args:
        env (QRMSAEnv): O ambiente QRMSAEnv.
        path_index (int): Índice da rota.
        modulation_index (int): Índice da modulação.
        initial_slot (int): Slot inicial para alocação.
    
    Returns:
        int: Índice da ação correspondente.
    """
    return path_index * len(env.modulations) * env.num_spectrum_resources + \
           modulation_index * env.num_spectrum_resources + \
           initial_slot

def shortest_available_path_first_fit_best_modulation(
    env: Env,
) -> Optional[int]:
    """
    Seleciona a rota mais curta disponível com a primeira alocação possível e a melhor modulação.
    
    Args:
        env (gym.Env): O ambiente potencialmente envolvido em wrappers.
    
    Returns:
        Optional[int]: Índice da ação correspondente, ou a ação de rejeição se permitido, ou None.
    """
    qrmsa_env: QRMSAEnv = get_qrmsa_env(env)  # Descompactar o ambiente
    bl_resource = True
    bl_osnr = False
    
    for idp, path in enumerate(qrmsa_env.k_shortest_paths[
        qrmsa_env.current_service.source,
        qrmsa_env.current_service.destination,
    ]):
        available_slots = qrmsa_env.get_available_slots(path)
        # print(f"Available slots: {available_slots}")
        # print(env.topology)
        for idm, modulation in zip(range(len(qrmsa_env.modulations) - 1, -1, -1), reversed(qrmsa_env.modulations)):  # da melhor para a pior
            number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation)# para guard band

            initial_indices, values, lengths = rle(available_slots)
            sufficient_indices = np.where(lengths >= (number_slots))
            available_indices = np.where(values == 1)
            final_indices = np.intersect1d(available_indices, sufficient_indices)
            if final_indices.size > 0:
                if initial_indices[final_indices][0] < env.num_spectrum_resources-1:
                    sufficient_indices = np.where(lengths >= (number_slots+1))
                    available_indices = np.where(values == 1)
                    final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.size > 0:  # há slots disponíveis
                bl_resource = False
                qrmsa_env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]  # first fit

                # Atualizar parâmetros do serviço
                qrmsa_env.current_service.path = path
                qrmsa_env.current_service.initial_slot = initial_slot
                qrmsa_env.current_service.number_slots = number_slots
                qrmsa_env.current_service.center_frequency = qrmsa_env.frequency_start + \
                    (qrmsa_env.frequency_slot_bandwidth * initial_slot) + \
                    (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
                qrmsa_env.current_service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
                qrmsa_env.current_service.launch_power = qrmsa_env.launch_power

                # Calcular OSNR
                # print("DEBUG HEURÍSTICA: Entrada para cálculo do OSNR")
                # print(f"Service ID: {qrmsa_env.current_service.service_id}")
                # print(f"Path: {[link.node1 + '-' + link.node2 for link in path.links]}")
                # print(f"Initial Slot: {initial_slot}")
                # print(f"Number of Slots: {number_slots}")
                # print(f"Launch Power: {qrmsa_env.current_service.launch_power}")
                # print(f"Bandwidth: {qrmsa_env.current_service.bandwidth}")
                # print(f"Center Frequency: {qrmsa_env.current_service.center_frequency}")
                # print(f"Topologia (serviços rodando):")
                # for link in path.links:
                #     running_services = qrmsa_env.topology[link.node1][link.node2]["running_services"]
                #     for svc in running_services:
                #         print(f"  - Service ID: {svc.service_id}, Center Frequency: {svc.center_frequency}, Bandwidth: {svc.bandwidth}, number_slots: {svc.number_slots}, initial_slot: {svc.initial_slot}")
                # print("Chamando calculate_osnr...")
                osnr, ase, nli = calculate_osnr(qrmsa_env, qrmsa_env.current_service)
                # print(f"DEBUG HEURÍSTICA: Saída do cálculo do OSNR")
                # print(f"OSNR calculado: {osnr}")
                # print(f"ASE calculado: {ase}")
                # print(f"NLI calculado: {nli}")
                if osnr >= modulation.minimum_osnr + qrmsa_env.margin:
                    bl_osnr = False
                    # Converter para índice de ação
                    action = get_action_index(qrmsa_env, idp, idm, initial_slot)
                    return action, bl_osnr, bl_resource
                else:
                    bl_osnr = True
    # Se nenhuma ação válida encontrada, retornar a ação de rejeição se permitido
    return qrmsa_env.reject_action, bl_osnr, bl_resource
    # if qrmsa_env.allow_rejection:
    #     return qrmsa_env.reject_action
    # else:
    #     return None  # ou uma ação padrão específica

def shortest_available_path_lowest_spectrum_best_modulation(
    env: Env,
) -> Optional[int]:
    """
    Seleciona a rota mais curta disponível com a menor utilização espectral e a melhor modulação.
    
    Args:
        env (gym.Env): O ambiente potencialmente envolvido em wrappers.
    
    Returns:
        Optional[int]: Índice da ação correspondente, ou a ação de rejeição se permitido, ou None.
    """
    qrmsa_env: QRMSAEnv = get_qrmsa_env(env)  # Descompactar o ambiente

    for idp, path in enumerate(qrmsa_env.k_shortest_paths[
        qrmsa_env.current_service.source,
        qrmsa_env.current_service.destination,
    ]):
        available_slots = qrmsa_env.get_available_slots(path)
        for idm, modulation in zip(
            range(len(qrmsa_env.modulations) - 1, -1, -1),
            reversed(qrmsa_env.modulations)
        ):
            number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation) + 2  # +2 para guard band

            initial_indices, values, lengths = rle(available_slots)
            sufficient_indices = np.where(lengths >= number_slots)
            available_indices = np.where(values == 1)
            final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.size > 0:
                qrmsa_env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]
                if initial_slot > 0:
                    initial_slot += 1  # guard band

                # Atualizar parâmetros do serviço
                qrmsa_env.current_service.path = path
                qrmsa_env.current_service.initial_slot = initial_slot
                qrmsa_env.current_service.number_slots = number_slots
                qrmsa_env.current_service.center_frequency = (
                    qrmsa_env.frequency_start +
                    (qrmsa_env.frequency_slot_bandwidth * initial_slot) +
                    (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
                )
                qrmsa_env.current_service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
                qrmsa_env.current_service.launch_power = qrmsa_env.launch_power

                # Calcular OSNR
                osnr, _, _ = calculate_osnr(qrmsa_env, qrmsa_env.current_service)
                if osnr >= modulation.minimum_osnr + qrmsa_env.margin:
                    # Converter para índice de ação
                    action = get_action_index(qrmsa_env, idp, idm, initial_slot)
                    return action

    # Se nenhuma ação válida encontrada, retornar a ação de rejeição se permitido
    if qrmsa_env.allow_rejection:
        return qrmsa_env.reject_action
    else:
        return None  # ou uma ação padrão específica

def best_modulation_load_balancing(
    env: Env,
) -> Optional[int]:
    """
    Balanceia a carga selecionando a melhor modulação e minimizando a carga na rota.
    
    Args:
        env (gym.Env): O ambiente potencialmente envolvido em wrappers.
    
    Returns:
        Optional[int]: Índice da ação correspondente, ou a ação de rejeição se permitido, ou None.
    """
    qrmsa_env: QRMSAEnv = get_qrmsa_env(env)  # Descompactar o ambiente
    solution = None
    lowest_load = float('inf')

    for idm, modulation in zip(
        range(len(qrmsa_env.modulations) - 1, -1, -1),
        reversed(qrmsa_env.modulations)
    ):
        for idp, path in enumerate(qrmsa_env.k_shortest_paths[
            qrmsa_env.current_service.source,
            qrmsa_env.current_service.destination,
        ]):
            available_slots = qrmsa_env.get_available_slots(path)
            number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation)  # +2 para guard band

            initial_indices, values, lengths = rle(available_slots)
            sufficient_indices = np.where(lengths >= number_slots+1)
            available_indices = np.where(values == 1)
            final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.size > 0:
                qrmsa_env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]

                # Atualizar parâmetros do serviço
                qrmsa_env.current_service.path = path
                qrmsa_env.current_service.initial_slot = initial_slot
                qrmsa_env.current_service.number_slots = number_slots
                qrmsa_env.current_service.center_frequency = qrmsa_env.frequency_start + \
                    (qrmsa_env.frequency_slot_bandwidth * initial_slot) + \
                    (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
                qrmsa_env.current_service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
                qrmsa_env.current_service.launch_power = qrmsa_env.launch_power

                # Calcular OSNR
                osnr, _, _ = calculate_osnr(qrmsa_env, qrmsa_env.current_service)
                if osnr >= modulation.minimum_osnr + qrmsa_env.margin:
                    # Converter para índice de ação
                    action = get_action_index(qrmsa_env, idp, idm, initial_slot)
                    return action

    # Se nenhuma ação válida encontrada, retornar a ação de rejeição se permitido
    return qrmsa_env.reject_action

def load_balancing_best_modulation(
    env: Env,
) -> Optional[int]:
    """
    Balanceia a carga selecionando a melhor modulação com a menor carga na rota.
    
    Args:
        env (gym.Env): O ambiente potencialmente envolvido em wrappers.
    
    Returns:
        Optional[int]: Índice da ação correspondente, ou a ação de rejeição se permitido, ou None.
    """
    qrmsa_env: QRMSAEnv = get_qrmsa_env(env)  # Descompactar o ambiente
    solution = None
    lowest_load = float('inf')

    for idp, path in enumerate(qrmsa_env.k_shortest_paths[
        qrmsa_env.current_service.source,
        qrmsa_env.current_service.destination,
    ]):
        available_slots = qrmsa_env.get_available_slots(path)
        current_load = available_slots.sum() / np.sqrt(len(path.links))
        if current_load >= lowest_load:
            continue  # não é uma rota melhor

        for idm, modulation in zip(
            range(len(qrmsa_env.modulations) - 1, -1, -1),
            reversed(qrmsa_env.modulations)
        ):
            number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation) + 2  # +2 para guard band

            initial_indices, values, lengths = rle(available_slots)
            sufficient_indices = np.where(lengths >= number_slots)
            available_indices = np.where(values == 1)
            final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.size > 0:
                qrmsa_env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]
                if initial_slot > 0:
                    initial_slot += 1  # guard band

                # Atualizar parâmetros do serviço
                qrmsa_env.current_service.path = path
                qrmsa_env.current_service.initial_slot = initial_slot
                qrmsa_env.current_service.number_slots = number_slots
                qrmsa_env.current_service.center_frequency = (
                    qrmsa_env.frequency_start +
                    (qrmsa_env.frequency_slot_bandwidth * initial_slot) +
                    (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
                )
                qrmsa_env.current_service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
                qrmsa_env.current_service.launch_power = qrmsa_env.launch_power

                # Calcular OSNR
                osnr, _, _ = calculate_osnr(qrmsa_env, qrmsa_env.current_service)
                if osnr >= modulation.minimum_osnr + qrmsa_env.margin and current_load < lowest_load:
                    lowest_load = current_load
                    solution = get_action_index(qrmsa_env, idp, idm, initial_slot)
                    break  # Mover para a próxima rota após encontrar uma modulação melhor

    # Retornar a melhor solução encontrada
    if solution is not None:
        return solution

    # Se nenhuma ação válida encontrada, retornar a ação de rejeição se permitido
    if qrmsa_env.allow_rejection:
        return qrmsa_env.reject_action
    else:
        return None  # ou uma ação padrão específica
