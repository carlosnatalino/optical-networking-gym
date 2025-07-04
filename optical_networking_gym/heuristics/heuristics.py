import math
from typing import Optional
import numpy as np
from optical_networking_gym.topology import Modulation, Path
from optical_networking_gym.utils import rle, link_shannon_entropy_, fragmentation_route_cuts, fragmentation_route_rss
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
        modulation_index (int): Índice absoluto da modulação.
        initial_slot (int): Slot inicial para alocação.
    
    Returns:
        int: Índice da ação correspondente.
    """
    # Converter o índice absoluto da modulação para o relativo
    relative_modulation_index = env.max_modulation_idx - modulation_index
    
    return (path_index * env.modulations_to_consider * env.num_spectrum_resources +
            relative_modulation_index * env.num_spectrum_resources +
            initial_slot)

# def decimal_to_array(env: QRMSAEnv, decimal: int, max_values: list[int] = None) -> list[int]:
#     if max_values is None:
#         max_values = [env.k_paths, len(env.modulations), env.num_spectrum_resources]
    
#     array = []
#     for max_val in reversed(max_values):
#         array.insert(0, decimal % max_val)
#         decimal //= max_val
#     print(f"Decimal converted to array {array}")

#     # Mapeia o índice relativo de modulação para o índice absoluto
#     allowed_mods = list(range(env.max_modulation_idx, env.max_modulation_idx - env.modulations_to_consider, -1))
    
#     # O mapeamento de modulação acontece para o segundo índice (array[1])
#     array[1] = allowed_mods[array[1]]
#     print(f"Modulation index mapped to absolute index: {array}")
#     return array



def heuristic_from_mask(env: Env, mask: np.ndarray) -> int:
    print("========== Iniciando heuristic_from_mask ==========")
    total_actions = len(mask)
    print("Total de ações a testar:", total_actions)

    valid_actions = []
    errors = []

    for action_index in range(total_actions):
        # Ação de rejeição deve ser sempre válida na máscara
        if action_index // 31 == 0:
            print("a")

        if action_index == env.action_space.n - 1:
            assert mask[action_index] == 1, "Erro: Ação de rejeição deve ser sempre válida"
            continue

        # Decodifica a ação e imprime os detalhes
        decoded = env.encoded_decimal_to_array(action_index)
        print(action_index,": Decodificação da ação:", decoded)
        path_index, modulation_index, candidate_index = decoded

        # Cria o ambiente de simulação (supondo que get_qrmsa_env retorne uma instância fresca)
        qrmsa_env = get_qrmsa_env(env)
        
        # Seleciona o caminho de acordo com os k-shortest paths
        path = qrmsa_env.k_shortest_paths[
            qrmsa_env.current_service.source,
            qrmsa_env.current_service.destination
        ][path_index]

        # Verifica se a modulação obtida é a esperada
        modulation = qrmsa_env.modulations[modulation_index]
        interval = action_index // 320
        print("*" * 30)
        # Define as modulações esperadas para os dois casos:
        if env.max_modulation_idx > 1:
            # Caso padrão: usamos a modulação de índice max_modulation_idx (a "maior") para intervalos pares
            # e a modulação imediatamente inferior (índice max_modulation_idx - 1) para intervalos ímpares.
            higher_mod = qrmsa_env.modulations[env.max_modulation_idx]
            lower_mod  = qrmsa_env.modulations[env.max_modulation_idx - 1]
        else:
            # Se max_modulation_idx é 0, isto indica que a maior modulação permitida é a primeira da lista (por exemplo, 8QAM).
            # Então, para intervalos pares usamos a próxima modulação (índice 1, ex: 16QAM) e para ímpares a própria.
            higher_mod = qrmsa_env.modulations[1]
            lower_mod  = qrmsa_env.modulations[0]

        # print(f"Expected modulations:\n- Para intervalo par: {higher_mod}\n- Para intervalo ímpar: {lower_mod}")
        # print("*" * 30)

        # Seleciona a modulação esperada com base no valor de 'interval'
        if interval % 2 == 0:
            expected_modulation = higher_mod
        else:
            expected_modulation = lower_mod

        assert modulation == expected_modulation, (
            f"Erro: A modulação {modulation.name} não é a esperada {expected_modulation.name} para o índice {modulation_index}."
        )

        # Determina o número de slots, slots disponíveis e os candidatos válidos
        number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation)
        available_slots = qrmsa_env.get_available_slots(path)
        valid_starts = qrmsa_env._get_candidates(available_slots, number_slots, env.num_spectrum_resources)
        
        resource_valid = candidate_index in valid_starts
        if not resource_valid:
            assert mask[action_index] == 0, (
                f"Erro: Ação {action_index} bloqueada por recurso, mas marcada como válida na máscara."
            )
            continue

        # Configura os parâmetros do serviço atual para a simulação
        service = qrmsa_env.current_service
        service.path = path
        service.initial_slot = candidate_index
        service.number_slots = number_slots
        service.center_frequency = (
            qrmsa_env.frequency_start +
            (qrmsa_env.frequency_slot_bandwidth * candidate_index) +
            (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
        )
        service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
        service.launch_power = qrmsa_env.launch_power

        # Calcula o OSNR e valida
        osnr, _, _ = calculate_osnr(qrmsa_env, service)
        # print(f"OSNR calculado para ação {action_index}: {osnr:.2f}")

        threshold = modulation.minimum_osnr + qrmsa_env.margin
        osnr_valid = osnr >= threshold
        if not osnr_valid:
            assert mask[action_index] == 0, (
                f"Erro: Ação {action_index} bloqueada por OSNR, mas marcada como válida na máscara."
            )
            continue

        # Caso a ação seja considerada válida pela simulação, registra a ação
        if mask[action_index] == 1 and not (resource_valid and osnr_valid):
            error_msg = (
                f"Erro: Ação {action_index} marcada como válida na máscara, mas simulação indica ação inválida "
                f"(Resource valid: {resource_valid}, OSNR valid: {osnr_valid}, OSNR: {osnr:.2f})."
            )
            print(error_msg)
            errors.append(error_msg)
        elif mask[action_index] == 0 and (resource_valid and osnr_valid):
            error_msg = (
                f"Erro: Ação {action_index} marcada como inválida na máscara, mas simulação indica ação válida "
                f"(Resource valid: {resource_valid}, OSNR valid: {osnr_valid}, OSNR: {osnr:.2f})."
            )
            print(error_msg)
            errors.append(error_msg)

        valid_actions.append(action_index)
        print(f"Ação {action_index} é considerada válida.")

    print("\n========== Resumo da validação ==========")
    if errors:
        print("Foram encontrados os seguintes erros:")
        for err in errors:
            print(" -", err)
    selected_action = np.random.choice(total_actions)
    return selected_action



def heuristic_shortest_available_path_first_fit_best_modulation(env: Env) -> int:
    blocked_due_to_resources, blocked_due_to_osnr = False, False
    sim_env = get_qrmsa_env(env)
    source = sim_env.current_service.source
    destination = sim_env.current_service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]
    for path_idx, path in enumerate(k_paths):
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]

            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
            if required_slots <= 0:
                continue 
            available_slots = sim_env.get_available_slots(path)
            valid_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)
            
            if not valid_starts:
                blocked_due_to_resources = True
                continue 
            candidate_start = valid_starts[0]
            service = sim_env.current_service
            service.path = path
            service.initial_slot = candidate_start
            service.number_slots = required_slots
            service.current_modulation = modulation
            service.center_frequency = (
                        sim_env.frequency_start +
                        (sim_env.frequency_slot_bandwidth * candidate_start) +
                        (sim_env.frequency_slot_bandwidth * (required_slots / 2)))
            
            service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service.launch_power = sim_env.launch_power
            
            osnr, _, _ = calculate_osnr(sim_env, service)
            threshold = modulation.minimum_osnr + sim_env.margin
            if osnr >= threshold:
                action_index = get_action_index(sim_env, path_idx, modulation_idx, candidate_start)
                return action_index, False, False 
            else:
                blocked_due_to_osnr = True
                if blocked_due_to_resources:
                    blocked_due_to_resources = False
    
    return env.action_space.n - 1, blocked_due_to_resources, blocked_due_to_osnr 


def heuristic_highest_snr(env: Env) -> int:
    best_osnr = -np.inf
    best_action = None
    any_blocked_resources = False
    any_blocked_osnr = False

    sim_env = get_qrmsa_env(env)
    source = sim_env.current_service.source
    destination = sim_env.current_service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]

    for path_idx, path in enumerate(k_paths):
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]

            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
            if required_slots <= 0:
                continue

            available_slots = sim_env.get_available_slots(path)
            valid_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)

            if not valid_starts:
                any_blocked_resources = True
                continue

            for candidate_start in valid_starts:
                service = sim_env.current_service
                service.path = path
                service.initial_slot = candidate_start
                service.number_slots = required_slots
                service.current_modulation = modulation
                service.center_frequency = (
                    sim_env.frequency_start +
                    (sim_env.frequency_slot_bandwidth * candidate_start) +
                    (sim_env.frequency_slot_bandwidth * (required_slots / 2))
                )
                service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
                service.launch_power = sim_env.launch_power

                osnr, _, _ = calculate_osnr(sim_env, service)
                threshold = modulation.minimum_osnr + sim_env.margin

                if osnr >= threshold:
                    if osnr > best_osnr or (osnr == best_osnr and best_action is None):
                        action_index = get_action_index(sim_env, path_idx, modulation_idx, candidate_start)
                        best_osnr = osnr
                        best_action = action_index
                else:
                    any_blocked_osnr = True

    if best_action is not None:
        return best_action, False, False 
    else:
        if any_blocked_osnr:
            any_blocked_resources = False
        return env.action_space.n - 1, any_blocked_resources, any_blocked_osnr

def heuristic_lowest_fragmentation(env: Env) -> tuple[int, bool, bool]:
    sim_env = get_qrmsa_env(env)
    service = sim_env.current_service
    source, destination = service.source, service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]

    best_score = math.inf
    best_action = None
    any_blocked_resources = False
    any_blocked_osnr       = False

    for path_idx, path in enumerate(k_paths):
        raw = sim_env._get_spectrum_slots(path_idx)

        if isinstance(raw, list):
            base_spectra = np.stack(raw, axis=0)
        else:
            arr = np.array(raw)
            if arr.ndim == 1:
                base_spectra = arr[None, :]    # (1, n_slots)
            elif arr.ndim == 2:
                base_spectra = arr
            else:
                raise ValueError(f"Formato inesperado em _get_spectrum_slots: ndim={arr.ndim}")

        for mod_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[mod_idx]
            req_slots  = sim_env.get_number_slots(service, modulation)+1
            if req_slots <= 0:
                continue

            available = sim_env.get_available_slots(path)
            candidates = sim_env._get_candidates(
                available, req_slots, sim_env.num_spectrum_resources
            )
            if not candidates:
                any_blocked_resources = True
                continue

            for start in candidates:
                # 3) simula a alocação: clone e pinta o bloco como 1=ocupado
                tmp = base_spectra.copy()
                tmp[:, start:start+req_slots] = 1

                # 4) calcula métricas de fragmentação
                # 4a) entropia média
                entropies = [ link_shannon_entropy_(row.tolist()) for row in tmp ]
                se   = sum(entropies) / len(entropies) if entropies else 0.0
                # 4b) cuts
                cuts = fragmentation_route_cuts([row.tolist() for row in tmp])
                # 4c) rss
                rss  = fragmentation_route_rss([row.tolist() for row in tmp])

                score = 0.33 * se + 0.33 * cuts + 0.34 * rss

                # 5) verifica OSNR
                service.path             = path
                service.initial_slot     = start
                service.number_slots     = req_slots
                service.current_modulation = modulation
                service.center_frequency = (
                    sim_env.frequency_start
                    + sim_env.frequency_slot_bandwidth * start
                    + sim_env.frequency_slot_bandwidth * (req_slots / 2)
                )
                service.bandwidth    = sim_env.frequency_slot_bandwidth * req_slots
                service.launch_power = sim_env.launch_power

                osnr, _, _ = calculate_osnr(sim_env, service)
                if osnr < modulation.minimum_osnr + sim_env.margin:
                    any_blocked_osnr = True
                    continue

                # 6) escolhe o candidato de menor fragmentação
                if score < best_score:
                    best_score  = score
                    best_action = get_action_index(sim_env, path_idx, mod_idx, start)

    # 7) retorna conforme convenção
    if best_action is not None:
        return best_action, False, False
    else:
        if any_blocked_osnr:
            any_blocked_resources = False
        return env.action_space.n - 1, any_blocked_resources, any_blocked_osnr




def shortest_available_path_first_fit_best_modulation(
    mask: np.ndarray,
) -> Optional[int]:
    return int(np.where(mask == 1)[0][0])

def rnd(
    mask: np.ndarray,
) -> Optional[int]:
    valid_actions = np.where(mask == 1)[0]
    return int(np.random.choice(valid_actions))


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
    qrmsa_env: QRMSAEnv = get_qrmsa_env(env) 

    for idp, path in enumerate(qrmsa_env.k_shortest_paths[
        qrmsa_env.current_service.source,
        qrmsa_env.current_service.destination,
    ]):
        available_slots = qrmsa_env.get_available_slots(path)
        for idm, modulation in zip(
            range(len(qrmsa_env.modulations) - 1, -1, -1),
            reversed(qrmsa_env.modulations)
        ):
            number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation) + 2 

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
