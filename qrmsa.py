from typing import Any, Literal, Sequence, SupportsFloat, Optional
from dataclasses import field

cimport cython
cimport numpy as cnp
from libc.stdint cimport uint32_t
from libc.math cimport log, exp, asinh, log10
cnp.import_array()

import gymnasium as gym
from gymnasium.utils import seeding
import functools
import heapq
import networkx as nx
import random
import numpy as np
from collections import defaultdict
from numpy.random import SeedSequence
from optical_networking_gym.utils import rle
from optical_networking_gym.core.osnr import calculate_osnr, calculate_osnr_observation
import math
import typing
import os
from scipy.signal import convolve

if typing.TYPE_CHECKING:
    from optical_networking_gym.topology import Link, Span, Modulation, Path

cdef class Service:
    cdef public int service_id
    cdef public str source
    cdef public int source_id
    cdef public object destination
    cdef public object destination_id
    cdef public float arrival_time
    cdef public float holding_time
    cdef public float bit_rate
    cdef public object path
    cdef public int service_class
    cdef public int initial_slot
    cdef public double center_frequency
    cdef public double bandwidth
    cdef public int number_slots
    cdef public int core
    cdef public double launch_power
    cdef public bint accepted
    cdef public bint blocked_due_to_resources
    cdef public bint blocked_due_to_osnr
    cdef public double OSNR
    cdef public double ASE
    cdef public double NLI
    cdef public object current_modulation
    cdef public bint recalculate

    def __init__(
        self,
        int service_id,
        str source,
        int source_id,
        str destination = None,
        str destination_id = None,
        float arrival_time = 0.0,
        float holding_time = 0.0,
        float bit_rate = 0.0,
        object path = None,
        int service_class = 0,
        int initial_slot = 0,
        int center_frequency = 0,
        int bandwidth = 0,
        int number_slots = 0,
        int core = 0,
        double launch_power = 0.0,
        bint accepted = False,
        bint blocked_due_to_resources = True,
        bint blocked_due_to_osnr = True,
        float OSNR = 0.0,
        float ASE = 0.0,
        float NLI = 0.0,
        object current_modulation = None
    ):
        self.service_id = service_id
        self.source = source
        self.source_id = source_id
        self.destination = destination
        self.destination_id = destination_id
        self.arrival_time = arrival_time
        self.holding_time = holding_time
        self.bit_rate = bit_rate
        self.path = path
        self.service_class = service_class
        self.initial_slot = initial_slot
        self.center_frequency = center_frequency
        self.bandwidth = bandwidth
        self.number_slots = number_slots
        self.core = core
        self.launch_power = launch_power
        self.accepted = accepted
        self.blocked_due_to_resources = blocked_due_to_resources
        self.blocked_due_to_osnr = blocked_due_to_osnr
        self.OSNR = OSNR
        self.ASE = ASE
        self.NLI = NLI
        self.current_modulation = current_modulation
        self.recalculate = False

    def __repr__(self):
        return (
            f"Service(service_id={self.service_id}, source='{self.source}', source_id={self.source_id}, "
            f"destination='{self.destination}', destination_id={self.destination_id}, arrival_time={self.arrival_time}, "
            f"holding_time={self.holding_time}, bit_rate={self.bit_rate}, path={self.path}, service_class={self.service_class}, "
            f"initial_slot={self.initial_slot}, center_frequency={self.center_frequency}, bandwidth={self.bandwidth}, "
            f"number_slots={self.number_slots}, core={self.core}, launch_power={self.launch_power}, accepted={self.accepted}, "
            f"blocked_due_to_resources={self.blocked_due_to_resources}, blocked_due_to_osnr={self.blocked_due_to_osnr}, "
            f"OSNR={self.OSNR}, ASE={self.ASE}, NLI={self.NLI}, current_modulation={self.current_modulation}, "
            f"recalculate={self.recalculate})"
        )

cdef class QRMSAEnv:
    cdef public uint32_t input_seed
    cdef public double load
    cdef int episode_length
    cdef double mean_service_holding_time
    cdef public int num_spectrum_resources
    cdef public double channel_width
    cdef bint allow_rejection
    cdef readonly object topology
    cdef readonly str bit_rate_selection
    cdef public tuple bit_rates
    cdef double bit_rate_lower_bound
    cdef double bit_rate_higher_bound
    cdef object bit_rate_probabilities
    cdef object node_request_probabilities
    cdef public object k_shortest_paths
    cdef int k_paths
    cdef public double launch_power_dbm
    cdef public double launch_power
    cdef double bandwidth
    cdef public double frequency_start
    cdef public double frequency_end
    cdef public double frequency_slot_bandwidth
    cdef public double margin
    cdef public object modulations
    cdef bint measure_disruptions
    cdef public object _np_random
    cdef public int _np_random_seed
    cdef object spectrum_use
    cdef object spectrum_allocation
    cdef public Service current_service
    cdef int service_id_counter
    cdef list services_in_progress
    cdef list release_times
    cdef int services_processed
    cdef int services_accepted
    cdef int episode_services_processed
    cdef int episode_services_accepted
    cdef double bit_rate_requested
    cdef double bit_rate_provisioned
    cdef double episode_bit_rate_requested
    cdef double episode_bit_rate_provisioned
    cdef object bit_rate_requested_histogram
    cdef object bit_rate_provisioned_histogram
    cdef object slots_provisioned_histogram
    cdef object episode_slots_provisioned_histogram
    cdef int disrupted_services
    cdef int episode_disrupted_services
    cdef list disrupted_services_list
    cdef public object action_space
    cdef public object observation_space
    cdef object episode_actions_output
    cdef object episode_actions_taken
    cdef object episode_modulation_histogram
    cdef object episode_bit_rate_requested_histogram
    cdef object episode_bit_rate_provisioned_histogram
    cdef object spectrum_slots_allocation
    cdef int reject_action
    cdef object actions_output
    cdef object actions_taken
    cdef bint _new_service
    cdef double current_time
    cdef double mean_service_inter_arrival_time
    cdef public object frequency_vector
    cdef object rng
    cdef object bit_rate_function
    cdef list _events
    cdef object file_stats
    cdef unicode final_file_name
    cdef int j

    topology: cython.declare(nx.Graph, visibility="readonly")
    bit_rate_selection: cython.declare(Literal["continuous", "discrete"], visibility="readonly")
    bit_rates: cython.declare(tuple[int, int, int] or tuple[float, float, float], visibility="readonly")

    def __init__(
        self,
        topology: nx.Graph,
        num_spectrum_resources: int = 320,
        episode_length: int = 1000,
        load: float = 10.0,
        mean_service_holding_time: double = 10800.0,
        bit_rate_selection: str = "continuous",
        bit_rates: tuple = (10, 40, 100),
        bit_rate_probabilities = None,
        node_request_probabilities = None,
        bit_rate_lower_bound: float = 25.0,
        bit_rate_higher_bound: float = 100.0,
        launch_power_dbm: float = 0.0,
        bandwidth: float = 4e12,
        frequency_start: float = (3e8 / 1565e-9),
        frequency_slot_bandwidth: float = 12.5e9,
        margin: float = 0.0,
        measure_disruptions: bool = False,
        seed: object = None,
        allow_rejection: bool = False,
        reset: bool = True,
        channel_width: double = 12.5,
        k_paths: int = 5,
        file_name: str = "",
        j: int = 4
    ):
        self.rng = random.Random()
        self.mean_service_inter_arrival_time = 0
        self.set_load(load=load, mean_service_holding_time=mean_service_holding_time)
        self.bit_rate_selection = bit_rate_selection

        if self.bit_rate_selection == "continuous":
            self.bit_rate_lower_bound = bit_rate_lower_bound
            self.bit_rate_higher_bound = bit_rate_higher_bound
            self.bit_rate_function = functools.partial(
                self.rng.randint,
                int(self.bit_rate_lower_bound),
                int(self.bit_rate_higher_bound)
            )
        elif self.bit_rate_selection == "discrete":
            if bit_rate_probabilities is None:
                bit_rate_probabilities = [1.0 / len(bit_rates) for _ in range(len(bit_rates))]
            self.bit_rate_probabilities = bit_rate_probabilities
            self.bit_rates = bit_rates
            self.bit_rate_function = functools.partial(
                self.rng.choices, self.bit_rates, self.bit_rate_probabilities, k=1
            )
            self.bit_rate_requested_histogram = defaultdict(int)
            self.bit_rate_provisioned_histogram = defaultdict(int)
            self.episode_bit_rate_requested_histogram = defaultdict(int)
            self.episode_bit_rate_provisioned_histogram = defaultdict(int)
        self.topology = topology
        self.num_spectrum_resources = num_spectrum_resources
        self.episode_length = episode_length
        self.load = load
        self.mean_service_holding_time = mean_service_holding_time
        self.channel_width = channel_width
        self.allow_rejection = allow_rejection
        self.k_paths = k_paths
        self.k_shortest_paths = self.topology.graph["ksp"]
        if node_request_probabilities is not None:
            self.node_request_probabilities = node_request_probabilities
        else:
            tmp_probabilities = np.full(
                (self.topology.number_of_nodes(),),
                fill_value=1.0 / self.topology.number_of_nodes(),
                dtype=np.float64
            )
            self.node_request_probabilities = np.asarray(tmp_probabilities, dtype=np.float64)
        self.launch_power_dbm = launch_power_dbm
        self.launch_power = 10 ** ((self.launch_power_dbm - 30) / 10)
        self.bandwidth = bandwidth
        self.frequency_start = frequency_start
        self.frequency_slot_bandwidth = frequency_slot_bandwidth
        self.margin = margin
        self.measure_disruptions = measure_disruptions
        self.frequency_end = self.frequency_start + (self.frequency_slot_bandwidth * self.num_spectrum_resources)
        assert math.isclose(self.frequency_end - self.frequency_start, self.bandwidth, rel_tol=1e-5)
        self.frequency_vector = np.linspace(
            self.frequency_start,
            self.frequency_end,
            num=self.num_spectrum_resources,
            dtype=np.float64
        )
        assert self.frequency_vector.shape[0] == self.num_spectrum_resources, (
            f"Size of frequency_vector ({self.frequency_vector.shape[0]}) "
            f"does not match num_spectrum_resources ({self.num_spectrum_resources})."
        )
        self.topology.graph["available_slots"] = np.ones(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            dtype=np.int32
        )
       
        self.modulations = self.topology.graph.get("modulations", [])
        self.disrupted_services_list = []
        self.disrupted_services = 0
        self.episode_disrupted_services = 0

        self.action_space = gym.spaces.MultiDiscrete(
            (self.k_paths, len(self.modulations), self.frequency_vector.shape[0])
        )

        total_dim = (
            1  # Taxa de bits normalizada
            + (2 * self.topology.number_of_nodes())  # Origem e destino codificados
            + self.k_paths  # Comprimentos das rotas
            + (self.k_paths * len(self.modulations) * (2 * self.j + 5))  # Espectro
        )

        # Espaço de observação como um Box contínuo
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(
                low=-10.0,  # Limite inferior, com base nas normalizações
                high=10.0,  # Limite superior, com base nas normalizações
                shape=(total_dim,),  # Dimensão total calculada
                dtype=np.float32  # Tipo de dado usado na observação
            ),
            'mask': gym.spaces.Box(
                low=0.0,  # Máscara de ações válidas deve estar no intervalo [0, 1]
                high=1.0,
                shape=(self.k_paths, len(self.modulations), self.frequency_vector.shape[0]),  # Dimensão da máscara
                dtype=np.uint8
            )
        })
        if seed is None:
            ss = SeedSequence()
            input_seed = int(ss.generate_state(1)[0])
        elif isinstance(seed, int):
            input_seed = int(seed)
        else:
            raise ValueError("Seed must be an integer.")
        input_seed = input_seed % (2 ** 31)
        if input_seed >= 2 ** 31:
            input_seed -= 2 ** 32
        self.input_seed = int(input_seed)
        self._np_random, self._np_random_seed = seeding.np_random(self.input_seed)
        num_edges = self.topology.number_of_edges()
        num_resources = self.num_spectrum_resources
        self.spectrum_use = np.zeros(
            (num_edges, num_resources), dtype=np.int32
        )
        self.spectrum_allocation = np.full(
            (num_edges, num_resources),
            fill_value=-1,
            dtype=np.int64
        )
        self.current_service = None
        self.service_id_counter = 0
        self.services_in_progress = []
        self.release_times = []
        self.current_time = 0.0
        self._events = []
        self.services_processed = 0
        self.services_accepted = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.bit_rate_requested = 0.0
        self.bit_rate_provisioned = 0.0
        self.episode_bit_rate_requested = 0.0
        self.episode_bit_rate_provisioned = 0.0
        if self.bit_rate_selection == "discrete":
            if bit_rate_probabilities is None:
                bit_rate_probabilities = [1.0 / len(bit_rates)] * len(bit_rates)
            self.bit_rate_probabilities = bit_rate_probabilities
            self.bit_rate_requested_histogram = defaultdict(int)
            self.bit_rate_provisioned_histogram = defaultdict(int)
            self.slots_provisioned_histogram = defaultdict(int)
            self.episode_slots_provisioned_histogram = defaultdict(int)
        else:
            self.bit_rate_requested_histogram = None
            self.bit_rate_provisioned_histogram = None
        self.reject_action = 1 if allow_rejection else 0
        self.actions_output = np.zeros(
            (self.k_paths + 1, self.num_spectrum_resources + 1), dtype=np.int64
        )
        self.actions_taken = np.zeros(
            (self.k_paths + 1, self.num_spectrum_resources + 1), dtype=np.int64
        )
        if file_name != "":
            final_name = "_".join([
                file_name,
                str(self.topology.graph["name"]),
                str(self.launch_power_dbm),
                str(self.load),
                str(seed) + ".csv"
            ])

            dir_name = os.path.dirname(final_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            self.final_file_name = final_name
            self.file_stats = open(final_name, "wt", encoding="UTF-8")

            # Write to the file
            self.file_stats.write("# Service stats file from simulator\n")
            self.file_stats.write("id,source,destination,bit_rate,path_k,path_length,modulation,min_osnr,osnr,ase,nli,disrupted_services\n")
        else:
            self.file_stats = None

        if reset:
            self.reset()

    cpdef tuple reset(self, object seed=None, dict options=None):
        self.episode_bit_rate_requested = 0.0
        self.episode_bit_rate_provisioned = 0.0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_disrupted_services = 0
        self._events = []

        self.episode_actions_output = np.zeros(
            (self.k_paths + self.reject_action, self.num_spectrum_resources + self.reject_action),
            dtype=np.int32
        )
        self.episode_actions_taken = np.zeros(
            (self.k_paths + self.reject_action, self.num_spectrum_resources + self.reject_action),
            dtype=np.int32
        )

        if self.bit_rate_selection == "discrete":
            self.episode_bit_rate_requested_histogram = {}
            self.episode_bit_rate_provisioned_histogram = {}
            for bit_rate in self.bit_rates:
                self.episode_bit_rate_requested_histogram[bit_rate] = 0
                self.episode_bit_rate_provisioned_histogram[bit_rate] = 0

        self.episode_modulation_histogram = {}
        for modulation in self.modulations:
            self.episode_modulation_histogram[modulation.spectral_efficiency] = 0

        if self._new_service:
            self.episode_services_processed += 1
            self.episode_bit_rate_requested += self.current_service.bit_rate
            if self.bit_rate_selection == "discrete":
                self.episode_bit_rate_requested_histogram[self.current_service.bit_rate] += 1

        if options is not None and "only_episode_counters" in options and options["only_episode_counters"]:
            return self.observation(), {}

        # Reset other necessary variables
        self.bit_rate_requested = 0.0
        self.bit_rate_provisioned = 0.0
        self.disrupted_services = 0
        self.disrupted_services_list = []

        # Reset topology state
        self.topology.graph["services"] = []
        self.topology.graph["running_services"] = []
        self.topology.graph["last_update"] = 0.0

        for lnk in self.topology.edges():
            self.topology[lnk[0]][lnk[1]]["utilization"] = 0.0
            self.topology[lnk[0]][lnk[1]]["last_update"] = 0.0
            self.topology[lnk[0]][lnk[1]]["services"] = []
            self.topology[lnk[0]][lnk[1]]["running_services"] = []

        self.topology.graph["available_slots"] = np.ones(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            dtype=np.int32
        )

        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            fill_value=-1,
            dtype=np.int32
        )

        self.topology.graph["compactness"] = 0.0
        self.topology.graph["throughput"] = 0.0

        for lnk in self.topology.edges():
            self.topology[lnk[0]][lnk[1]]["external_fragmentation"] = 0.0
            self.topology[lnk[0]][lnk[1]]["compactness"] = 0.0

        self._new_service = False
        self._next_service()
        return self.observation(), {}

    
    def observation(self):
        topology = self.topology
        current_service = self.current_service
        num_spectrum_resources = self.num_spectrum_resources
        k_shortest_paths = self.k_shortest_paths
        modulations = self.modulations
        num_modulations = len(modulations)
        num_nodes = topology.number_of_nodes()
        frequency_slot_bandwidth = self.channel_width * 1e9 

        # IDs de origem e destino
        source_id = int(current_service.source_id)
        destination_id = int(current_service.destination_id)

        # Criação da matriz binária de origem e destino
        source_destination_tau = np.zeros((2, num_nodes), dtype=np.float32)
        source_destination_tau[0, source_id] = 1.0
        source_destination_tau[1, destination_id] = 1.0

        # Parâmetros fixos
        num_paths_to_evaluate = self.k_paths  # Número de rotas k mais curtas a avaliar
        num_blocks_to_consider = self.j  # Número máximo de blocos disponíveis a considerar por rota
        num_metrics_per_modulation = 2 * num_blocks_to_consider + 5  # Métricas por modulação

        # Inicialização da observação do espectro com valores -1.0
        spectrum_obs = np.full(
            (num_paths_to_evaluate, num_modulations, num_metrics_per_modulation),
            fill_value=-1.0,
            dtype=np.float32
        )

        # Inicialização da máscara de ações válidas com zeros
        action_mask = np.zeros((num_paths_to_evaluate, num_modulations, self.frequency_vector.shape[0]), dtype=np.uint8)

        # Estatísticas da topologia para normalização
        link_lengths = [topology[x][y]["length"] for x, y in topology.edges()]
        min_lenghts = min(link_lengths)
        max_lenghts = max(link_lengths)
        std_link_length = np.std(link_lengths)
        std_link_length = std_link_length if std_link_length != 0 else 1.0  # Evitar divisão por zero

        # Valores máximos para normalização
        max_block_length = num_spectrum_resources  # Máximo comprimento de bloco espectral
        max_bit_rate = max(self.bit_rates)         # Taxa de bits máxima suportada

        # Inicialização dos comprimentos das rotas normalizados
        route_lengths = np.zeros((num_paths_to_evaluate, 1), dtype=np.float32)

        # Processar cada rota k mais curta
        for path_index, route in enumerate(k_shortest_paths[current_service.source, current_service.destination]):
            if path_index >= num_paths_to_evaluate:
                break  # Considerar apenas as primeiras num_paths_to_evaluate rotas

            # Normalização do comprimento da rota
            route_length = route.length
            route_lengths[path_index, 0] = self.normalize(route_length, min_lenghts, max_lenghts)

            # Obter slots disponíveis para a rota
            available_slots = self.get_available_slots(route)  # Assume que retorna um array booleano

            # Processar cada modulação disponível
            for modulation_index, modulation in enumerate(modulations):
                # Número de slots necessários para o serviço atual com esta modulação
                num_slots_required = self.get_number_slots(current_service, modulation)

                # Inicialização dos valores de OSNR
                osnr_ok = False  # Inicialização
                osnr_value = -1.0  # Valor padrão em caso de falha

                # Identificar blocos disponíveis usando RLE
                idx, values, lengths = rle(available_slots)
                initial_indices = idx[values == 1]
                lengths_available = lengths[values == 1]

                # Filtrar blocos que possuem tamanho suficiente
                valid_blocks = lengths_available >= num_slots_required
                initial_indices = initial_indices[valid_blocks]
                lengths_valid = lengths_available[valid_blocks]

                # Se houver pelo menos um bloco válido, calcular a OSNR
                if initial_indices.size > 0:
                    # Selecionar o último bloco válido para calcular a OSNR 
                    initial_slot = initial_indices[-1]
                    service_bandwidth = num_slots_required * frequency_slot_bandwidth
                    service_center_frequency = self.frequency_start + (
                        frequency_slot_bandwidth * initial_slot
                    ) + (
                        frequency_slot_bandwidth * (num_slots_required / 2.0)
                    )
                    path_links = route.links
                    service_id = self.current_service.service_id
                    service_launch_power = 10 ** ((self.launch_power_dbm - 30) / 10)
                    gsnr_th = modulation.minimum_osnr
                    osnr_value = calculate_osnr_observation(
                        self,
                        path_links,
                        service_bandwidth,
                        service_center_frequency,
                        service_id,
                        service_launch_power,
                        gsnr_th
                    )

                    # Verificar se a OSNR atende ao requisito
                    if osnr_value >= 0:
                        osnr_ok = True

                # Atualizar a observação do espectro
                # Preencher observação do espectro (mesmo que a alocação possa não ser válida)
                for block_index, (initial_index, block_length) in enumerate(zip(initial_indices[:num_blocks_to_consider], lengths_valid[:num_blocks_to_consider])):
                    # Índice inicial do slot, normalizado
                    spectrum_obs[path_index, modulation_index, block_index * 2] = (
                        2 * (initial_index - 0.5 * num_spectrum_resources) / num_spectrum_resources
                    )
                    # Comprimento do bloco, normalizado
                    spectrum_obs[path_index, modulation_index, block_index * 2 + 1] = self.normalize(block_length, max_block_length, num_slots_required)

                # Número normalizado de slots necessários
                spectrum_obs[path_index, modulation_index, num_blocks_to_consider * 2] = (num_slots_required - 5.5) / 3.5

                # Proporção total de slots disponíveis
                total_available_slots = np.sum(available_slots)
                total_available_slots_ratio = (
                    2 * (total_available_slots - 0.5 * num_spectrum_resources) / num_spectrum_resources
                )
                spectrum_obs[path_index, modulation_index, num_blocks_to_consider * 2 + 1] = total_available_slots_ratio

                # Tamanho médio dos blocos disponíveis
                if lengths_valid.size > 0:
                    mean_block_size = ((np.mean(lengths_valid) - 4) / 4)/100
                    spectrum_obs[path_index, modulation_index, num_blocks_to_consider * 2 + 2] = mean_block_size
                else:
                    spectrum_obs[path_index, modulation_index, num_blocks_to_consider * 2 + 2] = 0.0  # Sem blocos disponíveis

                # Cálculo da OSNR
                spectrum_obs[path_index, modulation_index, num_blocks_to_consider * 2 + 3] = osnr_value

                # Atualizar a máscara de ações
                if osnr_ok:
                    # Encontrar todos os possíveis inícios de alocação de blocos válidos
                    available_slots_int = available_slots.astype(int)
                    window = np.ones(num_slots_required, dtype=int)
                    convolution = convolve(available_slots_int, window, mode='valid')
                    valid_start_indices = np.where(convolution == num_slots_required)[0]
                    # Marcar na máscara as frequências onde a alocação é possível
                    for start_idx in valid_start_indices:
                        action_mask[path_index, modulation_index, start_idx] = 1.0
                else:
                    # Se OSNR não está ok, todas as frequências permanecem como 0 (já estão inicializadas)
                    pass
        # Observação da taxa de bits do serviço atual, normalizada
        bit_rate_obs = np.array([[current_service.bit_rate / max_bit_rate]], dtype=np.float32)

        # Concatenar todas as observações em um array de tamanho fixo (sem incluir a máscara)
        observation = np.concatenate((
            bit_rate_obs.flatten(),
            source_destination_tau.flatten(),
            route_lengths.flatten(),
            spectrum_obs.flatten()
        ), axis=0)

        # Garantir que a observação tenha o tipo correto
        observation = observation.astype(np.float32)

        return {'observation': observation, 'mask': action_mask}


        # available_slots = self.topology.graph["available_slots"]
        # running_services = self.topology.graph["running_services"]
        
        # # Create a padded array to store service IDs
        # padded_running_services = np.full((1000,), fill_value=-1, dtype=int)
        
        # # Extract the service_id from each Service object and store it in the array
        # service_ids = [service.service_id for service in running_services[:1000]]
        
        # # Store the service IDs in the padded array
        # padded_running_services[:len(service_ids)] = service_ids

        # return {
        #     "topology": available_slots,
        #     "running-services": padded_running_services
        # }


    cpdef tuple[object, float, bint, bint, dict] step(self, cnp.ndarray action):
        cdef int route
        cdef object modulation
        cdef int modulation_index
        cdef int initial_slot
        cdef int number_slots
        cdef double osnr, ase, nli
        cdef int disrupted_services = 0
        cdef object path
        cdef list services_to_measure = []
        cdef object link
        cdef object service
        cdef dict info
        cdef str line
        cdef str key
        cdef bint truncated = False
        cdef bint terminated
        if action is not None:
            # Unpack the action tuple
            route = int(action[0])
            modulation_index = int(action[1])
            modulation = self.modulations[modulation_index]
            initial_slot = int(action[2])

            # Get the modulation object
            modulation = self.modulations[modulation_index]

            # Get the path based on the route index
            path = self.k_shortest_paths[self.current_service.source, self.current_service.destination][route]
            # Calculate the number of slots required
            number_slots = self.get_number_slots(
                service=self.current_service,
                modulation=modulation,
            )
            # Check if the path is free for the given slots
            path_free_bool = self.is_path_free(path=path, initial_slot=initial_slot, number_slots=number_slots)
            if path_free_bool:
                self.current_service.path = path
                self.current_service.initial_slot = initial_slot
                self.current_service.number_slots = number_slots
                self.current_service.center_frequency = self.frequency_start + \
                    (self.frequency_slot_bandwidth * initial_slot) + \
                    (self.frequency_slot_bandwidth * (number_slots / 2.0))
                self.current_service.bandwidth = self.frequency_slot_bandwidth * number_slots
                self.current_service.launch_power = self.launch_power

                # Calculate OSNR, ASE, and NLI
                osnr, ase, nli = calculate_osnr(self, self.current_service)

                # Check if OSNR meets the required minimum
                if osnr >= modulation.minimum_osnr + self.margin:
                    self.current_service.OSNR = osnr
                    self.current_service.ASE = ase
                    self.current_service.NLI = nli
                    if self.current_service.service_id > self.episode_length:
                        self.max_gsnr = max(self.max_gsnr, osnr)
                        self.min_gsnr = min(self.min_gsnr, osnr)

                    self.current_service.current_modulation = modulation
                    self.episode_modulation_histogram[
                        modulation.spectral_efficiency
                    ] += 1

                    self._provision_path(
                        path,
                        initial_slot,
                        number_slots,
                    )

                    self.current_service.accepted = True

                    if self.bit_rate_selection == "discrete":
                        # If discrete bit rate is being used
                        self.slots_provisioned_histogram[number_slots] += 1  # Update histogram

                    self._add_release(self.current_service)
                else:
                    self.current_service.accepted = False
        else:
            self.current_service.accepted = False
        # Initialize disrupted services count
        disrupted_services = 0

        # Check for service disruptions
        if self.measure_disruptions and self.current_service.accepted:
            services_to_measure = []

            for link in self.current_service.path.links:
                for service in self.topology[link.node1][link.node2]["running_services"]:
                    if service not in services_to_measure and service not in self.disrupted_services_list:
                        services_to_measure.append(service)

            for service in services_to_measure:
                osnr, ase, nli = calculate_osnr(self, service)
                if osnr < service.current_modulation.minimum_osnr:
                    disrupted_services += 1
                    if service not in self.disrupted_services_list:
                        self.disrupted_services += 1
                        self.episode_disrupted_services += 1
                        self.disrupted_services_list.append(service)

        # Handle the case when the service is not accepted
        if not self.current_service.accepted:
            self.actions_taken[self.k_paths, self.num_spectrum_resources] += 1
            self.current_service.path = None
            self.current_service.initial_slot = -1
            self.current_service.number_slots = 0
            self.current_service.OSNR = 0.0
            self.current_service.ASE = 0.0
            self.current_service.NLI = 0.0

        # Append the current service to the services list
        self.topology.graph["services"].append(self.current_service)

        # Writing statistics to file if file_stats is enabled
        if self.file_stats is not None:
            line = "{},{},{},{},".format(
                self.current_service.service_id,
                self.current_service.source_id,
                self.current_service.destination_id,
                self.current_service.bit_rate,
            )
            if self.current_service.accepted:
                line += "{},{},{},{},{},{},{},{}".format(
                    self.current_service.path.k,
                    self.current_service.path.length,
                    self.current_service.current_modulation.spectral_efficiency,
                    self.current_service.current_modulation.minimum_osnr,
                    self.current_service.OSNR,
                    self.current_service.ASE,
                    self.current_service.NLI,
                    disrupted_services,
                )
            else:
                line += "-1,-1,-1,-1,-1,-1,-1,-1"
            line += "\n"
            self.file_stats.write(line)

        # Generate statistics for the episode info
        reward = self.reward()

        info = {
            "episode_services_accepted": self.episode_services_accepted,
            "service_blocking_rate": 0.0,
            "episode_service_blocking_rate": 0.0,
            "bit_rate_blocking_rate": 0.0,
            "episode_bit_rate_blocking_rate": 0.0,
            "disrupted_services": 0.0,
            "episode_disrupted_services": 0.0,
        }

        # Cálculos das taxas de bloqueio
        if self.services_processed > 0:
            info["service_blocking_rate"] = (
                self.services_processed - self.services_accepted
            ) / self.services_processed

        if self.episode_services_processed > 0:
            info["episode_service_blocking_rate"] = (
                self.episode_services_processed - self.episode_services_accepted
            ) / self.episode_services_processed

        if self.bit_rate_requested > 0:
            info["bit_rate_blocking_rate"] = (
                self.bit_rate_requested - self.bit_rate_provisioned
            ) / self.bit_rate_requested

        if self.episode_bit_rate_requested > 0:
            info["episode_bit_rate_blocking_rate"] = (
                self.episode_bit_rate_requested - self.episode_bit_rate_provisioned
            ) / self.episode_bit_rate_requested

        if self.disrupted_services > 0 and self.services_accepted > 0:
            info["disrupted_services"] = self.disrupted_services / self.services_accepted

        if self.episode_disrupted_services > 0 and self.episode_services_accepted > 0:
            info["episode_disrupted_services"] = (
                self.episode_disrupted_services / self.episode_services_accepted
            )

        # Atualização das contagens de modulação
        for current_modulation in self.modulations:
            spectral_eff = current_modulation.spectral_efficiency
            key = "modulation_{}".format(str(spectral_eff))  # Converter para string se necessário
            if spectral_eff in self.episode_modulation_histogram:
                info[key] = self.episode_modulation_histogram[spectral_eff]
            else:
                print(f"Warning: spectral_efficiency {spectral_eff} not found in histogram.")
                info[key] = 0  # Valor padrão ou outro tratamento adequado

        # Preparar para o próximo serviço
        self._new_service = False
        self._next_service()
        terminated = self.episode_services_processed == self.episode_length
        return (
            self.observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    cpdef _next_service(self):
        """
        Advances to the next service in the environment.
        """
        cdef float at
        cdef float ht, time
        cdef str src, dst,  dst_id
        cdef float bit_rate
        cdef object service
        cdef int src_id
        cdef object service_to_release
        cdef float lambd

        if self._new_service:
            return
        at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        
        self.current_time = at

        ht = self.rng.expovariate(1.0 / self.mean_service_holding_time)

        src, src_id, dst, dst_id = self._get_node_pair()
        if self.bit_rate_selection == "continuous":
            bit_rate = self.bit_rate_function()
        else:
            bit_rate = self.bit_rate_function()[0]

        service = Service(
            service_id=self.episode_services_processed,
            source=src,
            source_id=src_id,  
            destination=dst,
            destination_id=dst_id,
            arrival_time=at,
            holding_time=ht,
            bit_rate=bit_rate
        )
        self.current_service = service
        self._new_service = True

        self.services_processed += 1
        self.episode_services_processed += 1

        self.bit_rate_requested += self.current_service.bit_rate
        self.episode_bit_rate_requested += self.current_service.bit_rate
        if self.bit_rate_selection == "discrete":
            self.bit_rate_requested_histogram[self.current_service.bit_rate] += 1
            self.episode_bit_rate_requested_histogram[self.current_service.bit_rate] += 1

        while len(self._events) > 0:
            time, _, service_to_release = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:
                heapq.heappush(self._events, (time, service_to_release.service_id, service_to_release))
                break 
    
    cpdef void set_load(self, double load=-1.0, float mean_service_holding_time=-1.0):
        if load > 0:
            self.load = load
        if mean_service_holding_time > 0:
            self.mean_service_holding_time = mean_service_holding_time
        if self.load > 0 and self.mean_service_holding_time > 0:
            self.mean_service_inter_arrival_time = 1 / (self.load / self.mean_service_holding_time)
        else:
            raise ValueError("Both load and mean_service_holding_time must be positive values.")
    
    cdef tuple _get_node_pair(self):
        """
        Uses the `node_request_probabilities` variable to generate a source and a destination.

        :return: source node (int), source node id (int), destination node (int), destination node id (int)
        """
        cdef list nodes = [x for x in self.topology.nodes()]
        
        cdef str src = self.rng.choices(nodes, weights=self.node_request_probabilities)[0]
        cdef int src_id = self.topology.graph["node_indices"].index(src)  

        cdef cnp.ndarray[cnp.float64_t, ndim=1] new_node_probabilities = np.copy(self.node_request_probabilities)
        new_node_probabilities[src_id] = 0.0

        new_node_probabilities /= np.sum(new_node_probabilities)

        cdef str dst = self.rng.choices(nodes, weights=new_node_probabilities)[0]
        cdef str dst_id = str(self.topology.graph["node_indices"].index(dst))

        return src, src_id, dst, dst_id

    cpdef double _get_network_compactness(self):
            """
            Calculate network spectrum compactness based on:
            https://ieeexplore.ieee.org/abstract/document/6476152
            """

            cdef double sum_slots_paths = 0.0  
            cdef double sum_occupied = 0.0     
            cdef double sum_unused_spectrum_blocks = 0.0  

            cdef list running_services = self.topology.graph["running_services"]

            for service in running_services:
                sum_slots_paths += service.number_slots * service.path.hops

            for n1, n2 in self.topology.edges():
                index = self.topology[n1][n2]["index"]
                available_slots = self.topology.graph["available_slots"][index, :]

                initial_indices, values, lengths = rle(available_slots)

                used_blocks = [i for i, x in enumerate(values) if x == 0]

                if len(used_blocks) > 1:
                    lambda_min = initial_indices[used_blocks[0]]
                    lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]

                    sum_occupied += lambda_max - lambda_min

                    # Analyze the spectrum only within the used portion of the spectrum
                    internal_idx, internal_values, internal_lengths = rle(
                        available_slots[lambda_min:lambda_max]
                    )
                    sum_unused_spectrum_blocks += np.sum(internal_values)

            if sum_unused_spectrum_blocks > 0:
                cur_spectrum_compactness = (sum_occupied / sum_slots_paths) * (
                    self.topology.number_of_edges() / sum_unused_spectrum_blocks
                )
            else:
                cur_spectrum_compactness = 1.0  

            return cur_spectrum_compactness

    cpdef int get_number_slots(self, object service, object modulation):
            """
            Computes the number of spectrum slots necessary to accommodate the service request into the path.
            Adds the guardband.
            """
            cdef double required_slots
            required_slots = service.bit_rate / (modulation.spectral_efficiency * self.channel_width)
            return int(math.ceil(required_slots))


    cpdef bint is_path_free(self, object path, int initial_slot, int number_slots):
        cdef int start, end, i, num_nodes, link_index
        cdef int num_spectrum_resources = self.num_spectrum_resources
        cdef cnp.ndarray[cnp.int32_t, ndim=2] available_slots_np = self.topology.graph["available_slots"]
        cdef cnp.int32_t[:, :] available_slots = available_slots_np  # Typed memoryview
        cdef cnp.int32_t[:] slots
        cdef Py_ssize_t slot_idx, num_slots

        if initial_slot + number_slots > num_spectrum_resources:
            return False

        # Considerando a guard band
        if initial_slot > 0:
            start = initial_slot - 1
        else:
            start = 0

        end = initial_slot + number_slots + 1
        if end == num_spectrum_resources:
            end -= 1
        cdef tuple node_list = path.get_node_list()
        num_nodes = len(node_list)
        for i in range(num_nodes - 1):
            # Obtendo o índice do link
            link_data = self.topology[node_list[i]][node_list[i + 1]]
            link_index = link_data["index"]

            # Obtendo os slots como um memoryview para melhorar a performance
            slots = available_slots[link_index, start:end]
            num_slots = slots.shape[0]

            # Verificando se algum slot está ocupado (igual a 0)
            for slot_idx in range(num_slots):
                if slots[slot_idx] == 0:
                    return False

        return True

    cpdef double reward(self):
        cdef double osnr, min_osnr, osnr_diff, numerator, log_value, reward_value
        cdef bint accepted

        accepted = self.current_service.accepted
        if not accepted:
            return 0.0

        osnr = self.current_service.OSNR
        min_osnr = self.current_service.current_modulation.minimum_osnr

        osnr_diff = osnr - min_osnr
        numerator = 1.0 + osnr_diff
        log_value = log10(numerator)
        reward_value = 1.0 - log_value / 1.6

        return reward_value
    
    cpdef _provision_path(self, object path, cnp.int64_t initial_slot, int number_slots):
        cdef int i, path_length, link_index
        cdef int start_slot = <int>initial_slot
        cdef int end_slot = start_slot + number_slots
        cdef tuple node_list = path.get_node_list() 
        cdef object link  # Replace with appropriate type if possible

        # Check if the path is free
        if not self.is_path_free(path, initial_slot, number_slots):
            available_slots = self.get_available_slots(path)
            raise ValueError(
                f"Path {node_list} has not enough capacity on slots {start_slot}-{end_slot} / "
                f"needed: {number_slots} / available: {available_slots}"
            )

        # Provision the path
        path_length = len(node_list)
        for i in range(path_length - 1):
            # Get the link index
            link_index = self.topology[node_list[i]][node_list[i + 1]]["index"]

            # Update available slots
            self.topology.graph["available_slots"][
                link_index,
                start_slot:end_slot
            ] = 0

            # Update spectrum slots allocation
            self.spectrum_slots_allocation[
                link_index,
                start_slot:end_slot
            ] = self.current_service.service_id

            # Append the current service to the link's services
            self.topology[node_list[i]][node_list[i + 1]]["services"].append(self.current_service)
            self.topology[node_list[i]][node_list[i + 1]]["running_services"].append(self.current_service)

        # Update running services in the topology
        self.topology.graph["running_services"].append(self.current_service)

        self.current_service.path = path
        self.current_service.initial_slot = initial_slot
        self.current_service.number_slots = number_slots
        self.current_service.center_frequency = self.frequency_start + (
            self.frequency_slot_bandwidth * initial_slot
        ) + (
            self.frequency_slot_bandwidth * (number_slots / 2.0)
        )
        self.current_service.bandwidth = self.frequency_slot_bandwidth * number_slots

        self.services_accepted += 1
        self.episode_services_accepted += 1
        self.bit_rate_provisioned += self.current_service.bit_rate
        self.episode_bit_rate_provisioned = <cnp.int64_t>(
            self.episode_bit_rate_provisioned + self.current_service.bit_rate
        )

        if self.bit_rate_selection == "discrete":
            self.slots_provisioned_histogram[self.current_service.number_slots] += 1
            self.bit_rate_provisioned_histogram[self.current_service.bit_rate] += 1
            self.episode_bit_rate_provisioned_histogram[self.current_service.bit_rate] += 1

    cpdef void _add_release(self, Service service):
        """
        Adds an event to the event list of the simulator.
        This implementation uses heapq to maintain a min-heap of events.

        :param service: The service that will be released after its holding time.
        :return: None
        """
        cdef double release_time
        release_time = service.arrival_time + service.holding_time
        heapq.heappush(self._events, (release_time, service.service_id, service))
    
    cpdef _release_path(self, Service service):
        cdef int i, link_index
        cdef int initial_slot = service.initial_slot
        cdef int number_slots = service.number_slots
        cdef tuple node_list = service.path.node_list  # Assuming node_list is a tuple or list

        # Iterate over each link in the service's path (node_list)
        for i in range(len(node_list) - 1):
            # Get the index of the link between node i and node i+1
            link_index = self.topology[node_list[i]][node_list[i + 1]]["index"]

            # Free the spectrum slots that were allocated to this service
            self.topology.graph["available_slots"][
                link_index,
                initial_slot : initial_slot + number_slots
            ] = 1

            # Mark the spectrum slots as unallocated (-1)
            self.spectrum_slots_allocation[
                link_index,
                initial_slot : initial_slot + number_slots
            ] = -1

            # Remove the service from the running services on this link
            self.topology[node_list[i]][node_list[i + 1]]["running_services"].remove(service)

            # Update link statistics after releasing the service
            self._update_link_stats(node_list[i], node_list[i + 1])

        # Remove the service from the global list of running services
        self.topology.graph["running_services"].remove(service)

    cpdef _update_link_stats(self, str node1, str node2):
        # Declare todas as variáveis 'cdef' no início da função
        cdef double last_update
        cdef double time_diff
        cdef double last_util
        cdef double cur_util
        cdef double utilization
        cdef double cur_external_fragmentation
        cdef double cur_link_compactness
        cdef double external_fragmentation
        cdef double link_compactness
        cdef int used_spectrum_slots
        cdef int max_empty
        cdef int lambda_min
        cdef int lambda_max
        cdef object link  # Presumindo que é um objeto tipo dict
        cdef cnp.ndarray[cnp.int32_t, ndim=1] slot_allocation  # Array NumPy tipado
        cdef list initial_indices
        cdef list values
        cdef list lengths
        cdef list unused_blocks
        cdef list used_blocks
        cdef double last_external_fragmentation
        cdef double last_compactness
        cdef double sum_1_minus_slot_allocation  # Somatório de (1 - slot_allocation)
        cdef double unused_spectrum_slots
        cdef Py_ssize_t allocation_size  # Declarado no início
        cdef int[:] slot_allocation_view
        cdef int[:] sliced_slot_allocation
        cdef int last_index  # Variável para armazenar o último índice de used_blocks

        # Atualização do link

        # Bloco 1: Inicialização e obtenção do link
        link = self.topology[node1][node2]
        last_update = link["last_update"]

        # Inicializar last_external_fragmentation e last_compactness
        last_external_fragmentation = link.get("external_fragmentation", 0.0)
        last_compactness = link.get("compactness", 0.0)

        # Bloco 2: Cálculos de tempo e utilização
        time_diff = self.current_time - last_update

        if self.current_time > 0:
            last_util = link["utilization"]

            slot_allocation = self.topology.graph["available_slots"][link["index"], :]

            # Convert slot_allocation para um array NumPy compatível com Cython de int32
            slot_allocation = <cnp.ndarray[cnp.int32_t, ndim=1]> np.asarray(slot_allocation, dtype=np.int32)
            slot_allocation_view = slot_allocation

            # Calcular a utilização atual
            used_spectrum_slots = self.num_spectrum_resources - np.sum(slot_allocation)

            # Garantir divisão de ponto flutuante
            cur_util = <double> used_spectrum_slots / self.num_spectrum_resources

            # Atualizar utilização usando uma média ponderada
            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            link["utilization"] = utilization

        # Bloco 3: Run-Length Encoding e cálculos de fragmentação
        # Chamar rle com array NumPy compatível com Cython e converter resultados para listas
        initial_indices_np, values_np, lengths_np = rle(slot_allocation)


        # Verificar sincronização
        if len(initial_indices_np) != len(lengths_np):
            print(f"Error: initial_indices and lengths have different lengths!")
            raise ValueError("initial_indices and lengths have different lengths")

        initial_indices = initial_indices_np.tolist()
        values = values_np.tolist()
        lengths = lengths_np.tolist()

        # Calcular fragmentação externa
        unused_blocks = [i for i, x in enumerate(values) if x == 1]
        if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
            max_empty = max([lengths[i] for i in unused_blocks])
        else:
            max_empty = 0

        if np.sum(slot_allocation) > 0:
            # Calculando a fragmentação externa corretamente
            total_unused_slots = slot_allocation.shape[0] - int(np.sum(slot_allocation))
            cur_external_fragmentation = 1.0 - (<double> max_empty / <double> total_unused_slots)
        else:
            cur_external_fragmentation = 1.0

        # Calcular compactação espectral do link
        used_blocks = [i for i, x in enumerate(values) if x == 0]

        if isinstance(initial_indices, list) and isinstance(lengths, list):
            if len(used_blocks) > 1:
                # Verificar se used_blocks contém índices válidos
                valid = True
                for idx in used_blocks:
                    if not isinstance(idx, int):
                        valid = False
                        break
                    if idx < 0 or idx >= len(initial_indices):
                        valid = False
                        break
                if not valid:
                    raise IndexError("Invalid indices in used_blocks")

                # Substituir usado_blocks[-1] por usado_blocks[last_index]
                last_index = len(used_blocks) - 1
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[last_index]] + lengths[used_blocks[last_index]]
                # Garantir que lambda_min e lambda_max estão dentro dos limites
                allocation_size = slot_allocation.shape[0]  # Mantém como inteiro

                if lambda_min < 0 or lambda_max > allocation_size:
                    print(f"Error: lambda_min ({lambda_min}) or lambda_max ({lambda_max}) out of bounds for slot_allocation size {allocation_size}")
                    raise IndexError("lambda_min ou lambda_max fora dos limites")

                if lambda_min >= lambda_max:
                    print(f"Error: lambda_min ({lambda_min}) >= lambda_max ({lambda_max})")
                    raise ValueError("lambda_min >= lambda_max")

                # Slicing usando memory views
                sliced_slot_allocation = slot_allocation_view[lambda_min:lambda_max]
                sliced_slot_allocation_np = np.asarray(sliced_slot_allocation)  # Converter para numpy.ndarray

                # Avaliar a parte usada do espectro
                internal_idx_np, internal_values_np, internal_lengths_np = rle(sliced_slot_allocation_np)

                internal_values = internal_values_np.tolist()  # Converter para listas
                unused_spectrum_slots = <double> np.sum(1 - internal_values_np)

                sum_1_minus_slot_allocation = <double> np.sum(1 - slot_allocation)

                if unused_spectrum_slots > 0 and sum_1_minus_slot_allocation > 0:
                    cur_link_compactness = ((<double> (lambda_max - lambda_min)) / sum_1_minus_slot_allocation) * (1.0 / unused_spectrum_slots)
                else:
                    cur_link_compactness = 1.0
            else:
                cur_link_compactness = 1.0
        else:
            print(f"Error: initial_indices or lengths are not lists/arrays!")
            raise TypeError("initial_indices or lengths are not lists/arrays")


        # Atualizar fragmentação externa usando uma média ponderada
        external_fragmentation = ((last_external_fragmentation * last_update) + (cur_external_fragmentation * time_diff)) / self.current_time
        link["external_fragmentation"] = external_fragmentation

        link_compactness = ((last_compactness * last_update) + (cur_link_compactness * time_diff)) / self.current_time
        link["compactness"] = link_compactness

        link["last_update"] = self.current_time

    cpdef cnp.ndarray get_available_slots(self, object path):
        """
        Compute the available slots by element-wise multiplying the relevant rows
        from the available_slots matrix in the topology graph.
        """
        cdef Py_ssize_t i, n
        cdef tuple node_list = path.node_list
        cdef list indices
        cdef cnp.ndarray available_slots_matrix
        cdef cnp.ndarray product
        cdef long[:, :] slots_view
        cdef long[:] product_view
        cdef Py_ssize_t num_rows, num_cols

        n = len(node_list) - 1

        indices = [0] * n

        for i in range(n):
            indices[i] = self.topology[node_list[i]][node_list[i + 1]]["index"]

        available_slots_matrix = self.topology.graph["available_slots"][indices, :]

        num_rows = available_slots_matrix.shape[0]
        num_cols = available_slots_matrix.shape[1]

        slots_view = available_slots_matrix

        product = available_slots_matrix[0].copy()
        product_view = product
        for i in range(1, num_rows):
            for j in range(num_cols):
                product_view[j] *= slots_view[i, j]

        return product

    cpdef tuple get_available_blocks(self, int path, int slots, j):
        cdef cnp.ndarray available_slots = self.get_available_slots(
            self.k_shortest_paths[
                self.current_service.source, 
                self.current_service.destination
            ][path]
        )
        cdef cnp.ndarray initial_indices, values, lengths

        initial_indices, values, lengths = rle(available_slots)

        cdef cnp.ndarray available_indices_np = np.where(values == 1)[0]
        cdef cnp.ndarray sufficient_indices_np = np.where(lengths >= slots)[0]
        cdef cnp.ndarray final_indices_np = np.intersect1d(available_indices_np, sufficient_indices_np)[:j]

        return initial_indices[final_indices_np], lengths[final_indices_np]

    def close(self):
        return super().close()
    
    def normalize(self, x, x_min, x_max):
        return ((x - x_min) / (x_max - x_min)) * 2 - 1
