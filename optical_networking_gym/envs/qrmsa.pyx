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
    cdef public int k_paths
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
    cdef public int reject_action
    cdef object actions_output
    cdef object actions_taken
    cdef bint _new_service
    cdef public double current_time
    cdef double mean_service_inter_arrival_time
    cdef public object frequency_vector
    cdef object rng
    cdef object bit_rate_function
    cdef list _events
    cdef object file_stats
    cdef unicode final_file_name
    cdef int blocks_to_consider
    cdef int bl_resource 
    cdef int bl_osnr 
    cdef int bl_reject
    cdef public int max_modulation_idx
    cdef public int modulations_to_consider
    cdef int spectrum_efficiency_metric

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
        allow_rejection: bool = True,
        reset: bool = True,
        channel_width: double = 12.5,
        k_paths: int = 5,
        file_name: str = "",
        blocks_to_consider: int = 1,
        modulations_to_consider: int = 6,
    ):
        self.rng = random.Random()
        self.blocks_to_consider = blocks_to_consider
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
            self.slots_provisioned_histogram = defaultdict(int)
            self.episode_slots_provisioned_histogram = defaultdict(int)
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
        self.max_modulation_idx = len(self.modulations) - 1
        self.modulations_to_consider = min(modulations_to_consider, len(self.modulations))
        self.disrupted_services_list = []
        self.disrupted_services = 0
        self.episode_disrupted_services = 0

        # Redefinir o espaço de ações para Discrete
        self.action_space = gym.spaces.Discrete(
            (self.k_paths * self.modulations_to_consider * self.num_spectrum_resources)+1
        )

        total_dim = (
            1
            + 2
            + self.k_paths
            + (self.k_paths * self.modulations_to_consider * 12)
        )

        self.observation_space = gym.spaces.Box(
                low=-5,
                high=5,
                shape=(total_dim,),
                dtype=np.float32
            )
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
            self.bit_rate_requested_histogram = defaultdict(int)
            self.bit_rate_provisioned_histogram = defaultdict(int)
            self.slots_provisioned_histogram = defaultdict(int)
            self.episode_slots_provisioned_histogram = defaultdict(int)
        else:
            self.bit_rate_requested_histogram = None
            self.bit_rate_provisioned_histogram = None
        self.reject_action = self.action_space.n - 1 if allow_rejection else 0
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

            self.file_stats.write("# Service stats file from simulator\n")
            self.file_stats.write("id,source,destination,bit_rate,path_k,path_length,modulation,min_osnr,osnr,ase,nli,disrupted_services\n")
        else:
            self.file_stats = None

        self.bl_osnr = 0
        self.bl_resource = 0
        self.bl_reject = 0
        self.spectrum_efficiency_metric = 0
        if reset:
            self.reset()

    cpdef tuple reset(self, object seed=None, dict options=None):
        self.episode_bit_rate_requested = 0.0
        self.episode_bit_rate_provisioned = 0.0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_disrupted_services = 0
        self._events = []
        self.bl_resource = 0
        self.bl_osnr = 0
        self.bl_reject = 0
        self.max_modulation_idx = len(self.modulations) - 1

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

        if options is not None and "only_episode_counters" in options and options["only_episode_counters"]:
            observation, mask = self.observation()
            info = {}
            return observation, info

        self.bit_rate_requested = 0.0
        self.bit_rate_provisioned = 0.0
        self.disrupted_services = 0
        self.disrupted_services_list = []

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

        observation, mask = self.observation()
        info = mask.copy()
        return observation, info
    
    def normalize_value(self, value, min_v, max_v):
        """
        Normaliza um valor no intervalo [0,1]. 
        Se max_v == min_v, retorna 0 para evitar divisão por zero.
        """
        if max_v == min_v:
            return 0.0
        return (value - min_v) / (max_v - min_v)
    
    def _get_candidates(self, available_slots, num_slots_required, total_slots):
        """
        Gera todos os candidatos (initial slot) para alocação, usando RLE.
        Se o bloco se estende até o final, não exige guard band; caso contrário,
        exige num_slots_required+1 slots.
        
        Args:
            available_slots (np.array): Vetor com slots disponíveis (1) e indisponíveis (0).
            num_slots_required (int): Número de slots necessários para o serviço.
            total_slots (int): Número total de slots no espectro.
        
        Returns:
            list: Lista de candidatos (índices) válidos.
        """
        initial_indices, values, lengths = rle(available_slots)
        candidates = []
        for start, val, length in zip(initial_indices, values, lengths):
            if val == 1:
                if start + length == total_slots:
                    if length >= num_slots_required:
                        for candidate in range(start, start + length - num_slots_required + 1):
                            candidates.append(candidate)
                else:
                    if length >= (num_slots_required + 1):
                        for candidate in range(start, start + length - (num_slots_required + 1) + 1):
                            candidates.append(candidate)
        return candidates

    def get_max_modulation_index(self):
        """
        Atualiza self.max_modulation_idx com a melhor modulação (maior índice) cuja alocação
        apresenta OSNR aceitável (>= limiar + margin). Usa a mesma lógica de geração de candidatos.
        """
        for path in self.k_shortest_paths[self.current_service.source, self.current_service.destination]:
            available_slots = self.get_available_slots(path)
            
            for idm, modulation in enumerate(reversed(self.modulations)):
                number_slots = self.get_number_slots(self.current_service, modulation)
                candidatos = self._get_candidates(available_slots, number_slots, self.num_spectrum_resources)
                
                if candidatos:
                    for candidate in candidatos:
                        self.current_service.path = path
                        self.current_service.initial_slot = candidate
                        self.current_service.number_slots = number_slots
                        self.current_service.center_frequency = (
                            self.frequency_start +
                            self.frequency_slot_bandwidth * (candidate + number_slots / 2)
                        )
                        self.current_service.bandwidth = self.frequency_slot_bandwidth * number_slots
                        self.current_service.launch_power = self.launch_power
                        self.current_service.blocked_due_to_resources = False

                        osnr, _, _ = calculate_osnr(self, self.current_service)

                        self.current_service.path = None
                        self.current_service.initial_slot = -1
                        self.current_service.number_slots = 0
                        self.current_service.center_frequency = 0.0
                        self.current_service.bandwidth = 0.0
                        self.current_service.launch_power = 0.0

                        if osnr >= modulation.minimum_osnr + self.margin:
                            self.max_modulation_idx = max(len(self.modulations) - idm - 1,
                                                        self.modulations_to_consider - 1)
                            return
        self.max_modulation_idx = self.modulations_to_consider - 1

    def observation(self):
        def compute_modulation_features(available_slots, num_slots_required, route_links, modulation):
            valid_starts = self._get_candidates(available_slots, num_slots_required, self.num_spectrum_resources)
            candidate_count = len(valid_starts)
            feature_candidate_count = candidate_count / self.num_spectrum_resources
            if candidate_count > 0:
                avg_candidate = np.mean(valid_starts)
                std_candidate = np.std(valid_starts)
                max_candidate = max(valid_starts)
            else:
                avg_candidate = std_candidate = max_candidate = 0.0
            feature_avg_candidate = avg_candidate / (self.num_spectrum_resources - 1)
            feature_std_candidate = std_candidate / (self.num_spectrum_resources - 1)
            feature_max_candidate = max_candidate / (self.num_spectrum_resources - 1)
            
            osnr_values = []
            osnr_best = 0.0
            for init_slot in valid_starts:
                service_bandwidth = num_slots_required * self.channel_width * 1e9
                service_center_frequency = (
                    self.frequency_start +
                    (self.channel_width * 1e9 * init_slot) +
                    (self.channel_width * 1e9 * (num_slots_required / 2.0))
                )
                osnr_current = calculate_osnr_observation(
                    self,
                    route_links,
                    service_bandwidth,
                    service_center_frequency,
                    self.current_service.service_id,
                    10 ** ((self.launch_power_dbm - 30) / 10),
                    modulation.minimum_osnr
                )
                osnr_values.append(osnr_current)
                if osnr_current > osnr_best:
                    osnr_best = osnr_current
            if osnr_values:
                osnr_mean = np.mean(osnr_values)
                osnr_var = np.var(osnr_values)
            else:
                osnr_mean = osnr_var = 0.0
            
            adjusted_slots_required = max((num_slots_required - 5.5) / 3.5, 0.0)
            
            total_available_slots = np.sum(available_slots)
            total_available_slots_ratio = 2.0 * (total_available_slots - 0.5 * self.num_spectrum_resources) / self.num_spectrum_resources
            
            blocks_sizes = []
            current_len = 0
            for slot in available_slots:
                if slot == 1:
                    current_len += 1
                else:
                    if current_len > 0:
                        blocks_sizes.append(current_len)
                    current_len = 0
            if current_len > 0:
                blocks_sizes.append(current_len)
            if blocks_sizes:
                mean_block_size = ((np.mean(blocks_sizes) - 4.0) / 4.0) / 100.0
                std_block_size = (np.std(blocks_sizes) / 100.0)
            else:
                mean_block_size = std_block_size = 0.0
            
            link_usage_normalized = 2.0 * ((np.sum(available_slots) / self.num_spectrum_resources) - 0.5)
            
            features = [feature_candidate_count,
                        feature_avg_candidate,
                        feature_std_candidate,
                        adjusted_slots_required,
                        total_available_slots_ratio,
                        mean_block_size,
                        std_block_size,
                        osnr_best,
                        osnr_mean,
                        osnr_var,
                        link_usage_normalized,
                        feature_max_candidate]
            return valid_starts, features, osnr_values

        # ========================
        # Observações comuns
        # ========================
        topology = self.topology
        current_service = self.current_service
        num_spectrum_resources = self.num_spectrum_resources
        k_shortest_paths = self.k_shortest_paths
        modulations = self.modulations
        num_mod_to_consider = self.modulations_to_consider
        num_nodes = topology.number_of_nodes()
        frequency_slot_bandwidth = self.channel_width * 1e9
        max_bit_rate = max(self.bit_rates)
        self.get_max_modulation_index()

        source_id = int(current_service.source_id)
        destination_id = int(current_service.destination_id)
        source_norm = source_id / (num_nodes - 1) if num_nodes > 1 else 0
        destination_norm = destination_id / (num_nodes - 1) if num_nodes > 1 else 0
        source_destination = np.array([source_norm, destination_norm], dtype=np.float32)

        bit_rate_obs = np.array([current_service.bit_rate / max_bit_rate], dtype=np.float32)

        num_paths_to_evaluate = self.k_paths
        # Obter os comprimentos dos links para normalização das rotas
        link_lengths = [topology[x][y]["length"] for x, y in topology.edges()]
        min_length = min(link_lengths)
        max_length = max(link_lengths)
        route_lengths = np.zeros((num_paths_to_evaluate,), dtype=np.float32)

        # Pré-cálculo das informações de cada caminho: (route, available_slots)
        paths_info = []
        source = current_service.source
        destination = current_service.destination
        for path_index, route in enumerate(k_shortest_paths[source, destination]):
            if path_index >= num_paths_to_evaluate:
                break
            normalized_length = self.normalize_value(route.length, min_length, max_length)
            route_lengths[path_index] = normalized_length
            available_slots = self.get_available_slots(route)
            paths_info.append((route, available_slots))

        # ========================
        # Cálculo das features de observação por (caminho, modulação)
        # ========================
        mod_features_obs = np.full((num_paths_to_evaluate * num_mod_to_consider, 12), fill_value=-1.0, dtype=np.float32)
        mod_features_cache = {}
        for p_idx in range(num_paths_to_evaluate):
            route, available_slots = paths_info[p_idx]
            start_index = 0 if self.max_modulation_idx <= 1 else max(0, self.max_modulation_idx - (num_mod_to_consider - 1))
            mod_list = list(reversed(modulations[start_index: num_mod_to_consider + start_index]))
            for m_idx in range(num_mod_to_consider):
                modulation = mod_list[m_idx]
                num_slots_required = self.get_number_slots(current_service, modulation)
                valid_starts, features, osnr_values = compute_modulation_features(available_slots, num_slots_required, route.links, modulation)
                mod_features_obs[p_idx * num_mod_to_consider + m_idx, :] = np.array(features, dtype=np.float32)
                mod_features_cache[(p_idx, m_idx)] = (valid_starts, features, num_slots_required, modulation)

        # ========================
        # Geração da máscara de ações (permanece inalterada)
        # ========================
        total_actions = num_paths_to_evaluate * num_mod_to_consider * num_spectrum_resources
        action_mask = np.zeros(total_actions + 1, dtype=np.uint8)

        for action_index in range(total_actions):
            p_idx = action_index // (num_mod_to_consider * num_spectrum_resources)
            mod_and_slot = action_index % (num_mod_to_consider * num_spectrum_resources)
            m_idx = mod_and_slot // num_spectrum_resources
            init_slot = mod_and_slot % num_spectrum_resources

            route, available_slots = paths_info[p_idx]
            # Recupera os dados de modulação do cache
            valid_starts, features, num_slots_required, modulation = mod_features_cache[(p_idx, m_idx)]

            valid_action = False
            osnr_current = 0.0
            if init_slot in valid_starts:
                service_bandwidth = num_slots_required * frequency_slot_bandwidth
                service_center_frequency = (
                    self.frequency_start +
                    (frequency_slot_bandwidth * init_slot) +
                    (frequency_slot_bandwidth * (num_slots_required / 2.0))
                )
                osnr_current = calculate_osnr_observation(
                    self,
                    route.links,
                    service_bandwidth,
                    service_center_frequency,
                    current_service.service_id,
                    10 ** ((self.launch_power_dbm - 30) / 10),
                    modulation.minimum_osnr
                )
                if osnr_current >= 0:
                    valid_action = True

            if valid_action:
                action_mask[action_index] = 1

        # Define a ação dummy (última posição) como válida
        action_mask[-1] = 1

        # ========================
        # Construção final da observação
        # ========================
        # As features de ação agora vêm de mod_features_obs, que tem dimensão:
        # (k_paths * modulations_to_consider, 12)
        spectrum_obs_flat = mod_features_obs.flatten().astype(np.float32)
        observation = np.concatenate([
            bit_rate_obs,                         # 1 valor
            source_destination.flatten(),         # 2 valores
            route_lengths.flatten(),              # k_paths valores
            spectrum_obs_flat                     # (k_paths * modulations_to_consider * 12) valores
        ], axis=0).astype(np.float32)

        return observation, {'mask': action_mask}



    def decimal_to_array(self, decimal: int, max_values: list[int] = None) -> list[int]:
        if max_values is None:
            max_values = [self.k_paths, self.modulations_to_consider, self.num_spectrum_resources]       
        array = []
        for max_val in reversed(max_values):
            digit = decimal % max_val
            array.insert(0, digit)
            decimal //= max_val
        if self.max_modulation_idx > 1:
            allowed_mods = list(range(self.max_modulation_idx, self.max_modulation_idx - (self.modulations_to_consider - 1), -1))
        else:
            allowed_mods = list(range(0, self.modulations_to_consider))
        array[1] = allowed_mods[array[1]]
        # print(f"k:{self.k_shortest_paths[self.current_service.source,self.current_service.destination][array[0]]}, mod: {self.modulations[array[1]]}, slot: {array[2]}")
        return array

    def encoded_decimal_to_array(self, decimal: int, max_values: list[int] = None) -> list[int]:
        # Usa divisão inteira para garantir um valor inteiro para part_size.
        part_size = self.num_spectrum_resources // self.modulations_to_consider  
        mod_idx = decimal // part_size  # Calculado mas não usado para modulação
        # print(f"Input decimal: {decimal}")
        
        if max_values is None:
            max_values = [self.k_paths, self.modulations_to_consider, self.num_spectrum_resources]
        # print(f"Max values: {max_values}")
        
        array = []
        # Decomposição do número decimal com base nos valores máximos (ordem: [k_path, modulação, slot])
        for max_val in reversed(max_values):
            digit = decimal % max_val
            # print(f"Current max_val: {max_val}, digit: {digit}")
            array.insert(0, digit)
            decimal //= max_val
            # print(f"Updated decimal: {decimal}")
        
        # Cria a lista de modulações permitidas
        if self.max_modulation_idx > 1:
            # Remove o 'reversed' para manter a ordem desejada: maior para menor.
            allowed_mods = list(range(self.max_modulation_idx, self.max_modulation_idx - self.modulations_to_consider, -1))
        else:
            allowed_mods = list(reversed(list(range(0, self.modulations_to_consider))))
        # print(f"Allowed mods: {allowed_mods}")
        
        # Atualiza o dígito de modulação usando o dígito extraído da decomposição
        array[1] = allowed_mods[array[1]]
        # print(f"Decoded array: {array}")
        # print(f"k: {self.k_shortest_paths[self.current_service.source, self.current_service.destination][array[0]]}, "
        #     f"mod: {self.modulations[array[1]]}, slot: {array[2]}")
        
        return array



    cpdef tuple[object, float, bint, bint, dict] step(self, int action):
        cdef int route = -1
        cdef int modulation_idx = -1
        cdef int initial_slot = -1
        cdef int number_slots = 0

        cdef double osnr = 0.0
        cdef double ase = 0.0
        cdef double nli = 0.0
        cdef double osnr_req = 0.0

        cdef bint truncated = False
        cdef bint terminated
        cdef int disrupted_services = 0

        cdef object modulation = None
        cdef object path = None
        cdef list services_to_measure = []
        cdef dict info

        self.current_service.blocked_due_to_resources = False
        self.current_service.blocked_due_to_osnr = False

        if action == (self.action_space.n - 1):
            self.current_service.accepted = False
            self.current_service.blocked_due_to_resources = False
            self.current_service.blocked_due_to_osnr = False
            self.bl_reject += 1
        else:
            decoded = self.encoded_decimal_to_array(
                action,
                [self.k_paths, self.modulations_to_consider, self.num_spectrum_resources]
            )
            route = decoded[0]
            modulation_idx = decoded[1]
            initial_slot = decoded[2]
            modulation = self.modulations[modulation_idx]
            osnr_req = modulation.minimum_osnr + self.margin

            path = self.k_shortest_paths[
                self.current_service.source,
                self.current_service.destination
            ][route]

            number_slots = self.get_number_slots(
                service=self.current_service,
                modulation=modulation
            )

            if self.is_path_free(path=path, initial_slot=initial_slot, number_slots=number_slots):
                self.current_service.path = path
                self.current_service.initial_slot = initial_slot
                self.current_service.number_slots = number_slots
                self.current_service.center_frequency = (
                    self.frequency_start
                    + (self.frequency_slot_bandwidth * initial_slot)
                    + (self.frequency_slot_bandwidth * (number_slots / 2.0))
                )
                self.current_service.bandwidth = self.frequency_slot_bandwidth * number_slots
                self.current_service.launch_power = self.launch_power

                osnr, ase, nli = calculate_osnr(self, self.current_service)
                if osnr >= osnr_req:
                    self.current_service.accepted = True
                    self.current_service.OSNR = osnr
                    self.current_service.ASE = ase
                    self.current_service.NLI = nli
                    self.current_service.current_modulation = modulation
                    self.spectrum_efficiency_metric += modulation.spectral_efficiency
                    self.episode_modulation_histogram[modulation.spectral_efficiency] += 1
                    self._provision_path(path, initial_slot, number_slots)

                    if self.bit_rate_selection == "discrete":
                        self.slots_provisioned_histogram[number_slots] += 1

                    self._add_release(self.current_service)
                else:
                    self.current_service.accepted = False
                    self.current_service.blocked_due_to_osnr = True
                    self.bl_osnr += 1
            else:
                self.current_service.accepted = False
                self.current_service.blocked_due_to_resources = True
                self.bl_resource += 1

        if self.measure_disruptions and self.current_service.accepted:
            services_to_measure = []
            for link in self.current_service.path.links:
                for service_in_link in self.topology[link.node1][link.node2]["running_services"]:
                    if (service_in_link not in services_to_measure
                            and service_in_link not in self.disrupted_services_list):
                        services_to_measure.append(service_in_link)

            for svc in services_to_measure:
                osnr_svc, ase_svc, nli_svc = calculate_osnr(self, svc)
                if osnr_svc < svc.current_modulation.minimum_osnr:
                    disrupted_services += 1
                    if svc not in self.disrupted_services_list:
                        self.disrupted_services += 1
                        self.episode_disrupted_services += 1
                        self.disrupted_services_list.append(svc)

        if not self.current_service.accepted:
            if action == (self.action_space.n - 1):
                self.actions_taken[self.k_paths, self.num_spectrum_resources] += 1
            else:
                self.actions_taken[self.k_paths, self.num_spectrum_resources] += 1

            self.current_service.path = None
            self.current_service.initial_slot = -1
            self.current_service.number_slots = 0
            self.current_service.OSNR = 0.0
            self.current_service.ASE = 0.0
            self.current_service.NLI = 0.0

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
            self.file_stats.flush()

        if action != (self.action_space.n - 1):
            reward = self.reward()
        else:
            reward = -3.0
        info = {
            "episode_services_accepted": self.episode_services_accepted,
            "service_blocking_rate": 0.0,
            "episode_service_blocking_rate": 0.0,
            "bit_rate_blocking_rate": 0.0,
            "episode_bit_rate_blocking_rate": 0.0,
            "disrupted_services": 0.0,
            "episode_disrupted_services": 0.0,
            "osnr": osnr,
            "osnr_req": osnr_req,
            "chosen_path_index": route,
            "chosen_slot": initial_slot,
        }

        if self.services_processed > 0:
            info["service_blocking_rate"] = float(
                self.services_processed - self.services_accepted
            ) / self.services_processed

        if self.episode_services_processed > 0:
            info["episode_service_blocking_rate"] = float(
                self.episode_services_processed - self.episode_services_accepted
            ) / self.episode_services_processed
            info["episode_service_blocking_rate"] = (
                float(self.episode_services_processed - self.episode_services_accepted)
            ) / float(self.episode_services_processed)

        if self.bit_rate_requested > 0:
            info["bit_rate_blocking_rate"] = float(
                self.bit_rate_requested - self.bit_rate_provisioned
            ) / self.bit_rate_requested

        if self.episode_bit_rate_requested > 0:
            info["episode_bit_rate_blocking_rate"] = float(
                self.episode_bit_rate_requested - self.episode_bit_rate_provisioned
            ) / self.episode_bit_rate_requested

        if self.disrupted_services > 0 and self.services_accepted > 0:
            info["disrupted_services"] = float(self.disrupted_services) / self.services_accepted

        if self.episode_disrupted_services > 0 and self.episode_services_accepted > 0:
            info["episode_disrupted_services"] = float(
                self.episode_disrupted_services / self.episode_services_accepted
            )

        cdef float spectral_eff
        for current_modulation in self.modulations:
            spectral_eff = current_modulation.spectral_efficiency
            key = "modulation_{}".format(str(spectral_eff))
            if spectral_eff in self.episode_modulation_histogram:
                info[key] = self.episode_modulation_histogram[spectral_eff]
            else:
                info[key] = 0

        self._new_service = False
        self._next_service()

        terminated = (self.episode_services_processed == self.episode_length)
        if terminated:
            info["blocked_due_to_resources"] = self.bl_resource
            info["blocked_due_to_osnr"] = self.bl_osnr
            info["rejected"] = self.bl_reject

        observation, mask = self.observation()
        info.update(mask)

        return (observation, reward, terminated, truncated, info)

    cpdef _next_service(self):
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
            cdef double required_slots
            required_slots = service.bit_rate / (modulation.spectral_efficiency * self.channel_width)
            return int(math.ceil(required_slots))


    def is_path_free(self, path: Path, initial_slot: int, number_slots: int) -> bool:
        end = initial_slot + number_slots
        if end  > self.num_spectrum_resources:
            return False
        start = initial_slot 
        if end < self.num_spectrum_resources:
            end +=1
        for i in range(len(path.node_list) - 1):
            if np.any(
                self.topology.graph["available_slots"][
                    self.topology[path.node_list[i]][path.node_list[i + 1]]["index"],
                    start : end
                ]
                == 0
            ):
                return False
        return True

    cpdef double reward(self):
        cdef double base_reward = 0.0
        if self.current_service.accepted:
            base_reward = 1.0
            base_reward += 0.1 * self.current_service.current_modulation.spectral_efficiency

        else:
            base_reward = -1.5
            
        return base_reward
            

    
    cpdef _provision_path(self, object path, cnp.int64_t initial_slot, int number_slots):
        cdef int i, path_length, link_index
        cdef int start_slot = initial_slot
        cdef int end_slot = start_slot + number_slots
        cdef tuple node_list = path.get_node_list() 
        cdef object link  

        if end_slot < self.num_spectrum_resources:
            end_slot+=1
        elif end_slot > self.num_spectrum_resources:
            raise ValueError("End slot is greater than the number of spectrum resources.")
            
        path_length = len(node_list)
        for i in range(path_length - 1):
            link_index = self.topology[node_list[i]][node_list[i + 1]]["index"]
            self.topology.graph["available_slots"][
                link_index,
                start_slot:end_slot
            ] = 0

            self.spectrum_slots_allocation[
                link_index,
                start_slot:end_slot
            ] = self.current_service.service_id

            self.topology[node_list[i]][node_list[i + 1]]["services"].append(self.current_service)
            self.topology[node_list[i]][node_list[i + 1]]["running_services"].append(self.current_service)

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
        cdef double release_time
        release_time = service.arrival_time + service.holding_time
        heapq.heappush(self._events, (release_time, service.service_id, service))
    
    def _release_path(self, service: Service):
        for i in range(len(service.path.node_list) - 1):
            self.topology.graph["available_slots"][
                self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                    "index"
                ],
                service.initial_slot : service.initial_slot + service.number_slots+1,
            ] = 1
            self.spectrum_slots_allocation[
                self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                    "index"
                ],
                service.initial_slot : service.initial_slot + service.number_slots+1,
            ] = -1
            self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                "running_services"
            ].remove(service)

        self.topology.graph["running_services"].remove(service)


    cpdef _update_link_stats(self, str node1, str node2):
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
        cdef object link
        cdef cnp.ndarray[cnp.int32_t, ndim=1] slot_allocation
        cdef list initial_indices
        cdef list values
        cdef list lengths
        cdef list unused_blocks
        cdef list used_blocks
        cdef double last_external_fragmentation
        cdef double last_compactness
        cdef double sum_1_minus_slot_allocation
        cdef double unused_spectrum_slots
        cdef Py_ssize_t allocation_size
        cdef int[:] slot_allocation_view
        cdef int[:] sliced_slot_allocation
        cdef int last_index

        link = self.topology[node1][node2]
        last_update = link["last_update"]

        last_external_fragmentation = link.get("external_fragmentation", 0.0)
        last_compactness = link.get("compactness", 0.0)

        time_diff = self.current_time - last_update

        if self.current_time > 0:
            last_util = link["utilization"]

            slot_allocation = self.topology.graph["available_slots"][link["index"], :]

            slot_allocation = <cnp.ndarray[cnp.int32_t, ndim=1]> np.asarray(slot_allocation, dtype=np.int32)
            slot_allocation_view = slot_allocation

            used_spectrum_slots = self.num_spectrum_resources - np.sum(slot_allocation)

            cur_util = <double> used_spectrum_slots / self.num_spectrum_resources

            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            link["utilization"] = utilization

        initial_indices_np, values_np, lengths_np = rle(slot_allocation)

        if len(initial_indices_np) != len(lengths_np):
            raise ValueError("initial_indices and lengths have different lengths")

        initial_indices = initial_indices_np.tolist()
        values = values_np.tolist()
        lengths = lengths_np.tolist()

        unused_blocks = [i for i, x in enumerate(values) if x == 1]
        if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
            max_empty = max([lengths[i] for i in unused_blocks])
        else:
            max_empty = 0

        if np.sum(slot_allocation) > 0:
            total_unused_slots = slot_allocation.shape[0] - int(np.sum(slot_allocation))
            cur_external_fragmentation = 1.0 - (<double> max_empty / <double> total_unused_slots)
        else:
            cur_external_fragmentation = 1.0

        used_blocks = [i for i, x in enumerate(values) if x == 0]

        if isinstance(initial_indices, list) and isinstance(lengths, list):
            if len(used_blocks) > 1:
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

                last_index = len(used_blocks) - 1
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[last_index]] + lengths[used_blocks[last_index]]

                allocation_size = slot_allocation.shape[0]

                if lambda_min < 0 or lambda_max > allocation_size:
                    raise IndexError("lambda_min ou lambda_max fora dos limites")

                if lambda_min >= lambda_max:
                    raise ValueError("lambda_min >= lambda_max")

                sliced_slot_allocation = slot_allocation_view[lambda_min:lambda_max]
                sliced_slot_allocation_np = np.asarray(sliced_slot_allocation)

                internal_idx_np, internal_values_np, internal_lengths_np = rle(sliced_slot_allocation_np)

                internal_values = internal_values_np.tolist()
                unused_spectrum_slots = <double> np.sum(1 - internal_values_np)

                sum_1_minus_slot_allocation = <double> np.sum(1 - slot_allocation)

                if unused_spectrum_slots > 0 and sum_1_minus_slot_allocation > 0:
                    cur_link_compactness = ((<double> (lambda_max - lambda_min)) / sum_1_minus_slot_allocation) * (1.0 / unused_spectrum_slots)
                else:
                    cur_link_compactness = 1.0
            else:
                cur_link_compactness = 1.0
        else:
            raise TypeError("initial_indices or lengths are not lists/arrays")


        external_fragmentation = ((last_external_fragmentation * last_update) + (cur_external_fragmentation * time_diff)) / self.current_time
        link["external_fragmentation"] = external_fragmentation

        link_compactness = ((last_compactness * last_update) + (cur_link_compactness * time_diff)) / self.current_time
        link["compactness"] = link_compactness

        link["last_update"] = self.current_time

    cpdef cnp.ndarray get_available_slots(self, object path):
        cdef Py_ssize_t i, n
        cdef tuple node_list = path.node_list
        cdef list indices
        cdef cnp.ndarray available_slots_matrix
        cdef cnp.ndarray product
        cdef int[:, :] slots_view
        cdef int[:] product_view
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
\