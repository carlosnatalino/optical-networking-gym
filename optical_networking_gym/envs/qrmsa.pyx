from typing import Any, Literal, Sequence, SupportsFloat, Optional
from dataclasses import field

cimport cython
cimport numpy as cnp
from libc.stdint cimport uint32_t
from libc.math cimport log, exp, asinh, log10
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
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
from optical_networking_gym.utils import rle, link_shannon_entropy_, fragmentation_route_cuts, fragmentation_route_rss
from optical_networking_gym.core.osnr import calculate_osnr, calculate_osnr_observation, compute_slot_osnr_vectorized, validate_osnr_vectorized
import math
import typing
import os
from scipy.signal import convolve

if typing.TYPE_CHECKING:
    from optical_networking_gym.topology import Link, Span, Modulation, Path

cdef class LinkCacheManager:
    cdef dict _cache
    cdef object topology
    
    def __init__(self, topology):
        self.topology = topology
        self._cache = {}
        self._initialize_cache()
    
    cdef void _initialize_cache(self):
        for edge in self.topology.edges():
            key = (edge[0], edge[1])
            edge_data = self.topology[edge[0]][edge[1]]
            self._cache[key] = {
                'index': edge_data.get('index', 0),
                'services': edge_data.get('services', []),
                'running_services': edge_data.get('running_services', []),
                'utilization': edge_data.get('utilization', 0.0),
                'last_update': edge_data.get('last_update', 0.0),
                'external_fragmentation': edge_data.get('external_fragmentation', 0.0),
                'compactness': edge_data.get('compactness', 0.0)
            }
    
    cdef dict get_link_data(self, str node1, str node2):
        key = (node1, node2)
        if key not in self._cache:
            key = (node2, node1)
        return self._cache[key]
    
    cdef void update_link_services(self, str node1, str node2, object service, bint add=True):
        key = (node1, node2)
        if key not in self._cache:
            key = (node2, node1)
        
        if add:
            self._cache[key]['services'].append(service)
            self._cache[key]['running_services'].append(service)
            self.topology[node1][node2]['services'].append(service)
            self.topology[node1][node2]['running_services'].append(service)
        else:
            if service in self._cache[key]['running_services']:
                self._cache[key]['running_services'].remove(service)
                self.topology[node1][node2]['running_services'].remove(service)

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
    cdef public int episode_length
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
    cdef public str qot_constraint
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

    cdef LinkCacheManager link_cache
    cdef dict modulation_slots_cache
    cdef cnp.int64_t[:, :] topology_indices_cache  
    cdef dict path_nodes_cache 
    cdef public int max_modulation_idx
    cdef public int modulations_to_consider
    
    # Novos caches para observações slot-wise
    cdef dict available_slots_cache
    cdef dict available_slots_signature_cache
    cdef dict block_info_cache
    cdef dict osnr_matrix_cache
    cdef dict fragmentation_cache
    cdef dict slot_window_cache
    cdef int spectrum_efficiency_metric
    cdef bint defragmentation
    cdef int n_defrag_services
    cdef int episode_defrag_cicles
    cdef int episode_service_realocations
    cdef bint gen_observation
    cdef public bint rl_mode

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
        defragmentation: bool = False,
        n_defrag_services: int = 0,
        gen_observation: bool = True,
        qot_constraint: str = "ASE+NLI",
        rl_mode: bool = False,
    ):
        self.gen_observation = gen_observation
        self.defragmentation = defragmentation
        self.n_defrag_services = n_defrag_services
        self.rl_mode = rl_mode
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
        self.qot_constraint = qot_constraint
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

        self.action_space = gym.spaces.Discrete(
            (self.k_paths * self.modulations_to_consider * self.num_spectrum_resources)+1
        )

        # Nova estrutura otimizada: 3 + k_paths + (num_slots × 6) + (k_paths × num_slots × 6)
        # Nova estrutura otimizada: basic + route_lengths + path/mod + global_slots + path/slots
        total_dim = (
            1  # bit_rate apenas
            + self.k_paths  # route lengths normalizadas
            + (self.k_paths * self.modulations_to_consider * 2)  # path/modulation features
            + (self.num_spectrum_resources * 4)  # features globais por slot (otimizadas)
            + (self.k_paths * self.num_spectrum_resources * 3)  # features path/slot (otimizadas)
        )
        
        self.observation_space = gym.spaces.Box(
                low=-1.0,  # Permitir -1 para slots indisponíveis
                high=1.0,
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
            self.file_stats.write("id,source,destination,bit_rate,path_k,path_length,modulation,min_osnr,osnr,ase,nli,disrupted_services,active_services\n")
        else:
            self.file_stats = None

        self.bl_osnr = 0
        self.bl_resource = 0
        self.bl_reject = 0
        self.spectrum_efficiency_metric = 0
        self.episode_defrag_cicles = 0
        self.episode_service_realocations = 0
        
        self.link_cache = LinkCacheManager(self.topology)
        self.modulation_slots_cache = {}
        self.path_nodes_cache = {}
        self._initialize_optimization_caches()
        
        if reset:
            self.reset()

    cdef void _initialize_optimization_caches(self):
        """Inicializa caches para otimizações de performance"""
        if hasattr(self, 'bit_rates') and hasattr(self, 'modulations'):
            for modulation in self.modulations:
                for bit_rate in self.bit_rates:
                    key = (modulation.spectral_efficiency, bit_rate)
                    bandwidth_required = bit_rate / modulation.spectral_efficiency
                    number_slots = int(np.ceil(bandwidth_required / self.frequency_slot_bandwidth * 1e9))
                    self.modulation_slots_cache[key] = number_slots

        if hasattr(self, 'k_shortest_paths'):
            for source_dest_key, paths in self.k_shortest_paths.items():
                if len(source_dest_key) == 2: 
                    source, destination = source_dest_key
                    for i, path in enumerate(paths):
                        cache_key = (source, destination, i)
                        self.path_nodes_cache[cache_key] = list(path.node_list)

        # Inicializar novos caches para observações slot-wise
        self.available_slots_cache = {}
        self.available_slots_signature_cache = {}
        self.block_info_cache = {}
        self.osnr_matrix_cache = {}
        self.fragmentation_cache = {}
        self.slot_window_cache = {}

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
        self.episode_defrag_cicles = 0
        self.episode_service_realocations = 0

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

        self.link_cache._initialize_cache()
        
        # Limpar caches para novas observações
        self.available_slots_cache.clear()
        self.available_slots_signature_cache.clear()
        self.block_info_cache.clear()
        self.osnr_matrix_cache.clear()
        self.fragmentation_cache.clear()
        self.slot_window_cache.clear()
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
    
    cpdef public normalize_value(self, value, min_v, max_v):

        if max_v == min_v:
            return 0.0
        return (value - min_v) / (max_v - min_v)
    
    cpdef public _get_candidates(self, available_slots, num_slots_required, total_slots):

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

    cpdef public get_max_modulation_index(self):
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

                        if self.qot_constraint == "DIST":
                            qot_acceptable = path.length <= modulation.maximum_length
                        else:
                            osnr, _, _ = calculate_osnr(self, self.current_service, self.qot_constraint)
                            qot_acceptable = osnr >= modulation.minimum_osnr + self.margin
                        
                        self.current_service.path = None
                        self.current_service.initial_slot = -1
                        self.current_service.number_slots = 0
                        self.current_service.center_frequency = 0.0
                        self.current_service.bandwidth = 0.0
                        self.current_service.launch_power = 0.0

                        if qot_acceptable:
                            self.max_modulation_idx = max(len(self.modulations) - idm - 1,
                                                        self.modulations_to_consider - 1)
                            return
        self.max_modulation_idx = self.modulations_to_consider - 1

    def observation_optimized(self):
        """Nova função observation com estrutura híbrida otimizada"""
        if not self.gen_observation:
            obs = np.zeros((self.observation_space.shape[0],), dtype=np.float32)
            action_mask = np.zeros((self.action_space.n,), dtype=np.uint8)
            return obs, {'mask': action_mask}
        
        # ========================
        # ETAPA 1: Setup & Caching
        # ========================
        topology = self.topology
        current_service = self.current_service
        num_spectrum_resources = self.num_spectrum_resources
        k_shortest_paths = self.k_shortest_paths
        num_nodes = topology.number_of_nodes()
        max_bit_rate = max(self.bit_rates)
        self.get_max_modulation_index()

        # Preparar informações dos caminhos
        paths_info = []
        source = current_service.source
        destination = current_service.destination
        
        for path_index, route in enumerate(k_shortest_paths[source, destination]):
            if path_index >= self.k_paths:
                break
            available_slots = self._get_cached_available_slots(route, f"path_{path_index}")
            paths_info.append((route, available_slots))

        # Cache de utilizações por path
        path_utilizations = self._compute_path_utilizations()
        
        # ========================
        # ETAPA 2: Action Mask (Reutilizar atual)
        # ========================
        total_actions = self.k_paths * self.modulations_to_consider * num_spectrum_resources
        action_mask = np.zeros(total_actions + 1, dtype=np.uint8)
        
        valid_actions = 0
        path_modulations_cache = {}
        path_window_masks = {}
        
        for action_index in range(total_actions):
            p_idx = action_index // (self.modulations_to_consider * num_spectrum_resources)
            mod_and_slot = action_index % (self.modulations_to_consider * num_spectrum_resources)
            m_idx = mod_and_slot // num_spectrum_resources
            init_slot = mod_and_slot % num_spectrum_resources
            
            if p_idx >= len(paths_info):
                continue
                
            route, available_slots = paths_info[p_idx]
            cache_prefix = f"path_{p_idx}"
            
            if available_slots[init_slot] == 0:
                continue
                
            if p_idx not in path_modulations_cache:
                start_index = max(0, self.max_modulation_idx - (self.modulations_to_consider - 1))
                path_modulations_cache[p_idx] = list(reversed(self.modulations[start_index: self.max_modulation_idx + 1][:self.modulations_to_consider]))
            modulation_list = path_modulations_cache[p_idx]
            if m_idx >= len(modulation_list):
                continue
            modulation = modulation_list[m_idx]
            num_slots_required = self.get_number_slots(self.current_service, modulation)
            
            if init_slot + num_slots_required > num_spectrum_resources:
                continue
            
            base_key = (p_idx, num_slots_required)
            if base_key not in path_window_masks:
                path_window_masks[base_key] = self._get_cached_window_mask(available_slots, cache_prefix, num_slots_required)
            base_mask = path_window_masks[base_key]
            if init_slot >= base_mask.shape[0] or base_mask[init_slot] == 0:
                continue
            
            guard_needed = (init_slot + num_slots_required) < num_spectrum_resources
            if guard_needed:
                guard_window = num_slots_required + 1
                guard_key = (p_idx, guard_window)
                if guard_key not in path_window_masks:
                    path_window_masks[guard_key] = self._get_cached_window_mask(available_slots, cache_prefix, guard_window)
                guard_mask = path_window_masks[guard_key]
                if init_slot >= guard_mask.shape[0] or guard_mask[init_slot] == 0:
                    continue
            
            # QoT check
            self.current_service.path = route
            self.current_service.initial_slot = init_slot
            self.current_service.number_slots = num_slots_required
            self.current_service.center_frequency = (
                self.frequency_start +
                self.frequency_slot_bandwidth * (init_slot + num_slots_required / 2)
            )
            self.current_service.bandwidth = self.frequency_slot_bandwidth * num_slots_required
            self.current_service.launch_power = self.launch_power

            if self.qot_constraint == "DIST":
                qot_acceptable = route.length <= modulation.maximum_length
            else:
                osnr, _, _ = calculate_osnr(self, self.current_service, self.qot_constraint)
                qot_acceptable = osnr >= modulation.minimum_osnr + self.margin
            
            # Reset service
            self.current_service.path = None
            self.current_service.initial_slot = -1
            self.current_service.number_slots = 0
            self.current_service.center_frequency = 0.0
            self.current_service.bandwidth = 0.0
            self.current_service.launch_power = 0.0
            
            if qot_acceptable:
                action_mask[action_index] = 1
                valid_actions += 1
        
        action_mask[-1] = 1  # Reject action sempre válida

        # ========================
        # ETAPA 3: Basic Features & Route Lengths
        # ========================
        bit_rate_norm = current_service.bit_rate / max_bit_rate
        basic_features = np.array([bit_rate_norm], dtype=np.float32)

        # Route lengths normalizadas
        route_lengths_raw = [route.length for route, _ in paths_info[:self.k_paths]]
        if len(route_lengths_raw) > 1:
            min_length = min(route_lengths_raw)
            max_length = max(route_lengths_raw)
            if max_length > min_length:
                route_lengths_norm = np.array([(length - min_length) / (max_length - min_length) 
                                             for length in route_lengths_raw], dtype=np.float32)
            else:
                route_lengths_norm = np.zeros(len(route_lengths_raw), dtype=np.float32)
        else:
            route_lengths_norm = np.array([0.0], dtype=np.float32)

        # ========================
        # ETAPA 4: Path/Modulation Features (NOVO)
        # ========================
        path_mod_features = self._compute_path_modulation_features(paths_info, path_utilizations)
        path_mod_features_flat = path_mod_features.flatten()

        # ========================
        # ETAPA 5: Global Slot Features (Otimizada)
        # ========================
        available_slots_global = paths_info[0][1]
        global_slot_features = self._compute_global_slot_features_optimized(available_slots_global)
        global_features_flat = global_slot_features.flatten()

        # ========================
        # ETAPA 6: Path/Slot Features (Nova)
        # ========================
        path_slot_features = self._compute_path_slot_features_optimized(paths_info, path_utilizations)
        path_slot_features_flat = path_slot_features.flatten()

        # ========================
        # ETAPA 7: Construir observação final
        # ========================
        observation = np.concatenate([
            basic_features,              # 1 elemento
            route_lengths_norm,          # k_paths elementos
            path_mod_features_flat,      # k_paths * modulations_to_consider * 2 elementos
            global_features_flat,        # num_spectrum_resources * 4 elementos
            path_slot_features_flat      # k_paths * num_spectrum_resources * 3 elementos
        ], axis=0).astype(np.float32)
        
        # Validação das dimensões
        expected_size = (1 + self.k_paths + 
                        (self.k_paths * self.modulations_to_consider * 2) + 
                        (self.num_spectrum_resources * 4) + 
                        (self.k_paths * self.num_spectrum_resources * 3))
        
        if observation.shape[0] != expected_size:
            print(f"[ERROR] Observation size mismatch! Got {observation.shape[0]}, expected {expected_size}")
            raise ValueError(f"Observation dimension mismatch: {observation.shape[0]} != {expected_size}")
        
        # Validação dos valores
        out_of_range_count = ((observation < -1) | (observation > 1)).sum()
        if out_of_range_count > 0:
            print(f"[WARNING] Some observation values outside [-1,1] range!")
            observation = np.clip(observation, -1.0, 1.0)
        
        # ========================
        # LOGS ESTRUTURADOS - Formato Detalhado para Validação
        # ========================
        step_counter = self.episode_services_processed if hasattr(self, 'episode_services_processed') else 0
        source_id = int(current_service.source_id)
        destination_id = int(current_service.destination_id)
        num_nodes = topology.number_of_nodes()
        
        # Cabeçalho principal
        source_norm = source_id / (num_nodes - 1) if num_nodes > 1 else 0.0
        destination_norm = destination_id / (num_nodes - 1) if num_nodes > 1 else 0.0
        #print(f"\nSTEP {step_counter} | svc_id=— | bitrate={current_service.bit_rate} Gbps (norm={bit_rate_norm:.2f}) | src={source_id}/{num_nodes-1} ({source_norm:.2f}) → dst={destination_id}/{num_nodes-1} ({destination_norm:.2f})")
        
        # Rotas com normalização
        route_lengths_raw = [route.length for route, _ in paths_info[:self.k_paths]]
        min_length = min(route_lengths_raw)
        max_length = max(route_lengths_raw)
        routes_info = []
        for i, (route, _) in enumerate(paths_info[:self.k_paths]):
            route_length = route.length
            if max_length > min_length:
                normalized_length = (route_length - min_length) / (max_length - min_length)
            else:
                normalized_length = 0.0
            routes_info.append(f"P{i}={route_length} km → {normalized_length:.2f}")
        #print(f"Routes (norm w.r.t. [{min_length}, {max_length}] km):  {' | '.join(routes_info)}")
        
        # Path/Modulation Features - mostrar para primeiro slot como exemplo
        #print(f"\nPATH/MOD FEATURES:")
        for path_idx in range(min(len(paths_info), self.k_paths)):
            route, available_slots = paths_info[path_idx]
            path_length = route.length
            utilization = path_utilizations[path_idx]
            
            #print(f"  P{path_idx} (length={path_length:.0f}km, util={utilization:.2f}):")
            
            for mod_idx in range(self.modulations_to_consider):
                if mod_idx >= len(self.modulations):
                    continue
                    
                modulation = self.modulations[mod_idx]
                
                # Recalcular features para debug
                distance_penalty = min(path_length / 2000.0, 1.0)
                max_spectral_eff = max([mod.spectral_efficiency for mod in self.modulations])
                efficiency_bonus = modulation.spectral_efficiency / max_spectral_eff
                utilization_penalty = utilization
                
                qot_quality = (efficiency_bonus * 0.5) + ((1.0 - distance_penalty) * 0.3) + ((1.0 - utilization_penalty) * 0.2)
                
                slots_required = self.get_number_slots(current_service, modulation)
                available_positions = self._count_available_positions(available_slots, slots_required, f"path_{path_idx}")
                max_possible_positions = max(1, self.num_spectrum_resources - slots_required + 1)
                competitiveness = available_positions / max_possible_positions
                
                #print(f"    M{mod_idx} ({modulation.name}): qot_quality={qot_quality:.3f} (eff={efficiency_bonus:.2f}, dist_pen={distance_penalty:.2f}, util_pen={utilization_penalty:.2f}) | competitiveness={competitiveness:.3f} ({available_positions}/{max_possible_positions})")
        
        # PATHS detalhadas (slot 0 como exemplo)
        #print(f"\nPATHS (slot=0):")
        for path_idx, (route, available_slots) in enumerate(paths_info[:self.k_paths]):
            if available_slots[0] == 1:  # Se slot 0 está disponível
                # Calcular OSNR para slot 0
                osnr_values = self._compute_slot_osnr_vectorized(route, available_slots)
                osnr_db = osnr_values[0]
                
                # Encontrar melhor modulação
                best_modulations = self._find_best_modulations_per_slot(available_slots, osnr_values)
                best_mod_idx = best_modulations[0]
                
                if best_mod_idx >= 0:
                    best_modulation = self.modulations[best_mod_idx]
                    osnr_req = best_modulation.minimum_osnr + self.margin
                    margin = osnr_db - osnr_req
                    mod_name = best_modulation.name
                else:
                    osnr_req = 0.0
                    margin = 0.0
                    mod_name = "None"
                    best_mod_idx = -1
                
                # Path quality
                path_quality = 1.0 - min(route.length / 2000.0, 1.0)
                
                # Load balancing para slot 0
                other_paths_util = np.concatenate([path_utilizations[:path_idx], path_utilizations[path_idx+1:]])
                mean_other_utilization = np.mean(other_paths_util) if len(other_paths_util) > 0 else 0.0
                current_path_utilization = path_utilizations[path_idx]
                load_imbalance = mean_other_utilization - current_path_utilization
                load_balancing_value = min(max(0.5 + load_imbalance, 0.0), 1.0)
                
                # Contiguous capacity para slot 0
                contiguous_from_slot = self._count_contiguous_from_slot(0, available_slots)
                contiguous_capacity = min(contiguous_from_slot / 50.0, 1.0)
                
                #print(f"  P{path_idx}: OSNR={osnr_db:.3f} dB | req={osnr_req:.2f} dB | margin={margin:+.3f} dB | best_mod={mod_name} (idx={best_mod_idx}) | path_quality={path_quality:.2f}")
                #print(f"      load_balance={load_balancing_value:.3f} (others_avg={mean_other_utilization:.2f}, current={current_path_utilization:.2f}) | contiguous={contiguous_capacity:.3f} ({contiguous_from_slot} slots)")
            #else:
                #print(f"  P{path_idx}: slot 0 unavailable")
        
        # SLOTS globais
        #print(f"\nSLOTS (0..{self.num_spectrum_resources-1}) [is_avail, block_norm, frag, edge_norm]")
        available_slots_global = paths_info[0][1]
        fragmentation_global = self._get_cached_fragmentation(available_slots_global, "global_optimized")
        
        # Mostrar apenas primeiros 5 slots para não poluir
        for slot in range(min(5, self.num_spectrum_resources)):
            if available_slots_global[slot] == 1:
                # Recalcular features para debug
                local_block_size = self._get_local_block_size(slot, available_slots_global)
                block_norm = min(local_block_size / 50.0, 1.0)
                frag = fragmentation_global[slot]
                edge_distance = min(slot, self.num_spectrum_resources - 1 - slot) / (self.num_spectrum_resources / 2.0)
                
                #print(f"  s{slot}: [1, {block_norm:.2f}, {frag:.2f}, {edge_distance:.2f}] | block_size={local_block_size}, edge_dist={min(slot, self.num_spectrum_resources - 1 - slot)}")
            #else:
                #print(f"  s{slot}: [0, -1.00, -1.00, -1.00] (unavailable)")
        
        #if self.num_spectrum_resources > 5:
            #print(f"  ... (showing first 5 of {self.num_spectrum_resources} slots)")
        
        # Estatísticas da observation
        #print(f"\nOBS (sanity): shape={observation.shape[0]} | min={observation.min():.2f} | max={observation.max():.2f} | out_of_range={out_of_range_count}")
        #print(f"Structure breakdown: basic={len(basic_features)} + routes={len(route_lengths_norm)} + path/mod={path_mod_features_flat.shape[0]} + global={global_features_flat.shape[0]} + path/slot={path_slot_features_flat.shape[0]}")
        
        # Simulação de decisão
        if valid_actions > 0:
            first_valid_action = np.where(action_mask[:-1] == 1)[0][0]
            p_idx = first_valid_action // (self.modulations_to_consider * self.num_spectrum_resources)
            mod_and_slot = first_valid_action % (self.modulations_to_consider * self.num_spectrum_resources)
            m_idx = mod_and_slot // self.num_spectrum_resources
            s_idx = mod_and_slot % self.num_spectrum_resources
            
            if p_idx < len(path_modulations_cache) and m_idx < len(path_modulations_cache[p_idx]):
                selected_mod = path_modulations_cache[p_idx][m_idx]
                #print(f"DECISÃO: path={p_idx} | slot={s_idx} | mod={selected_mod.name} | ok=True")
            #else:
                #print(f"DECISÃO: path={p_idx} | slot={s_idx} | mod=unknown | ok=True")
        #else:
            #print(f"DECISÃO: reject | reason=no_valid_actions")
        
        return observation, {'mask': action_mask}

    def observation_slot_wise(self):
        """Nova função observation com estrutura otimizada - 185 elementos sem redundância"""
        if not self.gen_observation:
            obs = np.zeros((self.observation_space.shape[0],), dtype=np.float32)
            action_mask = np.zeros((self.action_space.n,), dtype=np.uint8)
            return obs, {'mask': action_mask}
        
        # Preparação inicial
        topology = self.topology
        current_service = self.current_service
        num_spectrum_resources = self.num_spectrum_resources
        k_shortest_paths = self.k_shortest_paths
        num_nodes = topology.number_of_nodes()
        max_bit_rate = max(self.bit_rates)
        self.get_max_modulation_index()

        # ========================
        # ETAPA 1: Gerar ACTION MASK primeiro
        # ========================
        
        # Preparar informações dos caminhos
        paths_info = []
        source = current_service.source
        destination = current_service.destination
        
        for path_index, route in enumerate(k_shortest_paths[source, destination]):
            if path_index >= self.k_paths:
                break
            available_slots = self._get_cached_available_slots(route, f"path_{path_index}")
            paths_info.append((route, available_slots))

        # Gerar action mask usando informações dos slots calculados
        total_actions = self.k_paths * self.modulations_to_consider * num_spectrum_resources
        action_mask = np.zeros(total_actions + 1, dtype=np.uint8)
        
        valid_actions = 0
        # Cache para modulações por caminho
        path_modulations_cache = {}
        path_window_masks = {}
        
        for action_index in range(total_actions):
            p_idx = action_index // (self.modulations_to_consider * num_spectrum_resources)
            mod_and_slot = action_index % (self.modulations_to_consider * num_spectrum_resources)
            m_idx = mod_and_slot // num_spectrum_resources
            init_slot = mod_and_slot % num_spectrum_resources
            
            if p_idx >= len(paths_info):
                continue
                
            route, available_slots = paths_info[p_idx]
            cache_prefix = f"path_{p_idx}"
            
            # Verificar se slot está disponível
            if available_slots[init_slot] == 0:
                continue
                
            # Cache das modulações para este caminho
            if p_idx not in path_modulations_cache:
                start_index = max(0, self.max_modulation_idx - (self.modulations_to_consider - 1))
                path_modulations_cache[p_idx] = list(reversed(self.modulations[start_index: self.max_modulation_idx + 1][:self.modulations_to_consider]))
            modulation_list = path_modulations_cache[p_idx]
            if m_idx >= len(modulation_list):
                continue
            modulation = modulation_list[m_idx]
            num_slots_required = self.get_number_slots(self.current_service, modulation)
            
            # Verificar se há slots suficientes a partir desta posição
            if init_slot + num_slots_required > num_spectrum_resources:
                continue
            
            base_key = (p_idx, num_slots_required)
            if base_key not in path_window_masks:
                path_window_masks[base_key] = self._get_cached_window_mask(available_slots, cache_prefix, num_slots_required)
            base_mask = path_window_masks[base_key]
            if init_slot >= base_mask.shape[0] or base_mask[init_slot] == 0:
                continue
                
            # Verificar se path está livre usando verificação rigorosa
            if not self.is_path_free(route, init_slot, num_slots_required):
                continue
            
            # Verificar QoT constraint
            self.current_service.path = route
            self.current_service.initial_slot = init_slot
            self.current_service.number_slots = num_slots_required
            self.current_service.center_frequency = (
                self.frequency_start +
                self.frequency_slot_bandwidth * (init_slot + num_slots_required / 2)
            )
            self.current_service.bandwidth = self.frequency_slot_bandwidth * num_slots_required
            self.current_service.launch_power = self.launch_power
            self.current_service.blocked_due_to_resources = False

            if self.qot_constraint == "DIST":
                qot_acceptable = route.length <= modulation.maximum_length
            else:
                osnr, _, _ = calculate_osnr(self, self.current_service, self.qot_constraint)
                qot_acceptable = osnr >= modulation.minimum_osnr + self.margin
            
            # Limpar configuração temporária
            self.current_service.path = None
            self.current_service.initial_slot = -1
            self.current_service.number_slots = 0
            self.current_service.center_frequency = 0.0
            self.current_service.bandwidth = 0.0
            self.current_service.launch_power = 0.0
            
            if qot_acceptable:
                    action_mask[action_index] = 1
                    valid_actions += 1
        
        action_mask[-1] = 1  # Reject action sempre válida

        # ========================
        # ETAPA 2: Computar features básicas
        # ========================
        
        # Features básicas (3 valores)
        source_id = int(current_service.source_id)
        destination_id = int(current_service.destination_id)
        
        bit_rate_norm = current_service.bit_rate / max_bit_rate
        source_norm = source_id / (num_nodes - 1) if num_nodes > 1 else 0.0
        destination_norm = destination_id / (num_nodes - 1) if num_nodes > 1 else 0.0
        
        # ========================
        # LOGS ESTRUTURADOS - Formato Solicitado
        # ========================
        
        # Cabeçalho principal com step counter
        step_counter = self.episode_services_processed if hasattr(self, 'episode_services_processed') else 0
        #print(f"\nSTEP {step_counter} | bitrate={current_service.bit_rate} Gbps (norm={bit_rate_norm:.2f}) | src={source_id}/{num_nodes-1} ({source_norm:.2f}) → dst={destination_id}/{num_nodes-1} ({destination_norm:.2f})")
        
        # Informações das modulações para referência
        #print(f"Modulations: [{', '.join([f'{mod.name}@{mod.minimum_osnr:.1f}dB' for mod in self.modulations])}] | margin={self.margin:.1f}dB")
        
        basic_features = np.array([bit_rate_norm, source_norm, destination_norm], dtype=np.float32)

        # Route lengths (k_paths valores) - CORREÇÃO: usar path lengths reais
        route_lengths_raw = [route.length for route, _ in paths_info[:self.k_paths]]
        min_length = min(route_lengths_raw)
        max_length = max(route_lengths_raw)
        
        # Log das rotas em uma linha
        routes_info = []
        route_lengths = np.zeros(self.k_paths, dtype=np.float32)
        for i, (route, _) in enumerate(paths_info):
            if i < self.k_paths:
                route_length = route.length
                normalized_length = self.normalize_value(route_length, min_length, max_length)
                route_lengths[i] = normalized_length
                routes_info.append(f"P{i}={route_length:.0f} km → {normalized_length:.2f}")
        
        #print(f"Routes (norm w.r.t. [{min_length:.0f}, {max_length:.0f}] km):  {' | '.join(routes_info)}")

        # ========================
        # ETAPA 3: Features globais dos slots (UMA VEZ SÓ)
        # ========================
        
        # Usar primeiro path para features globais (são iguais para todos)
        available_slots_global = paths_info[0][1]
        fragmentation_global = self._get_cached_fragmentation(available_slots_global, "global")
        global_slot_features = self._compute_global_slot_features(available_slots_global, fragmentation_global)

        # ========================
        # ETAPA 4: Features path-específicas para cada caminho
        # ========================
        
        path_specific_features = []
        
        # Armazenar dados para logs estruturados
        paths_log_data = []
        
        for path_idx, (route, available_slots) in enumerate(paths_info):
            if path_idx >= self.k_paths:
                break
            
            # Calcular features path-específicas  
            osnr_values = self._compute_slot_osnr_vectorized(route, available_slots)
            best_modulations = self._find_best_modulations_per_slot(available_slots, osnr_values)
            
            path_features = self._compute_path_specific_features(route, available_slots, osnr_values, best_modulations)
            path_specific_features.append(path_features.flatten())
            
            # Guardar dados para logs (apenas slot 0 para exemplo)
            if available_slots[0] == 1:  # Se slot 0 está disponível
                osnr_db = osnr_values[0]
                best_mod_idx = best_modulations[0]
                
                if best_mod_idx >= 0:
                    best_modulation = self.modulations[best_mod_idx]
                    osnr_req = best_modulation.minimum_osnr
                    margin = osnr_db - osnr_req
                    mod_name = best_modulation.name
                    
                    # Path quality baseado na distância
                    path_quality = 1.0 - min(route.length / 2000.0, 1.0)
                else:
                    osnr_req = 0.0
                    margin = 0.0
                    mod_name = "None"
                    path_quality = 0.0
                
                paths_log_data.append({
                    'path_idx': path_idx,
                    'osnr_db': osnr_db,
                    'osnr_req': osnr_req,
                    'margin': margin,
                    'mod_name': mod_name,
                    'mod_idx': best_mod_idx,
                    'path_quality': path_quality
                })
            else:
                paths_log_data.append({
                    'path_idx': path_idx,
                    'osnr_db': -1,
                    'osnr_req': -1,
                    'margin': -1,
                    'mod_name': "N/A",
                    'mod_idx': -1,
                    'path_quality': 0.0
                })
        
        # LOG ESTRUTURADO DOS PATHS (slot=0)
        #print(f"\nPATHS (slot=0) + path_features_info")
        for path_data in paths_log_data:
            if path_data['osnr_db'] >= 0:
                # Calcular informações extras para features path-específicas
                path_idx = path_data['path_idx']
                route = paths_info[path_idx][0]
                available_slots = paths_info[path_idx][1]
                osnr_values = self._compute_slot_osnr_vectorized(route, available_slots)
                best_modulations = self._find_best_modulations_per_slot(available_slots, osnr_values)
                
                # Dados para normalização OSNR (por path)
                valid_osnr = osnr_values[osnr_values >= 0]
                osnr_min = np.min(valid_osnr) if len(valid_osnr) > 0 else 0.0
                osnr_max = np.max(valid_osnr) if len(valid_osnr) > 0 else 1.0
                osnr_range = osnr_max - osnr_min if osnr_max > osnr_min else 1.0
                
                # Eficiência espectral máxima disponível
                max_se = max([mod.spectral_efficiency for mod in self.modulations])
                best_mod_se = self.modulations[path_data['mod_idx']].spectral_efficiency if path_data['mod_idx'] >= 0 else 0.0
                
                #print(f"  P{path_data['path_idx']}: OSNR={path_data['osnr_db']:.3f} dB | req={path_data['osnr_req']:.2f} dB | margin={path_data['margin']:+.3f} dB | best_mod={path_data['mod_name']} (idx={path_data['mod_idx']}) | path_quality={path_data['path_quality']:.2f}")
                #print(f"    ↳ feat_calc: osnr_range=[{osnr_min:.1f}, {osnr_max:.1f}], se={best_mod_se:.1f}/{max_se:.1f}, length={route.length:.0f}km, margin_norm={path_data['margin']/10.0:.3f}")
            #else:
                #print(f"  P{path_data['path_idx']}: slot unavailable")
        
        # LOG ESTRUTURADO DOS SLOTS GLOBAIS
        #print(f"\nSLOTS (0..{num_spectrum_resources-1})  [is_avail, block_norm, frag, edge_norm, ctx] + calc_info")
        for slot in range(num_spectrum_resources):
            features = global_slot_features[slot]
            if available_slots_global[slot] == 1:
                # Calcular informações extras para validação
                local_block_size = self._get_local_block_size(slot, available_slots_global)
                edge_distance_raw = min(slot, num_spectrum_resources - 1 - slot)
                context_start = max(0, slot - 8)
                context_end = min(num_spectrum_resources, slot + 8 + 1)
                context_util_raw = 1.0 - np.mean(available_slots_global[context_start:context_end])
                
                #print(f"  s{slot}: [{features[0]:.0f}, {features[1]:.2f}, {features[2]:.2f}, {features[3]:.2f}, {features[4]:.2f}] | block_sz={local_block_size}, edge_dist={edge_distance_raw}, ctx_window=[{context_start}:{context_end}], ctx_util={context_util_raw:.3f}")
            #else:
                #print(f"  s{slot}: [0, -, -, -, -] (unavailable)")

        # ========================
        # ETAPA 5: Construir observação final otimizada
        # ========================
        
        # Flatten global features: (num_slots, 5) -> (num_slots * 5,)
        global_features_flat = global_slot_features.flatten()
        
        # Concatenar todas as features na nova estrutura
        observation = np.concatenate([
            basic_features,        # 3 valores
            route_lengths,         # k_paths valores  
            global_features_flat,  # num_slots * 5 valores (features globais)
            *path_specific_features # k_paths * (num_slots * 6) valores (features path-específicas)
        ], axis=0).astype(np.float32)
        
        # Validação das dimensões
        expected_size = 3 + self.k_paths + (self.num_spectrum_resources * 5) + (self.k_paths * self.num_spectrum_resources * 6)
        if observation.shape[0] != expected_size:
            #print(f"[ERROR] Observation size mismatch! Got {observation.shape[0]}, expected {expected_size}")
            raise ValueError(f"Observation dimension mismatch: {observation.shape[0]} != {expected_size}")
        
        # Validação dos valores (devem estar entre -1 e 1)
        out_of_range_count = ((observation < -1) | (observation > 1)).sum()
        if out_of_range_count > 0:
            #print(f"[WARNING] Some observation values outside [-1,1] range!")
            #print(f"          Min: {np.min(observation):.6f}, Max: {np.max(observation):.6f}")
            # Clip para garantir que está no range
            observation = np.clip(observation, -1.0, 1.0)
        
        # RESUMO FINAL DA OBSERVAÇÃO
        #print(f"\nOBS (sanity): shape={observation.shape[0]} | min={observation.min():.2f} | max={observation.max():.2f} | out_of_range={out_of_range_count}")
        
        # Simulação de decisão (para exemplo nos logs)
        valid_actions_count = np.sum(action_mask[:-1])  # Excluir reject action
        if valid_actions_count > 0:
            # Encontrar primeira ação válida para exemplo
            first_valid_action = np.where(action_mask[:-1] == 1)[0][0]
            p_idx = first_valid_action // (self.modulations_to_consider * num_spectrum_resources)
            mod_and_slot = first_valid_action % (self.modulations_to_consider * num_spectrum_resources)
            m_idx = mod_and_slot // num_spectrum_resources
            slot_idx = mod_and_slot % num_spectrum_resources
            
            # Obter nome da modulação
            start_index = max(0, self.max_modulation_idx - (self.modulations_to_consider - 1))
            path_modulations = list(reversed(self.modulations[start_index: self.max_modulation_idx + 1][:self.modulations_to_consider]))
            mod_name = path_modulations[m_idx].name if m_idx < len(path_modulations) else "Unknown"
            
            #print(f"DECISÃO (exemplo): path={p_idx} | slot={slot_idx} | mod={mod_name} | valid_actions={valid_actions_count} | ok=True")
        #else:
            #print(f"DECISÃO (exemplo): REJECT | valid_actions=0 | ok=False")
        
        #print(f"{'='*80}")
        
        return observation, {'mask': action_mask}

    def observation(self):
        # NOVA IMPLEMENTAÇÃO OTIMIZADA
        return self.observation_optimized()



    cpdef decimal_to_array(self, decimal: int, max_values=None):
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
        return array

    cpdef encoded_decimal_to_array(self, decimal: int, max_values=None):
        part_size = self.num_spectrum_resources // self.modulations_to_consider  
        mod_idx = decimal // part_size  
        
        if max_values is None:
            max_values = [self.k_paths, self.modulations_to_consider, self.num_spectrum_resources]        
        array = []
        for max_val in reversed(max_values):
            digit = decimal % max_val
            array.insert(0, digit)
            decimal //= max_val

        if self.max_modulation_idx > 1:

            allowed_mods = list(range(self.max_modulation_idx, self.max_modulation_idx - self.modulations_to_consider, -1))
        else:
            allowed_mods = list(reversed(list(range(0, self.modulations_to_consider))))

        array[1] = allowed_mods[array[1]]        
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
        cdef dict link_data

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

                if self.qot_constraint == "DIST":
                    path_distance = self.current_service.path.length
                    qot_acceptable = path_distance <= modulation.maximum_length
                    osnr = 0.0 if qot_acceptable else -1.0
                    ase = 0.0
                    nli = 0.0
                else:
                    path_distance = self.current_service.path.length  # For error message
                    osnr, ase, nli = calculate_osnr(self, self.current_service, self.qot_constraint)
                    qot_acceptable = osnr >= osnr_req

                if qot_acceptable:
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
                    # QoT não aceitável - serviço bloqueado por OSNR/distância
                    self.current_service.accepted = False
                    self.current_service.blocked_due_to_osnr = True
                    self.bl_osnr += 1
                    
                    # Se não está em modo RL, lança exceção (comportamento original)
                    if not self.rl_mode:
                        if self.qot_constraint == "DIST":
                            raise ValueError(
                                f"Path distance {path_distance:.2f} km is greater than maximum length "
                                f"{modulation.maximum_length} km for service {self.current_service.service_id} "
                                f"with modulation {modulation}."
                            )
                        else:
                            raise ValueError(
                                f"Osnr {osnr} is not enough for service {self.current_service.service_id} "
                                f"with modulation {modulation}, and osnr_req {osnr_req}."
                            )
            else:
                # Caminho não está livre - serviço bloqueado por recursos
                self.current_service.accepted = False
                self.current_service.blocked_due_to_resources = True
                self.bl_resource += 1
                
                # Se não está em modo RL, lança exceção (comportamento original)
                if not self.rl_mode:
                    raise ValueError(
                        f"Path {path} is not free for service {self.current_service.service_id} "
                        f"with initial slot {initial_slot} and number of slots {number_slots}."
                    )

        if self.measure_disruptions and self.current_service.accepted:
            services_to_measure = []
            
            for link in self.current_service.path.links:
                link_data = self.link_cache.get_link_data(link.node1, link.node2)
                running_services = link_data['running_services']
                
                for service_in_link in running_services:
                    if (service_in_link not in services_to_measure
                            and service_in_link not in self.disrupted_services_list):
                        services_to_measure.append(service_in_link)

            for svc in services_to_measure:
                if self.qot_constraint == "DIST":
                    path_distance = svc.path.length
                    qot_acceptable = path_distance <= svc.current_modulation.maximum_length
                    osnr_svc = 0.0 if qot_acceptable else -1.0
                else:
                    osnr_svc, ase_svc, nli_svc = calculate_osnr(self, svc, self.qot_constraint)
                    qot_acceptable = osnr_svc >= svc.current_modulation.minimum_osnr
                
                if not qot_acceptable:
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
                line += "{},{},{},{},{},{},{},{},{}".format(
                    self.current_service.path.k,
                    self.current_service.path.length,
                    self.current_service.current_modulation.spectral_efficiency,
                    self.current_service.current_modulation.minimum_osnr,
                    self.current_service.OSNR,
                    self.current_service.ASE,
                    self.current_service.NLI,
                    disrupted_services,
                    len(self.topology.graph["running_services"]),
                )
            else:
                line += "-1,-1,-1,-1,-1,-1,-1,-1,-1"
            line += "\n"
            self.file_stats.write(line)
            self.file_stats.flush()

        # Inicializar info dict (será preenchido antes de calcular reward)
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
            "episode_defrag_cicles": self.episode_defrag_cicles,
            "episode_service_realocations": self.episode_service_realocations,
        }
        # Verificar se isso vai se manter após o paper
        # Calculate current fragmentation metrics (ANTES da reward!)
        if len(self.topology.graph["running_services"]) > 0:
            # Get available slots for all links
            available_slots_matrix = []
            for n1, n2 in self.topology.edges():
                link_index = self.topology[n1][n2]["index"]
                available_slots = self.topology.graph["available_slots"][link_index, :].tolist()
                available_slots_matrix.append(available_slots)
            
            # Calculate fragmentation metrics
            if available_slots_matrix:
                # Shannon entropy (average across all links)
                entropies = [link_shannon_entropy_(row) for row in available_slots_matrix]
                shannon_entropy_avg = sum(entropies) / len(entropies) if entropies else 0.0
                
                # Route cuts (total across all links)
                route_cuts_total = fragmentation_route_cuts(available_slots_matrix)
                
                # Route RSS 
                route_rss = fragmentation_route_rss(available_slots_matrix)
                
                info["fragmentation_shannon_entropy"] = shannon_entropy_avg
                info["fragmentation_route_cuts"] = route_cuts_total
                info["fragmentation_route_rss"] = route_rss
            else:
                info["fragmentation_shannon_entropy"] = 0.0
                info["fragmentation_route_cuts"] = 0
                info["fragmentation_route_rss"] = 0.0
        else:
            info["fragmentation_shannon_entropy"] = 0.0
            info["fragmentation_route_cuts"] = 0
            info["fragmentation_route_rss"] = 0.0

        # Calcular reward APÓS ter métricas de fragmentação no info dict
        reward = self.reward(info)

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
        self.topology.graph["services"].append(self.current_service)
        self._next_service()

        terminated = (self.episode_services_processed == self.episode_length)
        if terminated:
            info["blocked_due_to_resources"] = self.bl_resource
            info["blocked_due_to_osnr"] = self.bl_osnr
            info["rejected"] = self.bl_reject

        # Clear caches before generating new observation since state has changed
        self.available_slots_cache.clear()
        self.available_slots_signature_cache.clear()
        self.block_info_cache.clear()
        self.osnr_matrix_cache.clear()
        self.fragmentation_cache.clear()
        self.slot_window_cache.clear()

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
                if self.defragmentation:
                    if self.n_defrag_services == 0 or self.episode_services_processed % self.n_defrag_services == 0:
                        self.defragment(self.n_defrag_services)
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int get_number_slots(self, object service, object modulation):
        cdef double required_slots
        cdef tuple cache_key
        
        cache_key = (modulation.spectral_efficiency, service.bit_rate)
        if cache_key in self.modulation_slots_cache:
            return self.modulation_slots_cache[cache_key]
        
        required_slots = service.bit_rate / (modulation.spectral_efficiency * self.channel_width)
        cdef int result = int(math.ceil(required_slots))
        
        self.modulation_slots_cache[cache_key] = result
        return result

    # Funções auxiliares para observações slot-wise eficientes
    
    cdef cnp.ndarray _get_cached_available_slots(self, object path, str cache_key):
        """Obtém available_slots do cache ou calcula se necessário"""
        if cache_key not in self.available_slots_cache:
            self.available_slots_cache[cache_key] = self.get_available_slots(path)
        return self.available_slots_cache[cache_key]
    
    cdef dict _get_cached_block_info(self, cnp.ndarray available_slots, str cache_key):
        """Obtém informações de blocos do cache ou calcula se necessário"""
        if cache_key not in self.block_info_cache:
            initial_indices, values, lengths = rle(available_slots)
            free_blocks = []
            for start, val, length in zip(initial_indices, values, lengths):
                if val == 1:
                    free_blocks.append((start, length))
            self.block_info_cache[cache_key] = {
                'free_blocks': free_blocks,
                'total_free_slots': np.sum(available_slots),
                'num_free_blocks': len(free_blocks)
            }
        return self.block_info_cache[cache_key]
    
    cdef cnp.ndarray _get_cached_fragmentation(self, cnp.ndarray available_slots, str cache_key):
        """Calcula fragmentação local para cada slot (janela ±16)"""
        cdef int num_slots = self.num_spectrum_resources
        cdef cnp.ndarray fragmentation
        cdef int window_size = 16
        cdef int i, start, end, free_in_window, window_length
        
        if cache_key not in self.fragmentation_cache:
            fragmentation = np.zeros(num_slots, dtype=np.float32)
            
            for i in range(num_slots):
                start = max(0, i - window_size)
                end = min(num_slots, i + window_size + 1)
                window_length = end - start
                free_in_window = np.sum(available_slots[start:end])
                
                # Fragmentação = 1 - (slots_livres / tamanho_janela)
                fragmentation[i] = 1.0 - (free_in_window / window_length)
                
            self.fragmentation_cache[cache_key] = fragmentation
        return self.fragmentation_cache[cache_key]

    cdef cnp.ndarray _get_cached_window_mask(self, cnp.ndarray available_slots, str cache_key, int window_size):
        """Retorna máscara booleana (via convolução) indicando janelas contínuas de slots livres."""
        cdef str window_key
        cdef Py_ssize_t total_slots = available_slots.shape[0]
        if window_size <= 0:
            return np.zeros((0,), dtype=np.uint8)
        if window_size > total_slots:
            return np.zeros((0,), dtype=np.uint8)
        window_key = f"{cache_key}_win_{window_size}"
        if window_key not in self.slot_window_cache:
            slot_view = np.asarray(available_slots, dtype=np.int8)
            kernel = np.ones(window_size, dtype=np.int8)
            window_sums = np.convolve(slot_view, kernel, mode="valid")
            self.slot_window_cache[window_key] = (window_sums == window_size).astype(np.uint8)
        return self.slot_window_cache[window_key]

    cdef cnp.ndarray _compute_slot_osnr_vectorized(self, object path, cnp.ndarray available_slots):
        """Calcula OSNR para todos os slots de uma vez usando vetorização"""
        cdef str cache_key = f"osnr_{path.k}_{hash(tuple(available_slots))}"
        
        if cache_key not in self.osnr_matrix_cache:
            # Usar função do core/osnr.pyx com qot_constraint flexível
            self.osnr_matrix_cache[cache_key] = compute_slot_osnr_vectorized(self, path, available_slots, self.qot_constraint)
            
            # Validar se solicitado (apenas no debug)
            if hasattr(self, 'validate_osnr') and self.validate_osnr:
                validation_passed = validate_osnr_vectorized(self, path, available_slots, 1e-6, self.qot_constraint)
                if not validation_passed:
                    print(f"[WARNING] OSNR vectorized validation failed for path {path.k}")
            
        return self.osnr_matrix_cache[cache_key]

    cdef dict _get_free_blocks_info(self, cnp.ndarray available_slots):
        """Extrai informações detalhadas sobre blocos livres"""
        cdef str cache_key = f"blocks_{hash(tuple(available_slots))}"
        
        if cache_key not in self.block_info_cache:
            initial_indices, values, lengths = rle(available_slots)
            free_blocks = []
            total_free_slots = 0
            
            for start, val, length in zip(initial_indices, values, lengths):
                if val == 1:  # Bloco livre
                    free_blocks.append({
                        'start': start,
                        'length': length,
                        'end': start + length - 1
                    })
                    total_free_slots += length
            
            block_info = {
                'free_blocks': free_blocks,
                'total_free_slots': total_free_slots,
                'num_free_blocks': len(free_blocks),
                'largest_block': max([b['length'] for b in free_blocks]) if free_blocks else 0,
                'avg_block_size': total_free_slots / len(free_blocks) if free_blocks else 0.0
            }
            
            self.block_info_cache[cache_key] = block_info

            
        return self.block_info_cache[cache_key]

    cdef list _find_best_modulations_per_slot(self, cnp.ndarray available_slots, cnp.ndarray osnr_values):
        """Determina a melhor modulação viável para cada slot"""
        cdef int num_slots = self.num_spectrum_resources
        cdef list best_modulations = []
        cdef int slot, best_mod_idx
        cdef double slot_osnr
        cdef object modulation
        
        for slot in range(num_slots):
            best_mod_idx = -1
            
            if available_slots[slot] == 1:  # Slot disponível
                slot_osnr = osnr_values[slot]
                
                # Procurar a melhor modulação (maior eficiência espectral) que satisfaz OSNR
                for mod_idx in range(len(self.modulations) - 1, -1, -1):  # Do maior para menor SE
                    modulation = self.modulations[mod_idx]
                    if slot_osnr >= modulation.minimum_osnr + self.margin:
                        best_mod_idx = mod_idx
                        break
                        
            best_modulations.append(best_mod_idx)
        
        return best_modulations

    cdef cnp.ndarray _compute_slot_features(self, object path, cnp.ndarray available_slots, 
                                           cnp.ndarray osnr_values, list best_modulations,
                                           dict block_info, cnp.ndarray fragmentation):
        """Gera as 10 features por slot usando as funções auxiliares"""
        cdef int num_slots = self.num_spectrum_resources
        cdef cnp.ndarray features = np.zeros((num_slots, 10), dtype=np.float32)
        cdef int slot, best_mod_idx, local_block_size
        cdef double edge_distance, osnr_normalized, osnr_margin, spectral_efficiency
        cdef object best_modulation
        

        
        # Normalização global para OSNR (calculada uma vez)
        cdef double osnr_min = np.min(osnr_values[osnr_values >= 0]) if np.any(osnr_values >= 0) else 0.0
        cdef double osnr_max = np.max(osnr_values)
        cdef double osnr_range = osnr_max - osnr_min if osnr_max > osnr_min else 1.0
        

        
        for slot in range(num_slots):
            # Feature 1: is_available (0.0 ou 1.0)
            features[slot, 0] = float(available_slots[slot])
            
            # Feature 2: local_block_size (tamanho do bloco que contém este slot)
            local_block_size = self._get_local_block_size(slot, available_slots)
            features[slot, 1] = min(local_block_size / 50.0, 1.0)  # Normalizar por 50 slots max
            
            # Feature 3: fragmentation_local (já calculada)
            features[slot, 2] = fragmentation[slot]
            
            # Feature 4: edge_proximity (distância das bordas, normalizada)
            edge_distance = min(slot, num_slots - 1 - slot) / (num_slots / 2.0)
            features[slot, 3] = edge_distance
            
            # Feature 5: osnr_normalized
            if available_slots[slot] == 1 and osnr_values[slot] >= 0:
                features[slot, 4] = (osnr_values[slot] - osnr_min) / osnr_range
            else:
                features[slot, 4] = 0.0
            
            # Feature 6: osnr_margin_best_mod
            best_mod_idx = best_modulations[slot]
            if best_mod_idx >= 0 and available_slots[slot] == 1:
                best_modulation = self.modulations[best_mod_idx]
                osnr_margin = osnr_values[slot] - best_modulation.minimum_osnr
                features[slot, 5] = min(osnr_margin / 10.0, 1.0)  # Normalizar por 10dB margem max
            else:
                features[slot, 5] = 0.0
            
            # Feature 7: path_quality_indicator (baseado no path length normalizado)
            path_quality = 1.0 - min(path.length / 2000.0, 1.0)  # Assumindo 2000km como max
            features[slot, 6] = path_quality
            
            # Feature 8: best_modulation_tier (tier da melhor modulação)
            if best_mod_idx >= 0:
                features[slot, 7] = best_mod_idx / (len(self.modulations) - 1)
            else:
                features[slot, 7] = 0.0
            
            # Feature 9: spectral_efficiency_potential
            if best_mod_idx >= 0:
                best_modulation = self.modulations[best_mod_idx]
                spectral_efficiency = best_modulation.spectral_efficiency
                # Normalizar pela máxima eficiência espectral disponível
                max_se = max([mod.spectral_efficiency for mod in self.modulations])
                features[slot, 8] = spectral_efficiency / max_se
            else:
                features[slot, 8] = 0.0
            
            # Feature 10: utilization_context (utilização na vizinhança ±8 slots)
            context_start = max(0, slot - 8)
            context_end = min(num_slots, slot + 8 + 1)
            context_utilization = 1.0 - np.mean(available_slots[context_start:context_end])
            features[slot, 9] = context_utilization
        



        
        return features

    cdef cnp.ndarray _compute_global_slot_features(self, cnp.ndarray available_slots, cnp.ndarray fragmentation):
        """Calcula features globais dos slots (independem do path) - 5 features por slot"""
        cdef int num_slots = self.num_spectrum_resources
        cdef cnp.ndarray features = np.zeros((num_slots, 5), dtype=np.float32)
        cdef int slot, local_block_size, context_start, context_end
        cdef double edge_distance, context_utilization
        
        # DEBUG LIMPO: Só mostrar features globais resumidas (comentado para logs estruturados)
        # print(f"\n🔍 DEBUG FEATURES GLOBAIS (5 features × {num_slots} slots):")
        
        for slot in range(num_slots):
            if available_slots[slot] == 1:  # Slot disponível
                # Feature 0: is_available  
                features[slot, 0] = 1.0
                
                # Feature 1: local_block_size
                local_block_size = self._get_local_block_size(slot, available_slots)
                features[slot, 1] = min(local_block_size / 50.0, 1.0)
                
                # Feature 2: fragmentation_local
                features[slot, 2] = fragmentation[slot]
                
                # Feature 3: edge_proximity  
                edge_distance = min(slot, num_slots - 1 - slot) / (num_slots / 2.0)
                features[slot, 3] = edge_distance
                
                # Feature 4: utilization_context
                context_start = max(0, slot - 8)
                context_end = min(num_slots, slot + 8 + 1)
                context_utilization = 1.0 - np.mean(available_slots[context_start:context_end])
                # Garantir que está em [0,1]
                context_utilization = min(max(context_utilization, 0.0), 1.0)
                features[slot, 4] = context_utilization
                
                # DEBUG comentado para logs estruturados
                # if slot == 0:
                #     print(f"  Slot {slot} (disponível):")
                #     print(f"    is_available: 1.0")
                #     print(f"    local_block_size: {local_block_size} → norm: {features[slot, 1]:.6f}")
                #     print(f"    fragmentation: {fragmentation[slot]:.6f}")
                #     print(f"    edge_distance: min({slot}, {num_slots-1-slot}) / {num_slots/2.0} → {edge_distance:.6f}")
                #     print(f"    context_util: window[{context_start}:{context_end}] → {context_utilization:.6f}")
                
            else:  # Slot indisponível
                # Todas as features = -1 para slots indisponíveis
                features[slot, 0] = -1.0
                features[slot, 1] = -1.0
                features[slot, 2] = -1.0
                features[slot, 3] = -1.0
                features[slot, 4] = -1.0
        
        return features

    cdef cnp.ndarray _compute_path_specific_features(self, object path, cnp.ndarray available_slots, 
                                                   cnp.ndarray osnr_values, list best_modulations):
        """Calcula features específicas do path - 6 features por slot"""
        cdef int num_slots = self.num_spectrum_resources
        cdef cnp.ndarray features = np.zeros((num_slots, 6), dtype=np.float32)
        cdef int slot, best_mod_idx, required_mod_idx
        cdef double osnr_min, osnr_max, osnr_range, path_quality, osnr_margin
        cdef double spectral_efficiency, max_se, modulation_efficiency_ratio
        cdef object best_modulation, required_modulation
        
        # DEBUG LIMPO: path-specific resumidas (comentado para logs estruturados)
        # print(f"\n🔍 DEBUG FEATURES PATH-ESPECÍFICAS - Path {path.k}:")
        
        # OSNR normalização POR PATH
        valid_osnr = osnr_values[osnr_values >= 0]
        if len(valid_osnr) > 0:
            osnr_min = np.min(valid_osnr)
            osnr_max = np.max(valid_osnr)
            osnr_range = osnr_max - osnr_min if osnr_max > osnr_min else 1.0
        else:
            osnr_min = 0.0
            osnr_max = 1.0
            osnr_range = 1.0
        
        # print(f"  OSNR range: min={osnr_min:.3f}, max={osnr_max:.3f}, range={osnr_range:.3f}")
        
        # Path quality POR PATH
        path_quality = 1.0 - min(path.length / 2000.0, 1.0)
        # print(f"  Path quality: length={path.length:.1f} → quality={path_quality:.6f}")
        
        # Encontrar modulação mínima necessária para o serviço atual
        required_mod_idx = 0
        for mod_idx, modulation in enumerate(self.modulations):
            if self.get_number_slots(self.current_service, modulation) <= self.num_spectrum_resources:
                required_mod_idx = mod_idx
                break
        # print(f"  Required modulation idx: {required_mod_idx}")
        
        for slot in range(num_slots):
            if available_slots[slot] == 1:  # Slot disponível
                # Feature 0: osnr_normalized (POR PATH)
                if osnr_values[slot] >= 0:
                    features[slot, 0] = (osnr_values[slot] - osnr_min) / osnr_range
                else:
                    features[slot, 0] = 0.0
                
                # Feature 1: osnr_margin_best_mod (POR PATH)
                best_mod_idx = best_modulations[slot]
                if best_mod_idx >= 0:
                    best_modulation = self.modulations[best_mod_idx]
                    osnr_margin = osnr_values[slot] - best_modulation.minimum_osnr
                    features[slot, 1] = min(osnr_margin / 10.0, 1.0)
                else:
                    features[slot, 1] = 0.0
                
                # Feature 2: path_quality (POR PATH)
                features[slot, 2] = path_quality
                
                # Feature 3: best_modulation_tier (POR PATH)
                if best_mod_idx >= 0:
                    features[slot, 3] = best_mod_idx / (len(self.modulations) - 1)
                else:
                    features[slot, 3] = 0.0
                
                # Feature 4: spectral_efficiency_potential (POR PATH)
                if best_mod_idx >= 0:
                    best_modulation = self.modulations[best_mod_idx]
                    spectral_efficiency = best_modulation.spectral_efficiency
                    max_se = max([mod.spectral_efficiency for mod in self.modulations])
                    features[slot, 4] = spectral_efficiency / max_se
                else:
                    features[slot, 4] = 0.0
                
                # Feature 5: modulation_efficiency_ratio (NOVA - POR PATH)
                if best_mod_idx >= 0 and required_mod_idx < len(self.modulations):
                    required_modulation = self.modulations[required_mod_idx]
                    best_modulation = self.modulations[best_mod_idx]
                    modulation_efficiency_ratio = best_modulation.spectral_efficiency / required_modulation.spectral_efficiency
                    # Normalizar adequadamente: ratio geralmente está entre 1.0 e 4.0
                    features[slot, 5] = min((modulation_efficiency_ratio - 1.0) / 3.0, 1.0)  # Mapear [1,4] -> [0,1]
                else:
                    features[slot, 5] = 0.0
                
                # DEBUG comentado para logs estruturados
                # if slot == 0:
                #     print(f"  Slot {slot} (disponível):")
                #     print(f"    osnr_raw: {osnr_values[slot]:.3f} → norm: {features[slot, 0]:.6f}")
                #     print(f"    best_mod_idx: {best_mod_idx}")
                #     if best_mod_idx >= 0:
                #         print(f"    osnr_margin: {osnr_margin:.3f} → norm: {features[slot, 1]:.6f}")
                #         print(f"    best_mod_tier: {best_mod_idx}/{len(self.modulations)-1} → {features[slot, 3]:.6f}")
                #         print(f"    spectral_eff: {spectral_efficiency:.2f}/{max_se:.2f} → {features[slot, 4]:.6f}")
                #         print(f"    mod_eff_ratio: {modulation_efficiency_ratio:.3f} → {features[slot, 5]:.6f}")
                #     else:
                #         print(f"    No viable modulation for this slot")
                #     print(f"    path_quality: {features[slot, 2]:.6f}")
                
            else:  # Slot indisponível
                # Todas as features = -1 para slots indisponíveis
                features[slot, 0] = -1.0
                features[slot, 1] = -1.0
                features[slot, 2] = -1.0
                features[slot, 3] = -1.0
                features[slot, 4] = -1.0
                features[slot, 5] = -1.0
        
        return features

    cdef int _get_local_block_size(self, int slot, cnp.ndarray available_slots):
        """Retorna o tamanho do bloco livre que contém o slot dado"""
        if available_slots[slot] == 0:
            return 0
            
        cdef int start = slot
        cdef int end = slot
        cdef int num_slots = self.num_spectrum_resources
        
        # Expandir para a esquerda
        while start > 0 and available_slots[start - 1] == 1:
            start -= 1
        
        # Expandir para a direita  
        while end < num_slots - 1 and available_slots[end + 1] == 1:
            end += 1
            
        return end - start + 1

    cdef cnp.ndarray _compute_path_utilizations(self):
        """Calcula utilização atual de cada path - NOVO para load balancing"""
        cdef cnp.ndarray utilizations = np.zeros(self.k_paths, dtype=np.float32)
        
        source = self.current_service.source
        destination = self.current_service.destination
        
        for path_idx, route in enumerate(self.k_shortest_paths[source, destination][:self.k_paths]):
            available_slots = self.get_available_slots(route)
            utilization = 1.0 - np.mean(available_slots)
            utilizations[path_idx] = utilization
        
        return utilizations

    cdef int _count_contiguous_from_slot(self, int start_slot, cnp.ndarray available_slots):
        """Conta slots contíguos disponíveis a partir de um slot"""
        cdef int count = 0
        cdef int slot = start_slot
        
        while slot < len(available_slots) and available_slots[slot] == 1:
            count += 1
            slot += 1
        
        return count

    cdef int _count_available_positions(self, cnp.ndarray available_slots, int slots_required, str cache_key=None):
        """Conta quantas posições válidas existem para alocar um bloco de tamanho slots_required usando máscara cacheada."""
        cdef int total_slots = available_slots.shape[0]
        cdef cnp.ndarray mask
        if slots_required <= 0 or slots_required > total_slots:
            return 0
        if cache_key is None:
            cache_key = f"slots_{id(available_slots)}"
        mask = self._get_cached_window_mask(available_slots, cache_key, slots_required)
        if mask.size == 0:
            return 0
        return int(mask.sum())

    cdef cnp.ndarray _compute_path_modulation_features(self, list paths_info, cnp.ndarray path_utilizations):
        """Computa features por path/modulação - K × M × 2"""
        cdef int K = len(paths_info)
        cdef int M = self.modulations_to_consider  
        cdef cnp.ndarray features = np.zeros((K, M, 2), dtype=np.float32)
        
        # Cache para modulações
        cdef tuple modulations = self.modulations
        cdef double max_spectral_eff = max([mod.spectral_efficiency for mod in modulations])
        
        for path_idx in range(K):
            route, available_slots = paths_info[path_idx]
            path_length = route.length
            cache_key = f"path_{path_idx}"
            
            for mod_idx in range(M):
                if mod_idx >= len(modulations):
                    continue
                    
                modulation = modulations[mod_idx]
                
                # Feature 0: qot_path_quality 
                # Combina: distância + eficiência espectral + utilização
                distance_penalty = min(path_length / 2000.0, 1.0)
                efficiency_bonus = modulation.spectral_efficiency / max_spectral_eff
                utilization_penalty = path_utilizations[path_idx]
                
                qot_quality = (efficiency_bonus * 0.5) + ((1.0 - distance_penalty) * 0.3) + ((1.0 - utilization_penalty) * 0.2)
                features[path_idx, mod_idx, 0] = qot_quality
                
                # Feature 1: allocation_competitiveness
                # Quantas posições disponíveis para esta modulação neste path
                slots_required = self.get_number_slots(self.current_service, modulation)
                available_positions = self._count_available_positions(available_slots, slots_required, cache_key)
                max_possible_positions = max(1, self.num_spectrum_resources - slots_required + 1)
                
                competitiveness = available_positions / max_possible_positions
                features[path_idx, mod_idx, 1] = competitiveness
        
        return features

    cdef cnp.ndarray _compute_global_slot_features_optimized(self, cnp.ndarray available_slots_global):
        """Computa features globais otimizadas por slot - S × 4 (removeu utilization_context)"""
        cdef int num_slots = len(available_slots_global)
        cdef cnp.ndarray features = np.zeros((num_slots, 4), dtype=np.float32)
        
        # Calcular fragmentação global uma vez
        cdef cnp.ndarray fragmentation = self._get_cached_fragmentation(available_slots_global, "global_optimized")
        
        for slot in range(num_slots):
            if available_slots_global[slot] == 1:  # Disponível
                # Feature 0: is_available
                features[slot, 0] = 1.0
                
                # Feature 1: local_block_size
                local_block_size = self._get_local_block_size(slot, available_slots_global)
                features[slot, 1] = min(local_block_size / 50.0, 1.0)
                
                # Feature 2: fragmentation_local
                features[slot, 2] = fragmentation[slot]
                
                # Feature 3: edge_proximity  
                edge_distance = min(slot, num_slots - 1 - slot) / (num_slots / 2.0)
                features[slot, 3] = edge_distance
                
            else:  # Slot indisponível
                features[slot, 0] = 0.0
                features[slot, 1] = -1.0
                features[slot, 2] = -1.0
                features[slot, 3] = -1.0
        
        return features

    cdef cnp.ndarray _compute_path_slot_features_optimized(self, list paths_info, cnp.ndarray path_utilizations):
        """Computa features otimizadas por path/slot - K × S × 3"""
        cdef int K = len(paths_info) 
        cdef int S = self.num_spectrum_resources
        cdef cnp.ndarray features = np.zeros((K, S, 3), dtype=np.float32)
        
        # Calcular utilização média dos outros paths para load balancing
        cdef double mean_other_utilization
        
        for path_idx in range(K):
            route, available_slots = paths_info[path_idx]
            
            # OSNR vectorizado para todo o path de uma vez (OTIMIZAÇÃO)
            osnr_values = self._compute_slot_osnr_vectorized(route, available_slots)
            
            # Normalização OSNR para este path
            valid_osnr = osnr_values[osnr_values >= 0]
            if len(valid_osnr) > 0:
                osnr_min, osnr_max = np.min(valid_osnr), np.max(valid_osnr)
                osnr_range = osnr_max - osnr_min if osnr_max > osnr_min else 1.0
            else:
                osnr_min, osnr_max, osnr_range = 0.0, 1.0, 1.0
            
            # Load balancing: utilização média dos outros paths
            other_paths_util = np.concatenate([path_utilizations[:path_idx], path_utilizations[path_idx+1:]])
            mean_other_utilization = np.mean(other_paths_util) if len(other_paths_util) > 0 else 0.0
            current_path_utilization = path_utilizations[path_idx]
            
            for slot in range(S):
                if available_slots[slot] == 1:  # Disponível
                    # Feature 0: osnr_normalized (mantida)
                    if osnr_values[slot] >= 0:
                        features[path_idx, slot, 0] = (osnr_values[slot] - osnr_min) / osnr_range
                    else:
                        features[path_idx, slot, 0] = 0.0
                    
                    # Feature 1: contiguous_capacity (NOVA)
                    contiguous_from_slot = self._count_contiguous_from_slot(slot, available_slots)
                    features[path_idx, slot, 1] = min(contiguous_from_slot / 50.0, 1.0)
                    
                    # Feature 2: load_balancing_value (NOVA)
                    load_imbalance = mean_other_utilization - current_path_utilization
                    load_balancing_value = min(max(0.5 + load_imbalance, 0.0), 1.0)
                    features[path_idx, slot, 2] = load_balancing_value
                    
                else:  # Indisponível
                    features[path_idx, slot, 0] = -1.0
                    features[path_idx, slot, 1] = -1.0  
                    features[path_idx, slot, 2] = -1.0
        
        return features


    cpdef public is_path_free(self, path, initial_slot: int, number_slots: int):
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

    cdef double _compute_fragmentation_score_from_info(self, dict info):
        """Calcula score de fragmentação usando métricas já computadas no step()"""
        cdef double shannon, route_cuts_norm, route_rss, frag_score
        cdef int num_links, route_cuts
        
        # Obter métricas do dict info (já calculadas no step())
        shannon = info.get("fragmentation_shannon_entropy", 0.0)
        route_cuts = info.get("fragmentation_route_cuts", 0)
        route_rss = info.get("fragmentation_route_rss", 0.0)
        
        # Normalizar route_cuts pelo número de links (tipicamente 0-2 cuts por link)
        num_links = self.topology.number_of_edges()
        if num_links > 0:
            route_cuts_norm = min(<double>route_cuts / (<double>num_links * 2.0), 1.0)
        else:
            route_cuts_norm = 0.0
        
        # Score combinado: pesos balanceados
        # shannon: [0,1] - maior = mais fragmentado
        # route_cuts_norm: [0,1] - maior = mais cortes
        # route_rss: [0,1] - maior = mais fragmentado
        frag_score = 0.4 * shannon + 0.3 * route_cuts_norm + 0.3 * route_rss
        
        return frag_score

    cpdef double reward(self, dict info=None):
        """
        Reward function otimizada para PPO.
        
        Para serviços ACEITOS:
            reward = 0.8 + 0.3×(SE/SE_max) - 0.3×frag_score - 0.3×osnr_waste
            
        Para BLOQUEIOS:
            reject = -1.8 (ação consciente de rejeitar)
            block_recursos = -2.0 (impossível alocar)
            block_osnr = -2.0 (qualidade insuficiente)
        
        NOTA: Não usa failed_ratio pois PPO já gerencia penalidades adaptativas
              através do advantage normalization e clipping.
        """
        cdef double reward_value
        cdef double modulation_bonus, fragmentation_penalty, osnr_waste_penalty
        cdef double current_se, max_se, se_normalized
        cdef double mod_osnr_waste, osnr_waste_normalized
        cdef double frag_score
        
        # ====================================
        # CASO 1: SERVIÇO NÃO ACEITO
        # ====================================
        if not self.current_service.accepted:
            # Penalidades fixas e simples (PPO gerencia o resto)
            if self.current_service.blocked_due_to_resources:
                return -2.0  # Bloqueio por falta de recursos
            elif self.current_service.blocked_due_to_osnr:
                return -2.0  # Bloqueio por OSNR (pior, pois indica má escolha de path/mod)
            else:
                return -1.8  # Reject explícito (ação válida, menor penalidade)
        
        # ====================================
        # CASO 2: SERVIÇO ACEITO
        # ====================================
        reward_value = 0.8

        # Bônus por eficiência espectral
        current_se = self.current_service.current_modulation.spectral_efficiency
        max_se = max([mod.spectral_efficiency for mod in self.modulations])
        se_normalized = current_se / max_se if max_se > 0 else 0.0
        modulation_bonus = 0.3 * se_normalized
        
        # Penalidade por fragmentação
        if info is not None:
            frag_score = self._compute_fragmentation_score_from_info(info)
        else:
            frag_score = 0.0 
        fragmentation_penalty = 0.3 * frag_score

        # Penalidade por desperdício de OSNR (usar modulação inferior ao possível)
        if self.current_service.current_modulation != self.modulations[self.max_modulation_idx]:
            mod_osnr_waste = self.current_service.OSNR - self.modulations[self.max_modulation_idx].minimum_osnr
            # Normalizar para [0, 1]: valores positivos de mod_osnr_waste indicam desperdício
            # Limitar a 3dB como referência (desperdício > 3dB = penalidade máxima)
            osnr_waste_normalized = min(max(mod_osnr_waste / 3.0, 0.0), 1.0)
            osnr_waste_penalty = 0.3 * osnr_waste_normalized
        else:
            mod_osnr_waste = 0.0
            osnr_waste_penalty = 0.0
        
        # Reward final
        reward_value = reward_value + modulation_bonus - fragmentation_penalty - osnr_waste_penalty
        
        # Debug opcional
        if hasattr(self, 'debug_reward') and self.debug_reward:
            print(f"  💰 REWARD DEBUG (service {self.current_service.service_id}):")
            print(f"     Base: +0.8")
            print(f"     Modulation ({self.current_service.current_modulation.name}, SE={current_se:.2f}): +{modulation_bonus:.3f}")
            print(f"     Fragmentation (score={frag_score:.3f}): -{fragmentation_penalty:.3f}")
            print(f"     OSNR Waste ({mod_osnr_waste:.2f}dB): -{osnr_waste_penalty:.3f}")
            print(f"     ➜ TOTAL: {reward_value:.3f}")
        
        # Clipping suave para evitar valores extremos (PPO funciona melhor com rewards bounded)
        if reward_value > 2.0:
            reward_value = 2.0
        elif reward_value < -2.0:
            reward_value = -2.0
        
        return reward_value

    
    cpdef _provision_path(self, object path, cnp.int64_t initial_slot, int number_slots):
        cdef int i, path_length, link_index
        cdef int start_slot = initial_slot
        cdef int end_slot = start_slot + number_slots
        cdef tuple node_list = path.get_node_list() 
        cdef object link  
        cdef dict link_data

        if end_slot < self.num_spectrum_resources:
            end_slot+=1
        elif end_slot > self.num_spectrum_resources:
            raise ValueError("End slot is greater than the number of spectrum resources.")
            
        path_length = len(node_list)
        
        for i in range(path_length - 1):
            link_data = self.link_cache.get_link_data(node_list[i], node_list[i + 1])
            link_index = link_data['index']
            
            self.topology.graph["available_slots"][
                link_index,
                start_slot:end_slot
            ] = 0

            self.spectrum_slots_allocation[
                link_index,
                start_slot:end_slot
            ] = self.current_service.service_id

            self.link_cache.update_link_services(
                node_list[i], 
                node_list[i + 1], 
                self.current_service, 
                add=True
            )

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
    
    cpdef public _release_path(self, service: Service):
        cdef int i, link_index, start_slot, end_slot
        cdef dict link_data
        
        start_slot = service.initial_slot
        end_slot = service.initial_slot + service.number_slots + 1
        
        for i in range(len(service.path.node_list) - 1):
            link_data = self.link_cache.get_link_data(
                service.path.node_list[i], 
                service.path.node_list[i + 1]
            )
            link_index = link_data['index']
            
            self.topology.graph["available_slots"][
                link_index,
                start_slot:end_slot,
            ] = 1
            
            self.spectrum_slots_allocation[
                link_index,
                start_slot:end_slot,
            ] = -1
            
            self.link_cache.update_link_services(
                service.path.node_list[i], 
                service.path.node_list[i + 1], 
                service, 
                add=False
            )

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef cnp.ndarray get_available_slots(self, object path):
        cdef Py_ssize_t i, j, n
        cdef tuple node_list = path.node_list
        cdef cnp.ndarray available_slots_matrix
        cdef cnp.ndarray product
        cdef int[:, :] slots_view
        cdef int[:] product_view
        cdef Py_ssize_t num_rows, num_cols
        cdef dict link_data
        cdef int path_k = path.k
        cdef str cache_key
        
        cache_key = f"avail_slots_{path_k}"

        n = len(node_list) - 1

        cdef cnp.ndarray[cnp.int64_t, ndim=1] indices = np.empty(n, dtype=np.int64)

        for i in range(n):
            link_data = self.link_cache.get_link_data(node_list[i], node_list[i + 1])
            indices[i] = link_data['index']

        available_slots_matrix = self.topology.graph["available_slots"][indices, :]
        current_signature = int(np.sum(available_slots_matrix))

        if cache_key in self.available_slots_cache:
            cached_signature = self.available_slots_signature_cache.get(cache_key, None)
            if cached_signature is not None and cached_signature == current_signature:
                return self.available_slots_cache[cache_key]

        num_rows = available_slots_matrix.shape[0]
        num_cols = available_slots_matrix.shape[1]

        slots_view = available_slots_matrix

        product = available_slots_matrix[0].copy()
        product_view = product

        for i in range(1, num_rows):
            for j in range(num_cols):
                if product_view[j] == 0:
                    continue
                product_view[j] = slots_view[i, j]

        self.available_slots_cache[cache_key] = product
        self.available_slots_signature_cache[cache_key] = current_signature
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


    cpdef list _get_spectrum_slots(self, int path):
        cdef dict link_data
        cdef int link_index
        
        spectrum_route = []
        for link in self.k_shortest_paths[
            self.current_service.source, 
            self.current_service.destination
        ][path].links:
            link_data = self.link_cache.get_link_data(link.node1, link.node2)
            link_index = link_data['index']
            spectrum_route.append(self.topology.graph["available_slots"][link_index, :])

        return spectrum_route


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef defragment(self, int num_services):
        self.episode_defrag_cicles += 1
        if num_services == 0:
            num_services = 1000000

        cdef int moved = 0
        cdef Service service
        cdef int number_slots, candidate, i, path_length, link_index
        cdef int start_slot, end_slot, old_start_slot, old_end_slot
        cdef object path, candidates
        cdef cnp.ndarray available_slots
        cdef tuple node_list
        cdef dict link_data

        cdef list active_services = list(self.topology.graph["running_services"])

        cdef int old_initial_slot = 0
        cdef double old_center_frequency = 0.0
        cdef double old_bandwidth = 0.0
        cdef double osnr, ase, nli
        
        for service in active_services:
            if moved >= num_services:
                break
            
            old_initial_slot = service.initial_slot
            old_center_frequency = service.center_frequency
            old_bandwidth = service.bandwidth

            path = service.path
            number_slots = service.number_slots
            node_list = path.get_node_list()
            path_length = len(node_list)

            old_start_slot = old_initial_slot
            old_end_slot = old_initial_slot + number_slots
            if old_end_slot < self.num_spectrum_resources:
                old_end_slot += 1

            # Temporary release
            for i in range(path_length - 1):
                link_data = self.link_cache.get_link_data(node_list[i], node_list[i+1])
                link_index = link_data['index']
                self.topology.graph["available_slots"][link_index, old_start_slot:old_end_slot] = 1
                self.spectrum_slots_allocation[link_index, old_start_slot:old_end_slot] = -1

            available_slots = self.get_available_slots(path)
            candidates = self._get_candidates(available_slots, number_slots, self.num_spectrum_resources)

            if not candidates:

                for i in range(path_length - 1):
                    link_data = self.link_cache.get_link_data(node_list[i], node_list[i+1])
                    link_index = link_data['index']
                    self.topology.graph["available_slots"][link_index, old_start_slot:old_end_slot] = 0
                    self.spectrum_slots_allocation[link_index, old_start_slot:old_end_slot] = service.service_id
                continue

            best_candidate = None
            for candidate in candidates:
                if candidate < service.initial_slot:
                    best_candidate = candidate
                    break

            if best_candidate is None:
                for i in range(path_length - 1):
                    link_data = self.link_cache.get_link_data(node_list[i], node_list[i+1])
                    link_index = link_data['index']
                    self.topology.graph["available_slots"][link_index, old_start_slot:old_end_slot] = 0
                    self.spectrum_slots_allocation[link_index, old_start_slot:old_end_slot] = service.service_id
                continue

            start_slot = best_candidate
            end_slot = start_slot + number_slots
            if end_slot < self.num_spectrum_resources:
                end_slot += 1
            elif end_slot > self.num_spectrum_resources:

                for i in range(path_length - 1):
                    link_data = self.link_cache.get_link_data(node_list[i], node_list[i+1])
                    link_index = link_data['index']
                    self.topology.graph["available_slots"][link_index, old_start_slot:old_end_slot] = 0
                    self.spectrum_slots_allocation[link_index, old_start_slot:old_end_slot] = service.service_id
                continue


            # Update service for OSNR test
            service.initial_slot = start_slot
            service.center_frequency = self.frequency_start + (
                self.frequency_slot_bandwidth * start_slot
            ) + (
                self.frequency_slot_bandwidth * (number_slots / 2.0)
            )
            service.bandwidth = self.frequency_slot_bandwidth * number_slots
            service.launch_power = self.launch_power

            if self.qot_constraint == "DIST":
                path_distance = service.path.length
                qot_acceptable = path_distance <= service.current_modulation.maximum_length
                osnr = 0.0 if qot_acceptable else -1.0
                ase = 0.0
                nli = 0.0
            else:
                osnr, ase, nli = calculate_osnr(self, service, self.qot_constraint)
                osnr_required = service.current_modulation.minimum_osnr
                qot_acceptable = osnr >= osnr_required
            
            if not qot_acceptable:
                service.initial_slot = old_initial_slot
                service.center_frequency = old_center_frequency
                service.bandwidth = old_bandwidth
                
                # Restore original allocation
                for i in range(path_length - 1):
                    link_data = self.link_cache.get_link_data(node_list[i], node_list[i+1])
                    link_index = link_data['index']
                    self.topology.graph["available_slots"][link_index, old_start_slot:old_end_slot] = 0
                    self.spectrum_slots_allocation[link_index, old_start_slot:old_end_slot] = service.service_id
                continue

            # Confirm new allocation
            for i in range(path_length - 1):
                link_data = self.link_cache.get_link_data(node_list[i], node_list[i+1])
                link_index = link_data['index']
                self.topology.graph["available_slots"][link_index, start_slot:end_slot] = 0
                self.spectrum_slots_allocation[link_index, start_slot:end_slot] = service.service_id

            service.OSNR = osnr
            service.ASE = ase
            service.NLI = nli

            moved += 1
            self.episode_service_realocations += 1
        
        return

    cpdef public close(self):
        return super().close()
