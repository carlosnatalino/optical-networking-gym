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
from optical_networking_gym.core.osnr import calculate_osnr
import math


cdef class QRMSAEnv:
    cdef public uint32_t input_seed
    cdef:
        float load
        int episode_length
        float mean_service_holding_time
        int num_spectrum_resources
        float channel_width
        bint allow_rejection
        readonly object topology
        readonly str bit_rate_selection
        readonly tuple bit_rates
        float bit_rate_lower_bound
        float bit_rate_higher_bound
        object bit_rate_probabilities
        object node_request_probabilities
        object k_shortest_paths
        int k_paths

        float launch_power_dbm
        float launch_power
        float bandwidth
        float frequency_start
        float frequency_end
        float frequency_slot_bandwidth
        float margin
        object modulations
        bint measure_disruptions

        public object _np_random
        public int _np_random_seed

        object spectrum_use
        object spectrum_allocation

        object current_service
        int service_id_counter
        list services_in_progress
        list release_times
    

        int services_processed
        int services_accepted
        int episode_services_processed
        int episode_services_accepted
        float bit_rate_requested
        float bit_rate_provisioned
        float episode_bit_rate_requested
        float episode_bit_rate_provisioned
        object bit_rate_requested_histogram
        object bit_rate_provisioned_histogram
        object slots_provisioned_histogram
        object episode_slots_provisioned_histogram

        int disrupted_services
        int episode_disrupted_services
        list disrupted_services_list

        public object action_space
        public object observation_space

        object episode_actions_output
        object episode_actions_taken
        object episode_modulation_histogram
        object episode_bit_rate_requested_histogram
        object episode_bit_rate_provisioned_histogram
        object spectrum_slots_allocation



        int reject_action
        object actions_output
        object actions_taken

        bint _new_service
        float current_time
        float mean_service_inter_arrival_time

        public object frequency_vector

        object rng
        object bit_rate_function
        list _events

    topology: cython.declare(nx.Graph, visibility="readonly")
    bit_rate_selection: cython.declare(Literal["continuous", "discrete"], visibility="readonly")
    bit_rates: cython.declare(tuple[int, int, int] or tuple[float, float, float], visibility="readonly")

    def __init__(
        self,
        topology: nx.Graph,
        num_spectrum_resources: int = 320,
        episode_length: int = 1000,
        load: float = 10.0,
        mean_service_holding_time: float = 10800.0,
        bit_rate_selection: str = "continuous",
        bit_rates: tuple = (10, 40, 100),
        bit_rate_probabilities=None,
        node_request_probabilities=None,
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
        channel_width: float = 12.5,
        k_paths: int = 5
    ):
        # Atributos de inicialização
        self.rng = random.Random()
        self._events = []
        self.mean_service_inter_arrival_time = 0
        self.set_load(load=load, mean_service_holding_time=mean_service_holding_time)

        self.bit_rate_selection = bit_rate_selection
        
        if self.bit_rate_selection == "continuous":
            self.bit_rate_lower_bound = bit_rate_lower_bound
            self.bit_rate_higher_bound = bit_rate_higher_bound

            # creating a partial function for the bit rate continuous selection
            self.bit_rate_function = functools.partial(
                self.rng.randint, int(self.bit_rate_lower_bound), int(self.bit_rate_higher_bound)
            )
        
        elif self.bit_rate_selection == "discrete":
            if bit_rate_probabilities is None:
                bit_rate_probabilities = [1.0 / len(bit_rates) for _ in range(len(bit_rates))]
            
            self.bit_rate_probabilities = bit_rate_probabilities
            self.bit_rates = bit_rates

            # creating a partial function for the discrete bit rate options
            self.bit_rate_function = functools.partial(
                self.rng.choices, self.bit_rates, self.bit_rate_probabilities, k=1
            )

            # defining histograms which are only used for the discrete bit rate selection
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

        # Cálculo da potência de lançamento
        self.launch_power_dbm = launch_power_dbm
        self.launch_power = 10 ** ((self.launch_power_dbm - 30) / 10)  # Convertendo dBm para watts

        # Parâmetros relacionados à frequência
        self.bandwidth = bandwidth
        self.frequency_start = frequency_start
        self.frequency_slot_bandwidth = frequency_slot_bandwidth
        self.margin = margin
        self.measure_disruptions = measure_disruptions

        self.frequency_end = self.frequency_start + self.frequency_slot_bandwidth * self.num_spectrum_resources

        assert math.isclose(self.frequency_end - self.frequency_start, self.bandwidth, rel_tol=1e-5)

        self.frequency_vector = np.linspace(
            self.frequency_start,
            self.frequency_end,
            num=self.num_spectrum_resources,
            dtype=np.float64
        )
        
        assert self.frequency_vector.shape[0] == self.num_spectrum_resources, (
            f"Tamanho do frequency_vector ({self.frequency_vector.shape[0]}) "
            f"não é igual a num_spectrum_resources ({self.num_spectrum_resources})."
        )

        self.topology.graph["available_slots"] = np.ones(
                                                    (self.topology.number_of_edges(), self.num_spectrum_resources),
                                                    dtype=np.int32
                                                )
        self.observation_space = gym.spaces.Dict({"topology": gym.spaces.Box(low=-1, high=1, dtype=int,
                                                    shape=self.topology.graph["available_slots"].shape),
                                                    "running-services":gym.spaces.Box(low=-1, high=np.inf, dtype=int,
                                                    shape=(1000, ))}
                                                    )

        self.modulations = self.topology.graph.get("modulations", [])
        self.disrupted_services_list = []
        self.disrupted_services = 0
        self.episode_disrupted_services = 0

        self.action_space = gym.spaces.MultiDiscrete(
            (self.k_paths ,len(self.modulations), self.frequency_vector.shape[0])
        )

        if seed is None:
            ss = SeedSequence()
            input_seed = int(ss.generate_state(1)[0])
        elif isinstance(seed, int):
            input_seed = int(seed)
        else:
            raise ValueError("Seed must be an integer.")

        input_seed = input_seed % (2**31)  
        if input_seed >= 2**31:  
            input_seed -= 2**32 

        self.input_seed = int(input_seed)  

        self._np_random, self._np_random_seed = seeding.np_random(self.input_seed)

        cdef int num_edges = self.topology.number_of_edges()
        cdef int num_resources = self.num_spectrum_resources

        self.spectrum_use = np.zeros(
            (num_edges, num_resources), dtype=np.int32
        )
        self.spectrum_allocation = np.full(
            (num_edges, num_resources),
            fill_value=-1,
            dtype=np.int64,
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

        if reset:
            self.reset()


    cpdef tuple reset(self, object seed=None, dict options=None):
        self.episode_bit_rate_requested = 0.0
        self.episode_bit_rate_provisioned = 0.0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_disrupted_services = 0

        self.episode_actions_output = np.zeros(
            (self.k_paths + self.reject_action, self.num_spectrum_resources + self.reject_action),
            dtype=np.int32
        )
        self.episode_actions_taken = np.zeros(
            (self.k_paths + self.reject_action, self.num_spectrum_resources + self.reject_action),
            dtype=np.int32
        )

        if self.bit_rate_selection == "discrete":
            self.episode_bit_rate_requested_histogram = defaultdict(int)
            self.episode_bit_rate_provisioned_histogram = defaultdict(int)

        self.episode_modulation_histogram = defaultdict(int)

        if self._new_service:
            self.episode_services_processed += 1
            self.episode_bit_rate_requested = np.int64(self.episode_bit_rate_requested + self.current_service.bit_rate)
            if self.bit_rate_selection == "discrete":
                self.episode_bit_rate_requested_histogram[self.current_service.bit_rate] += 1

        
        gym.Env.reset(self, seed=self.input_seed, options=options)

        if options is not None and "only_episode_counters" in options and options["only_episode_counters"]:
            return self.observation(), {}

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

        return self.observation(), {}


    
    def observation(self):
        available_slots = self.topology.graph["available_slots"]
        running_services = self.topology.graph["running_services"]
        
        padded_running_services = np.full((1000,), fill_value=-1, dtype=int)
        padded_running_services[:len(running_services)] = running_services[:1000]

        return {
            "topology": available_slots,
            "running-services": padded_running_services
        }

    cpdef tuple[object, float, bint, bint, dict] step(self, cnp.ndarray action):
        cdef int route
        cdef Modulation modulation
        cdef int modulation_index
        cdef int initial_slot
        cdef int number_slots
        cdef double osnr, ase, nli
        cdef int disrupted_services = 0
        cdef Path path
        cdef list services_to_measure = []
        cdef Link link
        cdef Service service
        cdef dict info
        cdef str line
        cdef str key
        cdef bint truncated = False
        cdef bint terminated

        if action is not None:
            print(f"action: {action}")
            # Unpack the action tuple
            route = int(action[0])
            modulation_index = int(action[1])
            modulation = self.modulations[modulation_index]
            initial_slot = int(action[2])

            # Get the modulation object
            modulation = self.modulations[modulation_index]

            # Get the path based on the route index
            path = self.k_shortest_paths[
                self.current_service.source,
                self.current_service.destination,
            ][route]

            # Calculate the number of slots required
            number_slots = self.get_number_slots(
                self.current_service,
                modulation,
            )

            # Check if the path is free for the given slots
            if self.is_path_free(path, initial_slot, number_slots):
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
            self.current_service.initial_slot = None
            self.current_service.number_slots = None
            self.current_service.OSNR = None
            self.current_service.ASE = None
            self.current_service.NLI = None

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

        # Update modulation histogram in the info dictionary
        for modulation in self.modulations:
            key = "modulation_{}".format(modulation.spectral_efficiency)
            info[key] = self.episode_modulation_histogram[modulation.spectral_efficiency]

        # Prepare for the next service
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
        cdef float ht
        cdef str src, dst,  dst_id
        cdef float bit_rate
        cdef Service service
        cdef int time, src_id
        cdef Service service_to_release
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
            time, service_to_release = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:
                heapq.heappush(self._events, (time, service_to_release))
                break 
    
    cpdef void set_load(self, float load=-1.0, float mean_service_holding_time=-1.0):
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

    cpdef int get_number_slots(self, Service service, Modulation modulation):
            """
            Computes the number of spectrum slots necessary to accommodate the service request into the path.
            Adds the guardband.
            """
            cdef double required_slots
            required_slots = service.bit_rate / (modulation.spectral_efficiency * self.channel_width)
            return int(math.ceil(required_slots))

    def close(self):
        return super().close()
