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
import typing
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
    cdef public float launch_power
    cdef public bint accepted
    cdef public bint blocked_due_to_resources
    cdef public bint blocked_due_to_osnr
    cdef public double OSNR
    cdef public double ASE
    cdef public double NLI
    cdef public object current_modulation
    cdef public bint recalculate  

    def __init__(self, int service_id, str source, int source_id, str destination=None,
                 str destination_id=None, float arrival_time=0.0, float holding_time=0.0,
                 float bit_rate=0.0, object path=None, int service_class=0,
                 int initial_slot=0, int center_frequency=0, int bandwidth=0,
                 int number_slots=0, int core=0, float launch_power=0.0,
                 bint accepted=False, bint blocked_due_to_resources=True, bint blocked_due_to_osnr=True,
                 float OSNR=0.0, float ASE=0.0, float NLI=0.0, object current_modulation=None):

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
        return (f"Service(service_id={self.service_id}, source='{self.source}', source_id={self.source_id}, "
                f"destination='{self.destination}', destination_id={self.destination_id}, arrival_time={self.arrival_time}, "
                f"holding_time={self.holding_time}, bit_rate={self.bit_rate}, path={self.path}, service_class={self.service_class}, "
                f"initial_slot={self.initial_slot}, center_frequency={self.center_frequency}, bandwidth={self.bandwidth}, "
                f"number_slots={self.number_slots}, core={self.core}, launch_power={self.launch_power}, accepted={self.accepted}, "
                f"blocked_due_to_resources={self.blocked_due_to_resources}, blocked_due_to_osnr={self.blocked_due_to_osnr}, "
                f"OSNR={self.OSNR}, ASE={self.ASE}, NLI={self.NLI}, current_modulation={self.current_modulation}, "
                f"recalculate={self.recalculate})")



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

        public Service current_service
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
        str file_stats

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
        k_paths: int = 5,
        file_name: str = ""
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
        if file_name != "":
            final_name = "_".join([file_name, str(self.topology.graph["name"]), str(self.launch_power_dbm), str(self.load), str(seed) + ".csv"])
            self.file_stats = open(final_name, "wt", encoding="UTF-8")
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
        
        # Create a padded array to store service IDs
        padded_running_services = np.full((1000,), fill_value=-1, dtype=int)
        
        # Extract the service_id from each Service object and store it in the array
        service_ids = [service.service_id for service in running_services[:1000]]
        
        # Store the service IDs in the padded array
        padded_running_services[:len(service_ids)] = service_ids

        return {
            "topology": available_slots,
            "running-services": padded_running_services
        }


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
            print(self.current_service.__repr__())
            print(self.topology.graph['available_slots'])
            print(info)
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
        cdef object service
        cdef int time, src_id
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

        print(
            f"{self.current_service.service_id} assigning path {node_list} on initial slot {start_slot} for {number_slots} slots"
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
            self.frequency_slot_bandwidth * (number_slots / 2)
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
        heapq.heappush(self._events, (release_time, service))
    
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
        cdef double last_update, time_diff, last_util, cur_util, utilization
        cdef double cur_external_fragmentation, cur_link_compactness
        cdef double external_fragmentation, link_compactness
        cdef int used_spectrum_slots, max_empty, lambda_min, lambda_max
        cdef object link  # Assuming this is a dict-like object
        cdef cnp.ndarray[cnp.int32_t, ndim=1] slot_allocation  # Typed NumPy array
        cdef list initial_indices, values, lengths, unused_blocks, used_blocks
        cdef double last_external_fragmentation, last_compactness

        # Get the link between node1 and node2
        link = self.topology[node1][node2]
        last_update = link["last_update"]
        time_diff = self.current_time - last_update

        if self.current_time > 0:
            last_util = link["utilization"]
            slot_allocation = self.topology.graph["available_slots"][link["index"], :]

            # Ensure slot_allocation is a Cython-compatible NumPy array of int32
            slot_allocation = <cnp.ndarray[cnp.int32_t, ndim=1]> np.asarray(slot_allocation, dtype=np.int32)

            # Calculate current utilization
            used_spectrum_slots = self.num_spectrum_resources - np.sum(slot_allocation)
            cur_util = used_spectrum_slots / self.num_spectrum_resources

            # Update utilization using a weighted average
            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            link["utilization"] = utilization

            # Call rle with Cython-compatible NumPy array and convert results to lists
            initial_indices_np, values_np, lengths_np = rle(slot_allocation)
            
            initial_indices = initial_indices_np.tolist()  # Convert NumPy arrays to lists
            values = values_np.tolist()
            lengths = lengths_np.tolist()

            # Compute external fragmentation
            unused_blocks = [i for i, x in enumerate(values) if x == 1]

            # Fix: Get the corresponding values from lengths using list comprehension
            if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
                max_empty = max([lengths[i] for i in unused_blocks])
            else:
                max_empty = 0

            cur_external_fragmentation = 1.0 - (float(max_empty) / float(np.sum(slot_allocation)))

            # Compute link spectrum compactness
            used_blocks = [i for i, x in enumerate(values) if x == 0]
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]

                # Evaluate the used part of the spectrum
                internal_idx_np, internal_values_np, internal_lengths_np = rle(slot_allocation[lambda_min:lambda_max])
                internal_values = internal_values_np.tolist()  # Convert to lists
                unused_spectrum_slots = np.sum(1 - internal_values_np)

                if unused_spectrum_slots > 0:
                    cur_link_compactness = ((lambda_max - lambda_min) / np.sum(1 - slot_allocation)) * (1 / unused_spectrum_slots)
                else:
                    cur_link_compactness = 1.0
            else:
                cur_link_compactness = 1.0

            # Update external fragmentation using a weighted average
            external_fragmentation = ((last_external_fragmentation * last_update) + (cur_external_fragmentation * time_diff)) / self.current_time
            link["external_fragmentation"] = external_fragmentation

            # Update link compactness using a weighted average
            link_compactness = ((last_compactness * last_update) + (cur_link_compactness * time_diff)) / self.current_time
            link["compactness"] = link_compactness

        # Update the last update time
        link["last_update"] = self.current_time

    def close(self):
        return super().close()
