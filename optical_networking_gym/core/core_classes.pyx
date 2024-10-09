cdef class Span:
    cdef public float length
    cdef public float attenuation_db_km
    cdef public float attenuation_normalized
    cdef public float noise_figure_db
    cdef public float noise_figure_normalized

    def __init__(self, float length, float attenuation, float noise_figure):
        self.length = length

        # Calculate attenuation normalization (dB/km to 1/m)
        self.attenuation_db_km = attenuation
        self.attenuation_normalized = self.attenuation_db_km / (2 * 10 * log(exp(1)) * 1e3)  # dB/km -> 1/m

        # Set noise figure and normalize
        self.noise_figure_db = noise_figure
        self.noise_figure_normalized = 10 ** (self.noise_figure_db / 10)  # dB -> normalized

    def set_attenuation(self, float attenuation):
        self.attenuation_db_km = attenuation
        self.attenuation_normalized = self.attenuation_db_km / (2 * 10 * log(exp(1)) * 1e3)  # dB/km -> 1/m

    def set_noise_figure(self, float noise_figure):
        self.noise_figure_db = noise_figure
        self.noise_figure_normalized = 10 ** (self.noise_figure_db / 10)  # dB -> normalized


cdef class Link:
    cdef public int id
    cdef public str node1
    cdef public str node2
    cdef public float length
    cdef public tuple spans  

    def __init__(self, int id, str node1, str node2, float length, tuple spans):
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.length = length
        self.spans = spans  


cdef class Path:
    cdef public int id
    cdef public int k
    cdef public tuple node_list   
    cdef public tuple links  
    cdef public int hops
    cdef public float length
    cdef public Modulation best_modulation  

    def __init__(self, int id, int k, tuple node_list, tuple links, int hops, float length, Modulation best_modulation=None):
        self.id = id
        self.k = k
        self.node_list = node_list
        self.links = links
        self.hops = hops
        self.length = length
        self.best_modulation = best_modulation  


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
    cdef public int center_frequency
    cdef public int bandwidth
    cdef public int number_slots
    cdef public int core
    cdef public float launch_power
    cdef public bint accepted
    cdef public bint blocked_due_to_resources
    cdef public bint blocked_due_to_osnr
    cdef public float OSNR
    cdef public float ASE
    cdef public float NLI
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

cdef class Modulation:
    cdef public str name
    cdef public double maximum_length  
    cdef public int spectral_efficiency
    cdef public double minimum_osnr  
    cdef public double inband_xt  

    def __init__(self, str name, double maximum_length, int spectral_efficiency, 
                 double minimum_osnr=-1.0, double inband_xt=-1.0):
        """
        Constructor for the Modulation class.
        
        :param name: The name of the modulation format.
        :param maximum_length: The maximum length in km.
        :param spectral_efficiency: The number of bits per Hz per sec.
        :param minimum_osnr: Minimum OSNR that allows it to work (optional, defaults to -1.0).
        :param inband_xt: Maximum in-band cross-talk (optional, defaults to -1.0).
        """
        self.name = name
        self.maximum_length = maximum_length
        self.spectral_efficiency = spectral_efficiency
        self.minimum_osnr = minimum_osnr if minimum_osnr >= 0 else -1
        self.inband_xt = inband_xt if inband_xt >= 0 else -1
