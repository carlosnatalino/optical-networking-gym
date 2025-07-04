def setup_environment(num_episodes: int, episode_length: int, load: int, num_slots: int):
    import random
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from optical_networking_gym.topology import Modulation, get_topology
    from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper

    ###################################
    # Define Modulations
    ###################################
    def define_modulations():
        return (
            Modulation(
                name="BPSK",
                maximum_length=100_000,  
                spectral_efficiency=1,
                minimum_osnr=12.6,
                inband_xt=-14,
            ),
            Modulation(
                name="QPSK",
                maximum_length=2_000,
                spectral_efficiency=2,
                minimum_osnr=12.6,
                inband_xt=-17,
            ),
            Modulation(
                name="8QAM",
                maximum_length=1_000,
                spectral_efficiency=3,
                minimum_osnr=18.6,
                inband_xt=-20,
            ),
            Modulation(
                name="16QAM",
                maximum_length=500,
                spectral_efficiency=4,
                minimum_osnr=22.4,
                inband_xt=-23,
            ),
            Modulation(
                name="32QAM",
                maximum_length=250,
                spectral_efficiency=5,
                minimum_osnr=26.4,
                inband_xt=-26,
            ),
            Modulation(
                name="64QAM",
                maximum_length=125,
                spectral_efficiency=6,
                minimum_osnr=30.4,
                inband_xt=-29,
            ),
        )

    cur_modulations = define_modulations()

    ###################################
    # Load Topology
    ###################################
    topology_name = "ring_4"
    topology_path = (
        rf"C:\\Users\\talle\\Documents\\Mestrado\\optical-networking-gym\\examples\\topologies\\{topology_name}.txt"
    )

    topology = get_topology(
        topology_path,
        "Ring4",
        cur_modulations,
        80,    # span length in km
        0.2,   # attenuation in dB/km
        4.5,   # noise figure in dB
        5      # number of shortest paths to precompute
    )

    ###################################
    # Environment Parameters
    ###################################
    seed = 42
    random.seed(seed)

    frequency_slot_bandwidth = 12.5e9
    frequency_start = 3e8 / 1565e-9
    bandwidth = num_slots * frequency_slot_bandwidth
    bit_rates = (10, 40, 80, 100, 400)
    margin = 0

    env_args = dict(
        topology=topology,
        seed=seed,
        allow_rejection=True,
        load=load,
        episode_length=episode_length,
        num_spectrum_resources=num_slots,
        launch_power_dbm=0,  # Fixed launch power
        bandwidth=bandwidth,
        frequency_start=frequency_start,
        frequency_slot_bandwidth=frequency_slot_bandwidth,
        bit_rate_selection="discrete",
        bit_rates=bit_rates,
        margin=margin,
        file_name=f"./results/PPO_setup_env",
        measure_disruptions=False,
        k_paths=2,
    )

    env_id = 'QRMSAEnvWrapper-v0'

    ###################################
    # Observation Wrapper
    ###################################
    class ObservationOnlyWrapper(gym.Wrapper):
        def __init__(self, env):
            super(ObservationOnlyWrapper, self).__init__(env)
            reset_obs, _ = env.reset()
            self.observation_space = spaces.Box(
                low=np.float32(np.min(reset_obs['observation'])),
                high=np.float32(np.max(reset_obs['observation'])),
                shape=reset_obs['observation'].shape,
                dtype=np.float32
            )

        def reset(self, **kwargs):
            obs, info = self.env.reset()
            return obs['observation'], info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return obs['observation'], reward, terminated, truncated, info

    ###################################
    # Environment Creation Function
    ###################################
    def make_env():
        env = gym.make(env_id, **env_args)
        env = ObservationOnlyWrapper(env)
        return env

    return make_env