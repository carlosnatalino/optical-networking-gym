from optical_networking_gym import make_env


def create_env(config):
    return make_env(
        scenario="nobel_eu_baseline",
        seed=config["seed"],
        load=config["load"],
        episode_length=config["episode_length"],
        num_spectrum_resources=320,
        k_paths=3,
        modulations_to_consider=4,
        bit_rates=(10, 40, 100, 400),
        modulations="BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM",
        overrides={"topology_id": "nobel-eu"},
    )