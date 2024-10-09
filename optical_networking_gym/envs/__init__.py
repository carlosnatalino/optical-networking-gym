from gymnasium.envs.registration import register

register(
    id='OpticalNetworking_Qrmsa-v0',  
    entry_point='optical_networking_gym.wrappers.qrmsa_gym:QRMSAEnvWrapper', 
    max_episode_steps=10000,  
)
