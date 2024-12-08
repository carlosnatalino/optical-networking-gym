# %% [markdown]
# # Treinamento PPO com Ambiente QRMSA
#
# Este notebook demonstra como configurar, treinar e acompanhar o treinamento de um modelo PPO
# com o ambiente `QRMSAEnvWrapper-v0` (parte do `optical_networking_gym`).
#
# **Passos:**
# 1. Definir as modulações e a topologia.
# 2. Configurar o ambiente.
# 3. Criar wrappers para ajustes de espaço de observação.
# 4. Definir e treinar o modelo PPO.
# 5. Acompanhar o treinamento pelo TensorBoard.

# %% [markdown]
# ## Instalações e Imports
# Certifique-se de que os pacotes necessários estão instalados.
#
# Se não estiverem, rode:
# `!pip install stable-baselines3 gymnasium optical-networking-gym tensorboard`
#

# %% [code]
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Imports específicos do optical_networking_gym
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.topology import Modulation, get_topology

# %% [markdown]
# ## Definição das Modulações

# %% [code]
def define_modulations() -> Tuple[Modulation, ...]:
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
            name="16QAM",
            maximum_length=500,
            spectral_efficiency=4,
            minimum_osnr=22.4,
            inband_xt=-23,
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

# %% [markdown]
# ## Carregar e Configurar a Topologia
# Ajuste o `topology_name` e o caminho para a topologia conforme necessário.

# %% [code]
topology_name = "ring_4"

topology_path = (
    rf"C:\Users\talle\Documents\Mestrado\optical-networking-gym\examples\topologies\{topology_name}.txt"
    # Ajuste o caminho conforme necessário
)

topology = get_topology(
    topology_path,
    "Ring4",          # Nome da topologia
    cur_modulations,  # Modulações
    80,               # Comprimento máximo do span
    0.2,              # Atenuação
    4.5,              # Figura de ruído
    5                 # Número de caminhos mais curtos
)

# %% [markdown]
# ## Definição dos Parâmetros do Ambiente

# %% [code]
seed = 10
random.seed(seed)

episode_length = 10000
load = 210
threads = 1
launch_power = 0

# Parâmetros do espectro
num_slots = 150
frequency_slot_bandwidth = 12.5e9
frequency_start = 3e8 / 1565e-9
bandwidth = num_slots * frequency_slot_bandwidth
frequency_end = frequency_start + bandwidth
bit_rates = (10, 40, 100, 400)
margin = 0

env_args = dict(
    topology=topology,
    seed=seed,
    allow_rejection=True,
    load=load,
    episode_length=episode_length,
    num_spectrum_resources=num_slots,
    launch_power_dbm=launch_power,
    bandwidth=bandwidth,
    frequency_start=frequency_start,
    frequency_slot_bandwidth=frequency_slot_bandwidth,
    bit_rate_selection="discrete",
    bit_rates=bit_rates,
    margin=margin,
    file_name=f"./results/PPO_{1}",
    measure_disruptions=False,
    k_paths=2,
)

# %% [markdown]
# ## Criação do Ambiente e Verificação
#
# Aqui utilizamos o `check_env` do Stable Baselines3 para garantir que o ambiente está configurado adequadamente.

# %% [code]
env_id = 'QRMSAEnvWrapper-v0'
env = gym.make(env_id, **env_args)
check_env(env)

# %% [markdown]
# ## Wrapper de Observação
#
# Esse wrapper extrai apenas o campo `'observation'` do dicionário retornado pelo ambiente, simplificando assim a entrada para o modelo.

# %% [code]
class ObservationOnlyWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ObservationOnlyWrapper, self).__init__(env)
        reset_obs = env.reset()[0]['observation']
        self.observation_space = spaces.Box(
            low=np.float32(np.min(reset_obs)),
            high=np.float32(np.max(reset_obs)),
            shape=reset_obs.shape,
            dtype=reset_obs.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(options={"only_episode_counters": True})
        return obs['observation'], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs['observation'], reward, terminated, truncated, info


# %% [markdown]
# ## Criação do Ambiente Vetorizado
#
# Criamos um ambiente vetorizado com apenas 1 ambiente, mas isso facilita a compatibilidade com o PPO.
# Caso deseje, é possível aumentar `n_envs` para ter paralelismo.

# %% [code]
def make_env():
    env = gym.make(env_id, **env_args)
    env = ObservationOnlyWrapper(env)
    return env

vec_env = make_vec_env(make_env, n_envs=1)

# %% [markdown]
# ## Definição e Treinamento do Modelo PPO
#
# Ajuste `total_timesteps` conforme necessário. Definimos também o diretório do TensorBoard.

# %% [code]
# Definição do modelo PPO
model = PPO(
    "MlpPolicy",  # Política baseada em redes neurais multi-layer perceptron
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    tensorboard_log="./ppo_qrmsa_tensorboard/"
)
# Treinamento do modelo
model.learn(total_timesteps=1_000_000)  # Ajuste o número de timesteps conforme necessário

# Salvando o modelo treinado
model.save("ppo_qrmsa_model")

# %% [markdown]
# ## Carregando o Modelo Treinado
#
# Você pode futuramente carregar o modelo e avaliar seu desempenho em episódios.

# %% [code]
# modelo_carregado = PPO.load("ppo_qrmsa_model", env=vec_env)

# %% [markdown]
# ## Visualização com TensorBoard
#
# Para visualizar o treinamento, abra um terminal na pasta deste notebook e execute:
#
# ```bash
# tensorboard --logdir=./ppo_qrmsa_tensorboard/
# ```
#
# Depois, abra o link exibido (por exemplo: http://localhost:6006) no seu navegador.

# %% [markdown]
# Após rodar a célula de treinamento, você deve ver os logs sendo gerados no diretório `./ppo_qrmsa_tensorboard/`.
# Isso possibilitará acompanhar métricas como reward médio, perda do modelo, etc.
