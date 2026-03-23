#!/usr/bin/env python3
"""
Configurações de hiperparâmetros para treinamento PPO

Este módulo define diferentes conjuntos de hiperparâmetros otimizados para
MaskablePPO em diferentes cenários de treinamento.

Referências:
- Stable-Baselines3 RL Zoo: https://github.com/DLR-RM/rl-baselines3-zoo
- PPO Paper: https://arxiv.org/abs/1707.06347
- MaskablePPO: https://sb3-contrib.readthedocs.io/
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import torch.nn as nn


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial learning rate
        
    Returns:
        Function that computes current learning rate based on remaining progress
        
    Example:
        >>> schedule = linear_schedule(3e-4)
        >>> lr_start = schedule(1.0)  # progress_remaining = 1.0 → lr = 3e-4
        >>> lr_end = schedule(0.0)    # progress_remaining = 0.0 → lr = 0
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        
        Args:
            progress_remaining: Remaining progress (1.0 at start, 0.0 at end)
            
        Returns:
            Current learning rate
        """
        return progress_remaining * initial_value
    
    return func


def exponential_schedule(initial_value: float, decay_rate: float = 0.96) -> Callable[[float], float]:
    """
    Exponential learning rate schedule.
    
    Args:
        initial_value: Initial learning rate
        decay_rate: Decay rate (0 < decay_rate < 1)
        
    Returns:
        Function that computes current learning rate
        
    Example:
        >>> schedule = exponential_schedule(3e-4, 0.96)
        >>> lr = schedule(0.5)  # At 50% progress
    """
    def func(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        return initial_value * (decay_rate ** progress)
    
    return func


def constant_schedule(value: float) -> Callable[[float], float]:
    """
    Constant learning rate (no decay).
    
    Args:
        value: Constant learning rate
        
    Returns:
        Function that always returns the same value
    """
    def func(progress_remaining: float) -> float:
        return value
    
    return func


@dataclass
class PPOHyperparameters:
    """
    Hiperparâmetros para MaskablePPO.
    
    Atributos principais:
        learning_rate: Taxa de aprendizado (pode ser float ou schedule)
        n_steps: Número de steps por environment antes de atualizar
        batch_size: Tamanho do minibatch para SGD
        n_epochs: Número de epochs para otimização
        gamma: Fator de desconto
        gae_lambda: Lambda para Generalized Advantage Estimation
        clip_range: Clipping do PPO
        ent_coef: Coeficiente de entropia (exploration)
        vf_coef: Coeficiente da função de valor
        max_grad_norm: Máximo para gradient clipping
        policy_kwargs: Arquitetura da rede neural
        
    Documentação completa:
    https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    """
    # Learning rate
    learning_rate: float | Callable = 3e-4
    
    # Rollout parameters
    n_steps: int = 2048  # Steps por env antes de update (horizonte de coleta)
    batch_size: int = 64  # Tamanho do minibatch
    n_epochs: int = 10  # Número de epochs de otimização
    
    # Discount factors
    gamma: float = 0.99  # Fator de desconto
    gae_lambda: float = 0.95  # Lambda para GAE
    
    # PPO-specific
    clip_range: float = 0.2  # Clipping range para PPO
    clip_range_vf: Optional[float] = None  # Clipping para value function
    
    # Regularization
    ent_coef: float = 0.0  # Coeficiente de entropia
    vf_coef: float = 0.5  # Coeficiente da função de valor
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Network architecture
    policy_kwargs: Optional[Dict[str, Any]] = None
    
    # Training
    verbose: int = 1
    tensorboard_log: Optional[str] = "./tensorboard_logs"
    
    def __post_init__(self):
        """Valida e configura parâmetros após inicialização."""
        if self.policy_kwargs is None:
            self.policy_kwargs = {
                "net_arch": {
                    "pi": [256, 256],  # Policy network
                    "vf": [256, 256]   # Value network
                },
                "activation_fn": nn.ReLU  # Função, não string!
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário compatível com MaskablePPO.
        
        Returns:
            Dicionário com todos os hiperparâmetros
        """
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "policy_kwargs": self.policy_kwargs,
            "verbose": self.verbose,
            "tensorboard_log": self.tensorboard_log
        }


# ============================================================================
# CONFIGURAÇÕES PRÉ-DEFINIDAS
# ============================================================================

# Configuração RÁPIDA - Para debugging e testes
HYPERPARAMS_FAST = PPOHyperparameters(
    learning_rate=3e-4,  # LR constante
    n_steps=256,         # Poucos steps
    batch_size=128,      # Batch pequeno
    n_epochs=4,          # Poucas epochs
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,       # Mais exploração
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},  # Rede menor
        "activation_fn": nn.ReLU
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

# Configuração PADRÃO - Balanceada para treinamento geral
HYPERPARAMS_DEFAULT = PPOHyperparameters(
    learning_rate=linear_schedule(3e-4),  # Linear decay
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.0,  # Sem bônus de entropia (PPO já explora bem)
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": nn.ReLU
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

# Configuração INTENSIVA - Para experimentos longos e completos
HYPERPARAMS_INTENSIVE = PPOHyperparameters(
    learning_rate=linear_schedule(3e-4),
    n_steps=4096,        # Horizonte mais longo
    batch_size=512,      # Batches maiores
    n_epochs=15,         # Mais epochs de otimização
    gamma=0.995,         # Fator de desconto maior
    gae_lambda=0.98,     # GAE mais alto
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={
        "net_arch": {"pi": [512, 512, 256], "vf": [512, 512, 256]},  # Rede maior
        "activation_fn": nn.ReLU
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

# Configuração HIGH EXPLORATION - Para ambientes com alta incerteza
HYPERPARAMS_HIGH_EXPLORATION = PPOHyperparameters(
    learning_rate=linear_schedule(5e-4),  # LR inicial maior
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,       # Entropia alta para exploração
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": nn.ReLU
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

# Configuração STABLE - Para convergência estável
HYPERPARAMS_STABLE = PPOHyperparameters(
    learning_rate=constant_schedule(1e-4),  # LR baixo e constante
    n_steps=2048,
    batch_size=128,      # Batches menores para updates mais frequentes
    n_epochs=8,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,      # Clipping mais conservador
    clip_range_vf=0.1,   # Também clipa value function
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.3,   # Gradient clipping mais agressivo
    policy_kwargs={
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": nn.ReLU
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

# Configuração LOW_PB - Otimizado para baixa probabilidade de bloqueio (<1%)
HYPERPARAMS_LOW_PB = PPOHyperparameters(
    learning_rate=linear_schedule(2e-4),  # LR moderado com decay
    n_steps=4096,        # Horizonte longo para melhor credit assignment
    batch_size=512,      # Batches grandes para estabilidade
    n_epochs=20,         # Muitas epochs para otimização profunda
    gamma=0.998,         # Gamma muito alto para valorizar futuro
    gae_lambda=0.98,     # GAE alto para melhor estimativa
    clip_range=0.15,     # Clipping moderado (não muito agressivo)
    clip_range_vf=None,
    ent_coef=0.005,      # Pouca entropia para exploração controlada
    vf_coef=0.8,         # Value function mais importante
    max_grad_norm=0.7,   # Gradient clipping menos agressivo
    policy_kwargs={
        "net_arch": {
            "pi": [512, 512, 384, 256],  # Rede profunda para policy
            "vf": [512, 512, 384, 256]   # Rede profunda para value
        },
        "activation_fn": nn.ReLU
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

HYPERPARAMS_LOW_PB_STABLE = PPOHyperparameters(
    learning_rate=constant_schedule(1e-4),  # LR baixo e constante para maior estabilidade
    n_steps=4096,
    batch_size=512,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.97,
    clip_range=0.12,
    clip_range_vf=0.12,
    ent_coef=0.003,
    vf_coef=0.7,
    max_grad_norm=0.3,
    policy_kwargs={
        "net_arch": {"pi": [512, 512, 256], "vf": [512, 512, 256]},
        "activation_fn": nn.Tanh,
        "ortho_init": False
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

HYPERPARAMS_LOW_PB_HYBRID = PPOHyperparameters(
    learning_rate=linear_schedule(2.5e-4),  # LR inicia alto e decai suavemente
    n_steps=4096,
    batch_size=512,
    n_epochs=15,
    gamma=0.997,
    gae_lambda=0.975,
    clip_range=0.15,
    clip_range_vf=0.15,
    ent_coef=0.006,
    vf_coef=0.7,
    max_grad_norm=0.4,
    policy_kwargs={
        "net_arch": {"pi": [512, 512, 256], "vf": [512, 512, 256]},
        "activation_fn": nn.ReLU,
        "ortho_init": True
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

# Configuração INTENSIVE V2 - Otimizado baseado em análise empírica
# Melhora o INTENSIVE original com ajustes para convergência mais rápida e estável
HYPERPARAMS_INTENSIVE_V2 = PPOHyperparameters(
    learning_rate=linear_schedule(4e-4),  # LR inicial maior para aprendizado rápido
    n_steps=4096,        # Horizonte longo mantido (bom para credit assignment)
    batch_size=1024,     # Batch MAIOR para updates mais estáveis com 44 envs
    n_epochs=12,         # Reduzido de 15 → 12 (menos overfitting por update)
    gamma=0.996,         # Entre 0.995 e 0.997 (melhor trade-off)
    gae_lambda=0.97,     # Reduzido de 0.98 (menos bias, mais estável)
    clip_range=0.2,      # Clipping padrão (0.2 é sweet spot comprovado)
    clip_range_vf=None,  # SEM clipping da VF (permite ajuste livre)
    ent_coef=0.01,       # Entropia moderada para exploração inicial
    vf_coef=0.6,         # Aumentado de 0.5 (value function mais importante)
    max_grad_norm=0.7,   # Aumentado para permitir gradientes maiores
    policy_kwargs={
        "net_arch": {
            "pi": [512, 512, 384],  # Policy: 3 camadas (512→512→384)
            "vf": [512, 512, 384]   # Value: mesma arquitetura
        },
        "activation_fn": nn.ReLU,
        "ortho_init": True  # Inicialização ortogonal para estabilidade
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

# Configuração INTENSIVE BALANCED - Próxima ao baseline original com refinamentos
HYPERPARAMS_INTENSIVE_BALANCED = PPOHyperparameters(
    learning_rate=exponential_schedule(3e-4, decay_rate=0.92),  # Decaimento mais rápido que o linear
    n_steps=4096,
    batch_size=768,      # Menos reuse de amostras por epoch
    n_epochs=8,          # Reduz overfitting mantendo updates suficientes
    gamma=0.996,
    gae_lambda=0.97,
    clip_range=0.2,
    clip_range_vf=0.2,   # Clipa value function para conter explosões do crítico
    ent_coef=0.008,      # Exploração moderada
    vf_coef=0.55,        # Menor peso para VF em relação ao baseline original
    max_grad_norm=0.6,
    policy_kwargs={
        "net_arch": {
            "pi": [512, 512, 256],
            "vf": [512, 512, 256]
        },
        "activation_fn": nn.ReLU,
        "ortho_init": True
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

# Configuração ULTRA - Para experimentos de longo prazo (>50k episódios)
# Máxima capacidade de representação e aprendizado profundo
HYPERPARAMS_ULTRA = PPOHyperparameters(
    learning_rate=linear_schedule(3.5e-4),  # LR moderado-alto com decay suave
    n_steps=8192,        # Horizonte muito longo para planning complexo
    batch_size=2048,     # Batches muito grandes (requer muita memória)
    n_epochs=10,         # Menos epochs mas com batches maiores
    gamma=0.997,         # Gamma alto para valorizar futuro distante
    gae_lambda=0.975,    # GAE balanceado
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.008,      # Entropia para manter exploração em treinos longos
    vf_coef=0.7,         # Value function importante
    max_grad_norm=0.8,   # Permite gradientes fortes
    policy_kwargs={
        "net_arch": {
            "pi": [768, 512, 384, 256],  # Policy: 4 camadas profundas
            "vf": [768, 512, 384, 256]   # Value: arquitetura profunda
        },
        "activation_fn": nn.ReLU,
        "ortho_init": True,
        "log_std_init": -0.5
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

# Configuração INTENSIVE_V2_REFINED - Baseado no melhor do primeiro treino + intensive_v2
# 🔥 RECOMENDADO: Combina sucesso do primeiro treino (PB 0.3%) com velocidade do intensive_v2
HYPERPARAMS_INTENSIVE_V2_REFINED = PPOHyperparameters(
    learning_rate=linear_schedule(3e-4),  # LR moderado (mesmo do primeiro treino)
    n_steps=4096,        # Horizonte longo mantido
    batch_size=1024,     # Batch grande do intensive_v2 (para 44 envs)
    n_epochs=10,         # Reduzido de 12 → 10 (melhor estabilidade)
    gamma=0.995,         # Entre primeiro treino e intensive_v2
    gae_lambda=0.97,     # GAE estável
    clip_range=0.2,      # Clipping padrão comprovado
    clip_range_vf=0.2,   # Clipa VF para evitar explosões
    ent_coef=0.005,      # Entropia baixa (já exploramos o suficiente)
    vf_coef=0.65,        # Value function importante (entre 0.5 e 0.7)
    max_grad_norm=0.5,   # Gradient clipping moderado
    policy_kwargs={
        "net_arch": {
            "pi": [512, 512, 256],  # Policy: 3 camadas (comprovada)
            "vf": [512, 512, 256]   # Value: mesma arquitetura
        },
        "activation_fn": nn.ReLU,
        "ortho_init": True  # Inicialização ortogonal
    },
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)


def get_hyperparams(profile: str = "default") -> PPOHyperparameters:
    """
    Obtém hiperparâmetros por nome de perfil.
    
    Args:
        profile: Nome do perfil ('fast', 'default', 'intensive', 
                                  'high_exploration', 'stable')
                                  
    Returns:
        Objeto PPOHyperparameters configurado
        
    Raises:
        ValueError: Se o perfil não existir
        
    Example:
        >>> hyperparams = get_hyperparams("default")
        >>> model = MaskablePPO("MultiInputPolicy", env, **hyperparams.to_dict())
    """
    profiles = {
        "fast": HYPERPARAMS_FAST,
        "default": HYPERPARAMS_DEFAULT,
        "intensive": HYPERPARAMS_INTENSIVE,
        "intensive_v2": HYPERPARAMS_INTENSIVE_V2,
        "intensive_v2_refined": HYPERPARAMS_INTENSIVE_V2_REFINED,
        "intensive_balanced": HYPERPARAMS_INTENSIVE_BALANCED,
        "ultra": HYPERPARAMS_ULTRA,
        "high_exploration": HYPERPARAMS_HIGH_EXPLORATION,
        "stable": HYPERPARAMS_STABLE,
        "low_pb": HYPERPARAMS_LOW_PB,
        "low_pb_stable": HYPERPARAMS_LOW_PB_STABLE,
        "low_pb_hybrid": HYPERPARAMS_LOW_PB_HYBRID
    }
    
    profile_lower = profile.lower()
    if profile_lower not in profiles:
        raise ValueError(
            f"Unknown profile '{profile}'. "
            f"Available profiles: {list(profiles.keys())}"
        )
    
    return profiles[profile_lower]


def print_hyperparams(hyperparams: PPOHyperparameters, name: str = ""):
    """
    Imprime hiperparâmetros de forma formatada.
    
    Args:
        hyperparams: Objeto PPOHyperparameters
        name: Nome do perfil (opcional)
    """
    print("\n" + "="*70)
    if name:
        print(f" HIPERPARÂMETROS PPO - {name.upper()}")
    else:
        print(" HIPERPARÂMETROS PPO")
    print("="*70)
    
    lr = hyperparams.learning_rate
    lr_str = f"{lr}" if callable(lr) else f"{lr:.2e}"
    
    print(f"\n📚 Learning:")
    print(f"   Learning rate: {lr_str}")
    print(f"   N steps: {hyperparams.n_steps}")
    print(f"   Batch size: {hyperparams.batch_size}")
    print(f"   N epochs: {hyperparams.n_epochs}")
    
    print(f"\n🎯 Discount:")
    print(f"   Gamma: {hyperparams.gamma}")
    print(f"   GAE lambda: {hyperparams.gae_lambda}")
    
    print(f"\n✂️ Clipping:")
    print(f"   Clip range: {hyperparams.clip_range}")
    if hyperparams.clip_range_vf:
        print(f"   Clip range VF: {hyperparams.clip_range_vf}")
    
    print(f"\n🔧 Regularization:")
    print(f"   Entropy coef: {hyperparams.ent_coef}")
    print(f"   Value coef: {hyperparams.vf_coef}")
    print(f"   Max grad norm: {hyperparams.max_grad_norm}")
    
    if hyperparams.policy_kwargs:
        print(f"\n🧠 Network Architecture:")
        if "net_arch" in hyperparams.policy_kwargs:
            arch = hyperparams.policy_kwargs["net_arch"]
            if isinstance(arch, dict):
                print(f"   Policy: {arch.get('pi', 'N/A')}")
                print(f"   Value: {arch.get('vf', 'N/A')}")
            else:
                print(f"   Shared: {arch}")
        if "activation_fn" in hyperparams.policy_kwargs:
            print(f"   Activation: {hyperparams.policy_kwargs['activation_fn']}")
    
    print("\n" + "="*70)


def get_available_profiles() -> Dict[str, str]:
    """
    Lista todos os perfis disponíveis com descrições.
    
    Returns:
        Dicionário {nome_perfil: descrição}
    """
    return {
        "fast": "Configuração rápida para debugging e testes",
        "default": "Configuração padrão balanceada",
        "intensive": "Configuração intensiva para experimentos longos",
        "intensive_v2": "INTENSIVE otimizado - LR 4e-4, batch 1024, epochs 12",
        "intensive_v2_refined": "🔥 RECOMENDADO - Combina primeiro treino (PB 0.3%) com ajustes de estabilidade",
        "intensive_balanced": "Baseline intensivo com decay mais rápido e VF clipado",
        "ultra": "⚡ Configuração máxima - n_steps 8192, batch 2048, rede 768→512→384→256",
        "high_exploration": "Alta exploração para ambientes incertos",
        "stable": "Convergência estável e conservadora",
        "low_pb": "Otimizado para baixa probabilidade de bloqueio (<1%)",
        "low_pb_stable": "⚠️ EVITAR - Baixa PB com clipping agressivo (performance ruim)",
        "low_pb_hybrid": "Combina agressividade inicial com refinamento estável"
    }


if __name__ == "__main__":
    """
    Demonstração dos perfis de hiperparâmetros.
    """
    print("\n" + "="*70)
    print(" PERFIS DE HIPERPARÂMETROS DISPONÍVEIS")
    print("="*70)
    
    profiles = get_available_profiles()
    for name, description in profiles.items():
        print(f"\n📋 {name.upper()}")
        print(f"   {description}")
    
    print("\n" + "="*70)
    
    # Mostra detalhes do perfil DEFAULT
    print("\n🔍 Detalhes do perfil DEFAULT:")
    hyperparams = get_hyperparams("default")
    print_hyperparams(hyperparams, "default")
    
    # Demonstra uso do linear schedule
    print("\n" + "="*70)
    print(" DEMONSTRAÇÃO: LINEAR LEARNING RATE SCHEDULE")
    print("="*70)
    
    schedule = linear_schedule(3e-4)
    print("\nProgress → Learning Rate:")
    for progress in [1.0, 0.75, 0.5, 0.25, 0.0]:
        lr = schedule(progress)
        print(f"  {progress:4.2f} → {lr:.2e}")
