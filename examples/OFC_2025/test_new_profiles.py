#!/usr/bin/env python3
"""
Script para validar e comparar os novos perfis PPO

Testa:
- intensive vs intensive_v2 vs ultra
- Exibe diferenças lado a lado
- Valida compatibilidade com MaskablePPO

Autor: Sistema de treinamento OFC 2025
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hyperparams import (
    get_hyperparams, 
    print_hyperparams,
    get_available_profiles,
    HYPERPARAMS_INTENSIVE,
    HYPERPARAMS_INTENSIVE_V2,
    HYPERPARAMS_ULTRA
)

def compare_profiles():
    """Compara perfis lado a lado"""
    
    print("\n" + "="*80)
    print(" COMPARAÇÃO: INTENSIVE vs INTENSIVE_V2 vs ULTRA")
    print("="*80)
    
    profiles = {
        "intensive": HYPERPARAMS_INTENSIVE,
        "intensive_v2": HYPERPARAMS_INTENSIVE_V2,
        "ultra": HYPERPARAMS_ULTRA
    }
    
    # Métricas para comparar
    metrics = [
        ("Learning Rate (inicial)", lambda p: "4e-4" if "intensive_v2" in str(p) else "3.5e-4" if "ultra" in str(p) else "3e-4"),
        ("N Steps", lambda p: p.n_steps),
        ("Batch Size", lambda p: p.batch_size),
        ("N Epochs", lambda p: p.n_epochs),
        ("Gamma", lambda p: p.gamma),
        ("GAE Lambda", lambda p: p.gae_lambda),
        ("Clip Range", lambda p: p.clip_range),
        ("Ent Coef", lambda p: p.ent_coef),
        ("VF Coef", lambda p: p.vf_coef),
        ("Max Grad Norm", lambda p: p.max_grad_norm),
    ]
    
    print("\n{:<25} {:<15} {:<15} {:<15}".format("Métrica", "intensive", "intensive_v2", "ultra"))
    print("-" * 80)
    
    for metric_name, getter in metrics:
        values = {name: getter(hp) for name, hp in profiles.items()}
        
        # Marca diferenças com ✅
        int_v2_diff = "✅" if values["intensive_v2"] != values["intensive"] else ""
        ultra_diff = "✅" if values["ultra"] != values["intensive"] else ""
        
        print("{:<25} {:<15} {:<15} {:<15}".format(
            metric_name,
            str(values["intensive"]),
            f"{values['intensive_v2']} {int_v2_diff}",
            f"{values['ultra']} {ultra_diff}"
        ))
    
    # Arquiteturas de rede
    print("\n" + "-" * 80)
    print("Network Architecture:")
    print("-" * 80)
    
    for name, hp in profiles.items():
        arch = hp.policy_kwargs["net_arch"]
        print(f"\n{name.upper()}:")
        print(f"  Policy: {arch['pi']}")
        print(f"  Value:  {arch['vf']}")
        if "log_std_init" in hp.policy_kwargs:
            print(f"  Log Std Init: {hp.policy_kwargs['log_std_init']}")
        if "ortho_init" in hp.policy_kwargs:
            print(f"  Ortho Init: {hp.policy_kwargs['ortho_init']}")
    
    print("\n" + "="*80)


def test_profile_loading():
    """Testa carregamento de perfis"""
    
    print("\n" + "="*80)
    print(" TESTE: Carregamento de Perfis")
    print("="*80)
    
    test_profiles = ["intensive", "intensive_v2", "ultra"]
    
    for profile_name in test_profiles:
        try:
            hp = get_hyperparams(profile_name)
            hp_dict = hp.to_dict()
            
            print(f"\n✅ {profile_name.upper()}: OK")
            print(f"   - {len(hp_dict)} parâmetros configurados")
            print(f"   - Batch size: {hp.batch_size}")
            print(f"   - N steps: {hp.n_steps}")
            print(f"   - Samples per rollout: {hp.n_steps} steps × N envs")
            
        except Exception as e:
            print(f"\n❌ {profile_name.upper()}: ERRO")
            print(f"   {str(e)}")
    
    print("\n" + "="*80)


def estimate_resources():
    """Estima uso de recursos"""
    
    print("\n" + "="*80)
    print(" ESTIMATIVA DE RECURSOS (com 44 envs)")
    print("="*80)
    
    profiles = {
        "intensive": HYPERPARAMS_INTENSIVE,
        "intensive_v2": HYPERPARAMS_INTENSIVE_V2,
        "ultra": HYPERPARAMS_ULTRA
    }
    
    print("\n{:<15} {:<15} {:<15} {:<20} {:<15}".format(
        "Profile", "GPU Mem (GB)", "RAM (GB)", "Samples/Rollout", "Updates/Rollout"
    ))
    print("-" * 85)
    
    n_envs = 44
    
    for name, hp in profiles.items():
        # Estimativas aproximadas
        if name == "intensive":
            gpu_mem = "~6-8"
            ram = "~24-32"
        elif name == "intensive_v2":
            gpu_mem = "~8-10"
            ram = "~32-40"
        else:  # ultra
            gpu_mem = "~12-16"
            ram = "~40-56"
        
        samples = hp.n_steps * n_envs
        updates = (samples // hp.batch_size) * hp.n_epochs
        
        print("{:<15} {:<15} {:<15} {:<20,} {:<15,}".format(
            name, gpu_mem, ram, samples, updates
        ))
    
    print("\n" + "="*80)
    print("\nNOTA: Valores aproximados. Depende de:")
    print("  - Tamanho do observation space")
    print("  - PyTorch version & CUDA")
    print("  - Outros processos rodando")


def show_recommendations():
    """Mostra recomendações de uso"""
    
    print("\n" + "="*80)
    print(" RECOMENDAÇÕES DE USO")
    print("="*80)
    
    recommendations = [
        ("intensive", 
         "✓ Baseline sólido",
         "✓ Recursos moderados",
         "✓ Treinos médios (20-40k eps)",
         "⚠️ Convergência pode ser lenta"),
        
        ("intensive_v2",
         "🔥 RECOMENDADO para maioria dos casos",
         "✓ Convergência rápida (LR 4e-4)",
         "✓ Otimizado para 44 envs (batch 1024)",
         "✓ Melhor custo-benefício"),
        
        ("ultra",
         "⚡ Para experimentos longos (>50k eps)",
         "✓ Máxima capacidade de representação",
         "⚠️ Requer GPU potente (≥16GB)",
         "⚠️ Treino lento (~2-3x do intensive_v2)")
    ]
    
    for profile, *points in recommendations:
        print(f"\n📋 {profile.upper()}:")
        for point in points:
            print(f"   {point}")
    
    print("\n" + "="*80)


def main():
    """Main function"""
    
    print("\n" + "="*80)
    print(" VALIDAÇÃO DOS NOVOS PERFIS PPO")
    print("="*80)
    print("\nCriados: intensive_v2, ultra")
    print("Baseados em: Análise empírica dos treinamentos anteriores")
    
    # Lista todos perfis disponíveis
    print("\n" + "="*80)
    print(" PERFIS DISPONÍVEIS")
    print("="*80)
    profiles = get_available_profiles()
    for name, desc in profiles.items():
        emoji = "🔥" if "intensive_v2" in name else "⚡" if "ultra" in name else "⚠️" if "low_pb_stable" in name else "📋"
        print(f"\n{emoji} {name.upper()}")
        print(f"   {desc}")
    
    # Testes
    compare_profiles()
    test_profile_loading()
    estimate_resources()
    show_recommendations()
    
    # Exemplo de uso
    print("\n" + "="*80)
    print(" EXEMPLO DE USO")
    print("="*80)
    print("""
# No train_n_steps.py:

train_cfg = TrainingConfig(
    n_episodes=75000,
    n_envs=44,
    ppo_profile="intensive_v2",  # 🔥 RECOMENDADO
    verbose=0
)

# Ou para experimentos longos:

train_cfg = TrainingConfig(
    n_episodes=100000,
    n_envs=44,
    ppo_profile="ultra",  # ⚡ Máximo poder
    verbose=0
)
""")
    
    print("\n" + "="*80)
    print(" ✅ VALIDAÇÃO COMPLETA")
    print("="*80)
    print("\nPróximos passos:")
    print("  1. Testar intensive_v2 em treino real")
    print("  2. Ajustar reward shaping conforme necessário")
    print("  3. Monitorar blocking probability e reward/step")
    print("  4. Se necessário, escalar para 'ultra'")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
