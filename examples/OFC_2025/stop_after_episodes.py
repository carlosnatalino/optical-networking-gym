#!/usr/bin/env python3
"""
Callback para parar treinamento após N episódios

Este callback monitora o número de episódios completados e para
o treinamento quando atingir o limite desejado.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StopAfterEpisodesCallback(BaseCallback):
    """
    Para o treinamento após um número específico de episódios.
    
    Este callback é útil quando você quer treinar por um número
    fixo de episódios ao invés de timesteps, o que é mais intuitivo
    para análise de convergência.
    
    Args:
        n_episodes: Número de episódios após o qual parar
        verbose: Nível de verbosidade (0: quiet, 1: info)
    """
    
    def __init__(self, n_episodes: int, verbose: int = 1):
        super().__init__(verbose)
        self.n_episodes = n_episodes
        self.episode_count = 0
        self.initial_episode_count = 0
        
    def _on_training_start(self) -> None:
        """Inicializa contagem de episódios."""
        # Pega contagem inicial se o modelo já foi treinado antes
        if hasattr(self.model, 'ep_info_buffer'):
            self.initial_episode_count = len(self.model.ep_info_buffer)
        
        if self.verbose >= 1:
            print(f"\n[StopAfterEpisodes] Treinamento parara apos {self.n_episodes} episodios")
            print(f"   Episodios iniciais: {self.initial_episode_count}")
    
    def _on_step(self) -> bool:
        """
        Verifica se atingiu o número de episódios.
        
        Returns:
            False para parar o treinamento, True para continuar
        """
        # Conta episódios completados
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        for done, info in zip(dones, infos):
            if not done or not isinstance(info, dict):
                continue

            self.episode_count += 1

            if self.verbose >= 1 and self.episode_count % 10 == 0:
                progress = (self.episode_count / self.n_episodes) * 100
                print(f"[Episodios] Completados: {self.episode_count}/{self.n_episodes} ({progress:.1f}%)")
        
        # Verifica se atingiu o limite
        if self.episode_count >= self.n_episodes:
            if self.verbose >= 1:
                print(f"\n[OK] Meta atingida! {self.episode_count} episodios completados.")
                print(f"   Timesteps totais: {self.num_timesteps:,}")
            return False  # Para o treinamento
        
        return True  # Continua treinando


if __name__ == "__main__":
    print("""
StopAfterEpisodesCallback - Para treinamento após N episódios

Uso:
    from stop_after_episodes import StopAfterEpisodesCallback
    
    callback = StopAfterEpisodesCallback(n_episodes=100, verbose=1)
    model.learn(total_timesteps=1000000, callback=callback)
    
    # O treinamento parará automaticamente após 100 episódios,
    # mesmo que não tenha completado 1M timesteps
    """)
