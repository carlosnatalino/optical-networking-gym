import numpy as np

cdef class Span:

    cdef readonly double length
    cdef readonly double attenuation_db_km
    cdef readonly double attenuation_normalized
    cdef readonly double noise_figure_db
    cdef readonly double noise_figure_normalized
    cdef public double x

    def __init__(self, length: float, attenuation: float, noise_figure: float):
        self.length = length

        self.attenuation_db_km = attenuation
        self.attenuation_normalized = self.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)  # dB/km ===> 1/m

        self.noise_figure_db = noise_figure
        self.noise_figure_normalized = 10 ** (self.noise_figure_db / 10)  # dB ===> norm
    
    def set_attenuation(self, attenuation: float) -> None:
        self.attenuation_db_km = attenuation
        self.attenuation_normalized = self.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)  # dB/km ===> 1/m
    
    def set_noise_figure(self, noise_figure: float) -> None:
        self.noise_figure_db = noise_figure
        self.noise_figure_normalized = 10 ** (self.noise_figure_db / 10)  # dB ===> norm
    
    def __repr__(self) -> str:
        return f"Span(length={self.length:.2f}, attenuation_db_km={self.attenuation_db_km}, noise_figure_db={self.noise_figure_db})"
