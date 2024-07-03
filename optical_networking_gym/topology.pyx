# cython: linetrace=True
import cython
import numpy as np


@cython.cclass
class Span:

    length = cython.declare(cython.double, visibility='readonly')
    attenuation_db_km = cython.declare(cython.double, visibility='readonly')
    attenuation_normalized = cython.declare(cython.double, visibility='readonly')
    noise_figure_db = cython.declare(cython.double, visibility='readonly')
    noise_figure_normalized = cython.declare(cython.double, visibility='readonly')

    def __cinit__(self, length: float, attenuation: float, noise_figure: float):
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


@cython.cclass
class Link:
    id: cython.declare(cython.int, visibility='readonly')
    node1: cython.declare(cython.str, visibility='readonly')
    node2: cython.declare(cython.str, visibility='readonly')
    length: cython.declare(cython.double, visibility='readonly')
    spans: tuple[Span] = cython.declare(cython.tuple, visibility='readonly')

    def __cinit__(self, id: cython.int, node1: cython.str, node2: cython.str, length: cython.double, spans: cython.tuple):
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.length = length
        self.spans = spans
