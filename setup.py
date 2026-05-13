from __future__ import annotations

from Cython.Build import cythonize
import numpy
from setuptools import Extension, setup


extensions = [
    Extension(
        "optical_networking_gym_v2.optical.kernels.allocation_kernel",
        ["src/optical_networking_gym_v2/optical/kernels/allocation_kernel.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "optical_networking_gym_v2.optical.kernels.qot_kernel",
        ["src/optical_networking_gym_v2/optical/kernels/qot_kernel.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "optical_networking_gym_v2.simulation._request_analysis_kernels",
        ["src/optical_networking_gym_v2/simulation/_request_analysis_kernels.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
    )
)
