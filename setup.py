import os
from setuptools import setup, Extension
import platform

import numpy as np
from Cython.Build import cythonize


def get_env_or_default(key, default):
    return os.environ.get(key, default)


DEBUG = get_env_or_default("DEBUG", "0") == "1"

compiler_directives = {"language_level": "3"}
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
extra_link_args = []

extra_compile_args = []
if platform.system() == "Windows":
    extra_compile_args = ["/O2", "/Wall"]
else:
    extra_compile_args = ["-O3", "-march=native", "-ffast-math"]


if DEBUG:
    define_macros.append(("CYTHON_TRACE_NOGIL", "1"))
else:
    extra_compile_args = ["-O3", "-march=native", "-ffast-math"]
    extra_link_args = ["-O3"]
    compiler_directives["boundscheck"] = False
    compiler_directives["wraparound"] = False
    compiler_directives["nonecheck"] = False
    compiler_directives["cdivision"] = True

setup(
    name="optical_networking_gym",
    # install_requires=["gymnasium", "numpy", "matplotlib", "networkx"],
    ext_modules=cythonize(
        [
            Extension(
                "optical_networking_gym.utils",
                ["optical_networking_gym/utils.pyx"],
                include_dirs=[np.get_include()],
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            ),
            Extension(
                "optical_networking_gym.core.osnr",
                ["optical_networking_gym/core/osnr.pyx"],
                include_dirs=[np.get_include()],
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            ),
            Extension(
                "optical_networking_gym.topology",
                ["optical_networking_gym/topology.pyx"],
                include_dirs=[np.get_include()],
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            ),
            Extension(
                "optical_networking_gym.envs.rmsa",
                ["optical_networking_gym/envs/rmsa.pyx"],
                include_dirs=[np.get_include()],
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            ),
            Extension(
                "optical_networking_gym.envs.qrmsa",
                ["optical_networking_gym/envs/qrmsa.pyx"],
                include_dirs=[np.get_include()],
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            ),
        ],
        compiler_directives=compiler_directives,
    ),
    include_dirs=[np.get_include()],
)
