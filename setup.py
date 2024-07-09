from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

setup(
    name="optical_networking_gym",
    # install_requires=["gymnasium", "numpy", "matplotlib", "networkx"],
    ext_modules=cythonize([
        Extension(
            "optical_networking_gym.utils",
            ["optical_networking_gym/utils.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "optical_networking_gym.topology",
            ["optical_networking_gym/topology.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "optical_networking_gym.envs.qrmsa",
            ["optical_networking_gym/envs/qrmsa.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ]),
    include_dirs=[np.get_include()],
)
