from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

setup(
    name="optical_networking_gym",
    # install_requires=["gymnasium", "numpy", "matplotlib", "networkx"],
    ext_modules=[
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
    # cythonize(
    #     [
    #         "optical_networking_gym/*.pyx",
    #     ],
    #     # include_path=[np.get_include()],
    #     language_level="3",
    #     # compiler_directives={"profile": True, "linetrace": True},
    # ),
    ],
    include_dirs=[np.get_include()],
)
