from setuptools import setup
# import numpy as np
from Cython.Build import cythonize

setup(
    name="optical_networking_gym",
    install_requires=["gymnasium", "numpy", "matplotlib", "networkx"],
    ext_modules=cythonize(
        [
            # "optical_networking_gym/envs/*.pyx",
            "optical_networking_gym/**/*.pyx",
        ],
        # include_path=[np.get_include()],
        language_level="3",
    ),
)
