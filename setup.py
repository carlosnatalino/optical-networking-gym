from setuptools import setup
import numpy as np
from Cython.Build import cythonize

debug_cythonize_kw = dict(force=True)

debug_cythonize_kw.update(dict(gdb_debug=True,
                              force=True,
                              annotate=True,
                              compiler_directives={'linetrace': True, 'binding': True}))

setup(
    name="optical_networking_gym",
    ext_modules=cythonize(
        [
            # "optical_networking_gym/envs/*.pyx",
            "optical_networking_gym/**/*.pyx",
        ],
        include_path=[np.get_include()],
        language_level="3",
        **debug_cythonize_kw
    ),
)
