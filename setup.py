from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["./src/*.pyx","./examples/sv_model_nsmc.pyx","./examples/lgss_model_nsmc.pyx"])
)

