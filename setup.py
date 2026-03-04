from setuptools import setup, Extension
import numpy

ext_modules = [
    Extension(
        "schainpy.model._noise",
        sources=["schainc/_noise.c"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "schainpy.model._HS_algorithm",
        sources=["schainc/_HS_algorithm.c"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    ext_modules=ext_modules,
)