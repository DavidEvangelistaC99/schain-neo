from setuptools import setup, Extension
import numpy

ext_modules = [
    Extension(
        "schainpy.model.data._noise",
        sources=["schainc/_noise.c"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "schainpy.model.data._HS_algorithm",
        sources=["schainc/_HS_algorithm.c"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    ext_modules=ext_modules,
)
