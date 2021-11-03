from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules= cythonize("efc/_energyclassifier_fast.pyx", annotate=True)
    )