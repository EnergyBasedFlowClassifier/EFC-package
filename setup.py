#! /usr/bin/env python

import codecs
import os

from setuptools import find_packages, setup
from Cython.Build import cythonize

# get __version__ from _version.py
ver_file = os.path.join("efc", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "efc"
DESCRIPTION = "A scikit-learn compatible package of the Energy-based Flow Classifier."
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Manuela M. C. de Souza"
MAINTAINER_EMAIL = "munak98@hotmail.com"
LICENSE = "new BSD"
VERSION = __version__
INSTALL_REQUIRES = ["numpy", "cython", "scikit-learn"]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cython", "pytest-cov"],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    version=VERSION,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    ext_modules=cythonize("efc/_base_fast.pyx"),
)
