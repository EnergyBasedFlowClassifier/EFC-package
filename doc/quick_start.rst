#############################
Energy-based Flow Classifier
#############################

This package implements the Energy-based Flow Classifier, a new machine learning algorithm developed in the context of network applications.
To use EFC, first you need to install it.

Dependencies
============

EFC package requires:

- Python (>= 3.8)
- Cython (>= 0.29)
- NumPy (>= 1.21.4)
- Scikit-learn (>= 1.0.1)
- joblib (>= 1.1.0)
- threadpoolctl (>= 3.0.0)
  
Installation from source
========================

Currently, EFC can only be installed from source via its github repository::
    
    >>> git clone https://github.com/EnergyBasedFlowClassifier/EFC-package
    ... cd EFC-package
    ... pip install -r requirements.txt
    ... pip install .
