============================
Energy-based Flow Classifier
============================

The Energy-Based Flow Classifier (EFC) is a new classification method developed in the context of network intrusion detection systems. It was first presented in
`A New Method for Flow-Based Network Intrusion Detection Using the Inverse Potts Model <https://ieeexplore.ieee.org/document/9415676>`_.

Dependencies
------------

EFC package requires:

- Python (>= 3.8)
- Cython (>= 0.29)
- NumPy (>= 1.21.4)
- Scikit-learn (>= 1.0.1)
- joblib (>= 1.1.0)
- threadpoolctl (>= .0.0)

Installation
------------

Currently, the only way to install EFC is from source, using its GitHub repository. To do so, clone this repository and run the following commands::

    git clone https://github.com/EnergyBasedFlowClassifier/EFC-package
    cd EFC-package
    pip install -r requirements.txt
    pip install .


Usage
-----
Use EFC like a scikit-learn estimator::

    from efc import EnergyBasedFlowClassifier

    clf = EnergyBasedFlowClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

It supports both binary and multiclass classification.
When the target is binary, EFC is a single-class algorithm. Therefore, the user must choose which class will be used as the base class.
When the target is multiclass, the user must choose whether to use the unknown label or not. 


The EFC internally normalizes and discretizes the input data. However, like any scikit-learn estimator, it requires categorical features to be encoded before input. It is also necessary that categorical columns are specified when calling the fit method, so that they are ignored during attribute preprocessing.
For a full explanation of each of EFC's parameters, read the `API documentation <https://efc-package.readthedocs.io/en/latest/generated/efc.EnergyBasedFlowClassifier.html#efc.EnergyBasedFlowClassifier>`_ in Read the Docs.

Citations
---------

If you use EFC in a scientific publication, please cite the original paper::

    @article {9415676,
    author={Pontes, Camila F. T. and de Souza, Manuela M. C. and Gondim, João J. C. and Bishop, Matt and Marotta, Marcelo Antonio},
    journal={IEEE Transactions on Network and Service Management},
    title={A New Method for Flow-Based Network Intrusion Detection Using the Inverse Potts Model},
    year={2021},
    volume={18},
    number={2},
    pages={1125-1136},
    doi={10.1109/TNSM.2021.3075503}}
    
Related Works
-------------
    [1] "C. F. T. Pontes, M. M. C. de Souza, J. J. C. Gondim, M. Bishop and M. A. Marotta, *A New Method for Flow-Based Network Intrusion Detection Using the Inverse Potts Model*, in *IEEE Transactions on Network and Service Management*, vol. 18, no. 2, pp. 1125-1136, June 2021, doi: 10.1109/TNSM.2021.3075503."

    [2] "J. M. De Almeida et al., *Abnormal Behavior Detection Based on Traffic Pattern Categorization in Mobile Cellular Networks*, in *IEEE Transactions on Network and Service Management*, doi: 10.1109/TNSM.2021.3125019."
