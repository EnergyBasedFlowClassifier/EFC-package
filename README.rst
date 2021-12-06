============================
Energy-based Flow Classifier
============================

The Energy-Based Flow Classifier (EFC) is a new classification method developed in the context of network intrusion detection systems. It was first presented in
`A New Method for Flow-Based Network Intrusion Detection Using the Inverse Potts Model <https://ieeexplore.ieee.org/document/9415676>`_ and latter used in the work `Abnormal Behavior Detection Based on Traffic Pattern Categorization in Mobile Cellular Networks <https://ieeexplore.ieee.org/document/9600445>`_.



Installation
------------

Currently, the only way to install EFC is from source, using its GitHub repository. To do so, clone this repository and run the command ``pip install .`` inside the root folder::

    git clone https://github.com/EnergyBasedFlowClassifier/EFC-package
    cd EFC-package
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

