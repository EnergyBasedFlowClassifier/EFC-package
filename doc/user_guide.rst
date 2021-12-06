.. title:: User guide : contents

.. _user_guide:

==================================================
User guide: using the Energy-based Flow Classifier
==================================================

The :class:`EnergyBasedFlowClassifier` is a scikit-learn compatible estimator. 
It is the central class of EFC and can be directly instanciated by end users.

To use EFC, first import it as::

    >>> from efc import EnergyBasedFlowClassifier

Once imported, you can create a instance of EFC and use the fit and predict methods like any other scikit-learn estimator::

    >>> clf = EnergyBasedFlowClassifier()
    ... clf.fit(X_train, y_train)
    ... clf.predict(X_test)

EFC supports both binary and multi-class targets. For binary targets, note that it is a single-class algorithm. In this case, the base_class parameters should be use to set the trainig class.
