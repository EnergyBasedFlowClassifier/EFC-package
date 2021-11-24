===============
Energy-based Flow Classifier
===============

Installation
------------
Clone this repository and run the command ``pip install .`` inside the root folder.


Usage
------------
Use EFC like a scikit-learn estimator:
::
  from efc import EnergyBasedFlowClassifier
  
  clf = EnergyBasedFlowClassifier()
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
