"""
=======================
Predicting iris dataset
=======================

An example of using the EFC to predict multi-class targets.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from efc import EnergyBasedFlowClassifier

# loading the iris toy dataset from scikit-learn
X, y = load_iris(return_X_y=True)

# spliting train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, shuffle=True, test_size=0.3
)

# train and test EFC
clf = EnergyBasedFlowClassifier(cutoff_quantile=0.8)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print(classification_report(y_test, y_pred))
