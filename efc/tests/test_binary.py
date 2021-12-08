import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from efc import EnergyBasedFlowClassifier


@pytest.fixture
def data():
    X, y = load_breast_cancer(return_X_y=True)
    return X, y


def test_binary(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y
    )

    # creating a model with class 0 as base class
    clf = EnergyBasedFlowClassifier()
    clf.fit(X_train, y_train, base_class=0)
    y_pred, y_energies = clf.predict(X_test, return_energies=True)

    # checks if predictions are binary
    assert np.unique(y_pred).shape[0] <= 2
    # checks if energies are returned
    assert y_pred.shape == y_energies.shape

    # creating a model with class 0 as base class
    clf_0 = EnergyBasedFlowClassifier()
    clf_0.fit(X_train, y_train, base_class=0)
    y_pred_0 = clf_0.predict(X_test)

    # creating a model with class 1 as base class
    clf_1 = EnergyBasedFlowClassifier()
    clf_1.fit(X_train, y_train, base_class=1)
    y_pred_1 = clf_1.predict(X_test)

    # checks if changing the base class changes the models and their predictions
    assert clf_0.estimators_[0] != clf_1.estimators_[0]
    assert not np.array_equal(y_pred_0, y_pred_1)
