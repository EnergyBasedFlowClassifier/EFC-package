import pytest
import warnings

from sklearn.datasets import load_iris
from sklearn.preprocessing import MaxAbsScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from efc import EnergyBasedFlowClassifier


@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    return X, y


def test_multiclass(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y
    )

    clf = EnergyBasedFlowClassifier()
    clf.fit(X_train, y_train)

    # checks if unknown class is being used when unknown_class is set to True
    y_pred_unknown = clf.predict(X_test, unknown_class=True)
    assert -1 in y_pred_unknown

    # checks if return_energies is returnig an array with correct size
    y_pred, y_energies = clf.predict(X_test, return_energies=True)
    assert y_pred.shape == y_energies.shape
