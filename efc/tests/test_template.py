import pytest
import numpy as np

from sklearn.datasets import load_iris
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from efc import EnergyBasedFlowClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_template_estimator(data):
    est = EnergyBasedFlowClassifier()
    assert est.demo_param == 'demo_param'

    est.fit(*data)
    assert hasattr(est, 'is_fitted_')

    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


def test_template_classifier(data):
    X, y = data
    clf = EnergyBasedFlowClassifier()
    assert clf.demo_param == 'demo'

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
