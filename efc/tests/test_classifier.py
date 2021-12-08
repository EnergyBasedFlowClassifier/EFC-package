import pytest
import numpy as np

from sklearn.datasets import load_iris
from efc import EnergyBasedFlowClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_classifier(data):
    X, y = data
    clf = EnergyBasedFlowClassifier()
    assert clf.pseudocounts == 0.5
    assert clf.cutoff_quantile == 0.95
    assert clf.n_jobs == None

    clf.fit(X, y)
    assert hasattr(clf, "max_bin_")
    assert hasattr(clf, "n_features_in_")
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "target_type_")
    assert hasattr(clf, "estimators_")

    for sub_clf in clf.estimators_:
        assert hasattr(sub_clf, "X_")
        assert hasattr(sub_clf, "sitefreq_")
        assert hasattr(sub_clf, "pairfreq_")
        assert hasattr(sub_clf, "coupling_matrix_")
        assert hasattr(sub_clf, "local_fields_")
        assert hasattr(sub_clf, "cutoff_")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
