import pytest
import numpy as np

from sklearn.datasets import load_iris
from numpy.testing import assert_array_equal

from sklearn.preprocessing import MaxAbsScaler, KBinsDiscretizer

from efc import coupling
from efc import local_fields
from efc import site_freq
from efc import pair_freq
from efc import compute_energy
from _base_pure import BaseEFC
import warnings


@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    return X, y


def test_extension(data):
    X, y = data
    X = X[np.where(y == 0)[0]]
    y = y[np.where(y == 0)[0]]

    # pure python version
    clf = BaseEFC()

    # normalize features
    norm = MaxAbsScaler()
    X = norm.fit_transform(X)

    # discretize features
    disc = KBinsDiscretizer(n_bins=30, encode="ordinal", strategy="quantile")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        X = disc.fit_transform(X).astype("int")

    clf.fit(X)

    # extension methods
    sitefreq = site_freq(X, clf.pseudocounts, clf.max_bin)
    pairfreq = pair_freq(X, sitefreq, clf.pseudocounts, clf.max_bin)
    coupling_matrix = coupling(pairfreq, clf.sitefreq_, clf.pseudocounts, clf.max_bin)
    fields = local_fields(
        coupling_matrix, pairfreq, clf.sitefreq_, clf.pseudocounts, clf.max_bin
    )
    coupling_matrix = np.log(coupling_matrix)
    fields = np.log(fields)

    # assert python version with extension
    assert_array_equal(clf.sitefreq_, sitefreq)
    assert_array_equal(clf.pairfreq_, pairfreq)
    assert_array_equal(clf.coupling_matrix_, coupling_matrix)
    assert_array_equal(clf.local_fields_, fields)
    assert_array_equal(clf._compute_energy(X), compute_energy(clf, X))
