"""
This is a module that implements the Energy-based Flow Classifier base estimator.
"""
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from ._base_fast import site_freq
from ._base_fast import pair_freq
from ._base_fast import coupling
from ._base_fast import local_fields
from ._base_fast import compute_energy
from ._base_fast import breakdown_energy


class BaseEFC(ClassifierMixin, BaseEstimator):
    """The Base estimator used by the EnergyBasedFlowClassifier estimator.

    Note:
        This class must not be instantiated by end users.

    Parameters
    ----------
    max_bin : int
        The maximum value assumed by a feature in X.

    pseudocounts : float, default=`0.5`
        The weight of the pseudocounts added to empirical frequencies. Must be in the interval
        `(0,1)`.

    cutoff_quantile : float, default=`0.95`
        The quantile used to define the model's energy threshold. It must be in range `(0,1)`.

    Attributes
    ----------

    X_ : ndarray, (n_samples, n_features)
        The training set.

    sitefreq_ : ndarray, shape (n_feature, max_bin)
        Observed frequency of attribute values ​​in each attribute.

    pairfreq_ : ndarray, shape (n_feature, max_bin, n_feature, max_bin)
        Observed frequency of attribute value pairs in attribute pairs.

    coupling_matrix_ : ndarray, shape (n_feature*max_bin, n_feature*max_bin)

    local_fields_ : ndarray, shape (n_samples*max_bin,)

    cutoff_ : float
        Energy cutoff used for classification.


    """

    def __init__(self, max_bin=30, pseudocounts=0.5, cutoff_quantile=0.95):
        self.max_bin = max_bin
        self.pseudocounts = pseudocounts
        self.cutoff_quantile = cutoff_quantile

    """
    
    Fit the Base estimator for the EnergyBasedFlowClassifier model according to the given training data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The training input samples.

    Returns
    -------
    self : object
        Returns self.
    """

    def fit(self, X):
        self.X_ = X
        self.sitefreq_ = self._site_freq()
        self.pairfreq_ = self._pair_freq()
        self.coupling_matrix_ = self._coupling()
        self.local_fields_ = self._local_fields()
        self.coupling_matrix_ = np.log(self.coupling_matrix_)
        self.local_fields_ = np.log(self.local_fields_)
        self.cutoff_ = self._define_cutoff()
        return self

    def _site_freq(self):
        return site_freq(self.X_, self.pseudocounts, self.max_bin)

    def _pair_freq(self):
        return pair_freq(self.X_, self.sitefreq_, self.pseudocounts, self.max_bin)

    def _coupling(self):
        return coupling(self.pairfreq_, self.sitefreq_, self.pseudocounts, self.max_bin)

    def _local_fields(self):
        return local_fields(
            self.coupling_matrix_,
            self.pairfreq_,
            self.sitefreq_,
            self.pseudocounts,
            self.max_bin,
        )

    def _compute_energy(self, X):
        return compute_energy(self, X)

    def _breakdown_energy(self, x):
        return breakdown_energy(self, x)

    def _define_cutoff(self):
        energies = compute_energy(self, self.X_)
        energies = np.sort(energies, axis=None)
        return energies[int(energies.shape[0] * self.cutoff_quantile)]

