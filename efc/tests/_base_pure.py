"""
This is a module that implements the Energy-based Flow Classifier.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseEFC(ClassifierMixin, BaseEstimator):
    """The Base estimator used by the Energy-based Flow Classifier.

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

    """Fit the Base estimator for the Energy-based Flow Classifier model according to the given training data.

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
        n_attr = self.X_.shape[1]
        sitefreq = np.empty((n_attr, self.max_bin), dtype="double")
        for i in range(n_attr):
            for aa in range(self.max_bin):
                sitefreq[i, aa] = np.sum(np.equal(self.X_[:, i], aa))

        sitefreq /= self.X_.shape[0]
        sitefreq = (1 - self.pseudocounts) * sitefreq + self.pseudocounts / self.max_bin

        return sitefreq

    def _cantor(self, x, y):
        return (x + y) * (x + y + 1) / 2 + y

    def _pair_freq(self):
        n_inst = self.X_.shape[0]
        n_attr = self.X_.shape[1]
        pairfreq = np.zeros(
            (n_attr, self.max_bin, n_attr, self.max_bin), dtype="double"
        )

        for i in range(n_attr):
            for j in range(n_attr):
                c = self._cantor(self.X_[:, i], self.X_[:, j])
                unique, aaIdx = np.unique(c, True)
                for x, item in enumerate(unique):
                    pairfreq[i, self.X_[aaIdx[x], i], j, self.X_[aaIdx[x], j]] = np.sum(
                        np.equal(c, item)
                    )

        pairfreq /= n_inst
        pairfreq = (1 - self.pseudocounts) * pairfreq + self.pseudocounts / (
            self.max_bin ** 2
        )

        for i in range(n_attr):
            for ai in range(self.max_bin):
                for aj in range(self.max_bin):
                    if ai == aj:
                        pairfreq[i, ai, i, aj] = self.sitefreq_[i, ai]
                    else:
                        pairfreq[i, ai, i, aj] = 0.0
        return pairfreq

    def _coupling(self):
        n_attr = self.sitefreq_.shape[0]
        corr_matrix = np.empty(
            (n_attr * (self.max_bin - 1), n_attr * (self.max_bin - 1)), dtype="double"
        )
        for i in range(n_attr):
            for j in range(n_attr):
                for ai in range(self.max_bin - 1):
                    for aj in range(self.max_bin - 1):
                        corr_matrix[
                            i * (self.max_bin - 1) + ai, j * (self.max_bin - 1) + aj
                        ] = (
                            self.pairfreq_[i, ai, j, aj]
                            - self.sitefreq_[i, ai] * self.sitefreq_[j, aj]
                        )

        inv_corr = np.linalg.inv(corr_matrix)
        return np.exp(np.negative(inv_corr))

    def _local_fields(self):
        n_inst = self.sitefreq_.shape[0]
        fields = np.empty((n_inst * (self.max_bin - 1)), dtype="double")

        for i in range(n_inst):
            for ai in range(self.max_bin - 1):
                fields[i * (self.max_bin - 1) + ai] = (
                    self.sitefreq_[i, ai] / self.sitefreq_[i, self.max_bin - 1]
                )
                for j in range(n_inst):
                    for aj in range(self.max_bin - 1):
                        fields[i * (self.max_bin - 1) + ai] /= (
                            self.coupling_matrix_[
                                i * (self.max_bin - 1) + ai, j * (self.max_bin - 1) + aj
                            ]
                            ** self.sitefreq_[j, aj]
                        )

        return fields

    def _compute_energy(self, X):
        n_inst, n_attr = X.shape[0], X.shape[1]
        energies = np.empty(n_inst, dtype="double")
        for i in range(n_inst):
            e = 0
            for j in range(n_attr):
                j_value = X[i, j]
                if j_value != (self.max_bin - 1):
                    for k in range(j+1, n_attr):
                        k_value = X[i, k]
                        if k_value != (self.max_bin - 1):
                            e -= self.coupling_matrix_[
                                j * (self.max_bin - 1) + j_value,
                                k * (self.max_bin - 1) + k_value,
                            ]
                    e -= self.local_fields_[j * (self.max_bin - 1) + j_value]
            energies[i] = e
        return energies

    def _define_cutoff(self):
        energies = self._compute_energy(self.X_)
        energies = np.sort(energies, axis=None)
        return energies[int(energies.shape[0] * self.cutoff_quantile)]
