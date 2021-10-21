"""
This is a module that implements the Energy-based Flow Classifier.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from joblib import Parallel, delayed
from pandas.api.types import is_numeric_dtype


class EnergyBasedFlowClassifier(ClassifierMixin, BaseEstimator):
    """ The Energy-based Flow Classifier algorithm.

    Parameters
    ----------
    pseudocounts : float, default=`0.5`
        The weight of the pseudocounts added to empirical
        frequencies. Must be in the interval `(0,1)`.

    cutoff_quantile : float, default=`0.95`s
        The quantile used to define the model's energy threshold.
        It must be in range `(0,1)`.

    n_jobs : int, default=None
        The number of parallel jobs to run on :meth:`fit`
        and :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Attributes
    ----------
    max_bin_ : int
        The maximum value assumed by a feature in X.

    n_features_in_ : int
        The number of features in X.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    n_classes_ : int
        The number of classes seen during :meth:`fit`.

    estimators_ : list of EnergyClassifierBase
        The collection of fitted sub-estimators. When the target
        is binary, this collection consists of only one estimator.


    """

    def __init__(self, pseudocounts=0.5, cutoff_quantile=0.95, n_jobs=None):
        self.pseudocounts = pseudocounts
        self.cutoff_quantile = cutoff_quantile
        self.n_jobs = n_jobs

    def _more_tags(self):
        return {'poor_score': True}

    def fit(self, X, y):
        """Fit the Energy-based Flow Classifier model according to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.


        Returns
        -------
        self : object
            Returns self.
        """
        if y is None:
            raise ValueError(
                "requires y to be passed, but the target y is None")

        X, y = check_X_y(X, y)

        X = X.astype("int64")

        self.max_bin_ = np.max(X) + 1

        self.n_features_in_ = X.shape[1]

        self.classes_, y = np.unique(y, return_inverse=True)

        self.n_classes_ = len(self.classes_)

        if self.n_classes_ > 10:
            raise ValueError("Unknown label type: ")

        if self.n_classes_ <= 2:
            self.estimators_ = [BaseEFC(self.max_bin_, self.pseudocounts,
                                        self.cutoff_quantile).fit(X[np.where(y == 0)[0], :])]

        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(BaseEFC(self.max_bin_, self.pseudocounts,
                        self.cutoff_quantile).fit)
                (X[np.where(y == idx)[0], :]) for idx in range(self.n_classes_))

        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples for classification.

        Returns
        -------
        y_pred : array-like, shape (n_samples, )
            Class labels for samples in X.
        """

        X = check_array(X, dtype='int64')
        check_is_fitted(self)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit.")

        energies = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(estimator._compute_energy)(X)
            for estimator in self.estimators_
        )
        )

        y_energies = np.empty(X.shape[0], dtype="float64")
        y_pred = np.empty(X.shape[0], dtype=self.classes_.dtype)
        if self.n_classes_ <= 2:
            for row in range(X.shape[0]):
                label_idx = int(energies[:, row] > self.estimators_[0].cutoff_)
                y_pred[row] = self.classes_[label_idx]
                y_energies[row] = energies[:, row]
        else:
            for row in range(X.shape[0]):
                min_energy = np.min(energies[:, row])
                label_idx = np.where(energies[:, row] == min_energy)[0][0]
                y_energies[row] = min_energy
                if min_energy < self.estimators_[label_idx].cutoff_:
                    y_pred[row] = self.classes_[label_idx]
                else:
                    if is_numeric_dtype(self.classes_):
                        y_pred[row] = 100
                    else:
                        y_pred[row] = "suspicious"

        return y_pred


class BaseEFC(ClassifierMixin, BaseEstimator):
    """ The Base estimator used by the Energy-based Flow Classifier.
    Used to model only one class.

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
        self._site_freq()
        self._pair_freq()
        self._coupling()
        self._local_fields()
        self.coupling_matrix_ = np.log(self.coupling_matrix_)
        self.local_fields_ = np.log(self.local_fields_)
        self._define_cutoff()

        return self

    def _site_freq(self):
        n_attr = self.X_.shape[1]
        sitefreq = np.empty((n_attr, self.max_bin), dtype='float64')
        for i in range(n_attr):
            for aa in range(self.max_bin):
                sitefreq[i, aa] = np.sum(np.equal(self.X_[:, i], aa))

        sitefreq /= self.X_.shape[0]
        sitefreq = ((1 - self.pseudocounts) * sitefreq
                    + self.pseudocounts / self.max_bin)

        self.sitefreq_ = sitefreq
        return

    def _pair_freq(self):
        n_attr = self.X_.shape[1]
        pairfreq = np.zeros((n_attr, self.max_bin, n_attr, self.max_bin),
                            dtype='float')

        for i in range(n_attr):
            for j in range(n_attr):
                unique, counts = np.unique(self.X_[:, [i, j]],
                                           return_counts=True, axis=0)
                for (pair, count) in zip(unique, counts):
                    pairfreq[i, pair[0], j, pair[1]] = count

        pairfreq /= self.X_.shape[0]
        pairfreq = ((1 - self.pseudocounts) * pairfreq
                    + self.pseudocounts / (self.max_bin**2))

        for i in range(n_attr):
            for ai in range(self.max_bin):
                for aj in range(self.max_bin):
                    if (ai == aj):
                        pairfreq[i, ai, i, aj] = self.sitefreq_[i, ai]
                    else:
                        pairfreq[i, ai, i, aj] = 0.0

        self.pairfreq_ = pairfreq
        return

    def _coupling(self):
        n_attr = self.sitefreq_.shape[0]
        corr_matrix = np.empty((n_attr * (self.max_bin - 1),
                                n_attr * (self.max_bin - 1)), dtype='float')
        for i in range(n_attr):
            for j in range(n_attr):
                for ai in range(self.max_bin - 1):
                    for aj in range(self.max_bin - 1):
                        corr_matrix[i * (self.max_bin - 1) + ai,
                                    j * (self.max_bin - 1) + aj] = (self.pairfreq_[i, ai, j, aj]
                                                                    - self.sitefreq_[i, ai]
                                                                    * self.sitefreq_[j, aj])

        inv_corr = np.linalg.inv(corr_matrix)
        self.coupling_matrix_ = np.exp(np.negative(inv_corr))
        return

    def _local_fields(self):
        n_inst = self.sitefreq_.shape[0]
        fields = np.empty((n_inst * (self.max_bin - 1)), dtype='double')

        for i in range(n_inst):
            for ai in range(self.max_bin - 1):
                fields[i * (self.max_bin - 1) + ai] = (self.sitefreq_[i, ai]
                                                       / self.sitefreq_[i, self.max_bin - 1])
                for j in range(n_inst):
                    for aj in range(self.max_bin - 1):
                        fields[i * (self.max_bin - 1) + ai] /= (
                            self.coupling_matrix_[i * (self.max_bin - 1) + ai, j * (self.max_bin - 1) + aj]**self.sitefreq_[j, aj])

        self.local_fields_ = fields
        return

    def _compute_energy(self, X):
        n_inst, n_attr = X.shape[0], X.shape[1]
        energies = np.empty(n_inst, dtype='float64')
        for i in range(n_inst):
            e = 0
            for j in range(n_attr - 1):
                j_value = X[i, j]
                if j_value != (self.max_bin - 1):
                    for k in range(j, n_attr):
                        k_value = X[i, k]
                        if k_value != (self.max_bin - 1):
                            e -= (self.coupling_matrix_[j * (self.max_bin - 1)
                                                        + j_value, k * (self.max_bin - 1) + k_value])
                    e -= (self.local_fields_[j * (self.max_bin - 1) + j_value])
            energies[i] = e
        return energies

    def _define_cutoff(self):
        energies = self._compute_energy(self.X_)
        energies = np.sort(energies, axis=None)
        self.cutoff_ = energies[int(energies.shape[0] * self.cutoff_quantile)]
        return
