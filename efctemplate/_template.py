"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from joblib import Parallel, delayed

class EnergyBasedFlowClassifier(ClassifierMixin, BaseEstimator):
    """ The Energy-based Flow Classifier algorithm.

    Parameters
    ----------
    max_bin : int, default=`30`
        The maximum value that a feature can assume after discretization.

    pseudocounts : float, default=`0.5`
        The weight of the pseudocounts added to empirical frequencies. Must be in the interval
        `(0,1)`.
    
    cutoff_quantile : float, default=`0.95`
        The quantile used to define the model's energy threshold. It must be in range `(0,1)`.

    n_jobs : int, default=None
        The number of parallel jobs to run on :meth:`fit`. and :meth:`predict`..
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. 

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    n_classes_ : int
        The number of classes seen at :meth:`fit`.
    
    estimators_ : list of EnergyClassifierBase
        The collection of fitted sub-estimators.


    """  
    def __init__(self, max_bin=30, pseudocounts=0.5, cutoff_quantile=0.99, n_jobs=None):
        self.max_bin = max_bin
        self.pseudocounts = pseudocounts
        self.cutoff_quantile = cutoff_quantile
        self.n_jobs = n_jobs


    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        
        if np.unique(y).shape[0] <= 2:
            self.classes_ = [0]
        else:
            self.classes_ = list(np.unique(y))

        self.n_classes_ = len(self.classes_)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                                    delayed(BaseEFC(self.max_bin, self.pseudocounts, self.cutoff_quantile, label).fit)
                                    (X[np.where(y == label)[0], :]) for label in self.classes_)
        return self

 
    def predict(self, X):
        X = check_array(X)

        energies = np.array(Parallel(n_jobs=self.n_jobs)(
                                    delayed(estimator._compute_energy)(X)
                                    for estimator in self.estimators_
                                    )
                            )

        y_pred = np.empty(X.shape[0], dtype=int)
        y_energies = np.empty(X.shape[0], dtype='float64')

        if len(self.classes_) <= 2:
            for row in range(X.shape[0]):
                y_pred[row] = energies[:, row] > self.estimators_[0].cutoff_
                y_energies[row] = energies[:, row]
        else:
            for row in range(X.shape[0]):
                min_energy = np.min(energies[:, row])
                indx = np.where(energies[:, row] == min_energy)[0][0]
                y_energies[row] = min_energy
                if min_energy < self.estimators_[indx].cutoff_:
                    y_pred[row] = self.estimators_[indx].label
                else:
                    y_pred[row] = 100
       
        self.y_energies_ = y_energies

        return y_pred


class BaseEFC(ClassifierMixin, BaseEstimator):
    def __init__(self, max_bin=30, pseudocounts=0.5, cutoff_quantile=0.99, label=0):
        self.max_bin = max_bin
        self.pseudocounts = pseudocounts
        self.cutoff_quantile = cutoff_quantile
        self.label = label

    def _site_freq(self):
        n_attr = self.X_.shape[1]
        sitefreq = np.empty((n_attr, self.max_bin),dtype='float64')
        for i in range(n_attr):
            for aa in range(self.max_bin):
                sitefreq[i,aa] = np.sum(np.equal(self.X_[:,i],aa))

        sitefreq /= self.X_.shape[0]
        sitefreq = (1-self.pseudocounts)*sitefreq + self.pseudocounts/self.max_bin
        self.sitefreq_ = sitefreq

    def _pair_freq(self):
        n_attr = self.X_.shape[1]
        pairfreq = np.zeros((n_attr, self.max_bin, n_attr, self.max_bin),dtype='float')
        for i in range(n_attr):
            for j in range(n_attr):
                unique, counts = np.unique(self.X_[:, [i, j]], return_counts=True, axis=0)
                for (pair, count) in zip(unique, counts):
                    pairfreq[i, pair[0], j, pair[1]] = count

        pairfreq /= self.X_.shape[0]
        pairfreq = (1-self.pseudocounts)*pairfreq + self.pseudocounts/(self.max_bin**2)

        for i in range(n_attr):
            for am_i in range(self.max_bin):
                for am_j in range(self.max_bin):
                    if (am_i==am_j):
                        pairfreq[i,am_i,i,am_j] = self.sitefreq_[i,am_i]
                    else:
                        pairfreq[i,am_i,i,am_j] = 0.0

        self.pairfreq_ = pairfreq

    def _coupling(self):
        n_attr = self.sitefreq_.shape[0]
        corr_matrix = np.empty(((n_attr)*(self.max_bin-1), (n_attr)*(self.max_bin-1)),dtype='float')
        for i in range(n_attr):
            for j in range(n_attr):
                for am_i in range(self.max_bin-1):
                    for am_j in range(self.max_bin-1):
                        corr_matrix[i*(self.max_bin-1)+am_i, j*(self.max_bin-1)+am_j] = self.pairfreq_[i,am_i,j,am_j] - self.sitefreq_[i,am_i]*self.sitefreq_[j,am_j]

        inv_corr = np.linalg.inv(corr_matrix)
        self.coupling_matrix_ = np.exp(np.negative(inv_corr))


    def _local_fields(self):
        n_inst = self.sitefreq_.shape[0]
        fields = np.empty((n_inst*(self.max_bin-1)),dtype='double')
        for i in range(n_inst):
            for ai in range(self.max_bin-1):
                fields[i*(self.max_bin-1) + ai] = self.sitefreq_[i,ai]/self.sitefreq_[i,self.max_bin-1]
                for j in range(n_inst):
                    for aj in range(self.max_bin-1):
                        fields[i*(self.max_bin-1) + ai] /= self.coupling_matrix_[i*(self.max_bin-1) + ai, j*(self.max_bin-1) + aj]**self.sitefreq_[j,aj]

        self.local_fields_ = fields

    def _compute_energy(self, X):
        n_inst, n_attr = X.shape[0], X.shape[1]
        energies = np.empty(n_inst, dtype= 'float64')
        for i in range(n_inst):
            e = 0
            for j in range(n_attr-1):
                j_value = X[i, j]
                if j_value != (self.max_bin-1):
                    for k in range(j,n_attr):
                        k_value = X[i,k]
                        if k_value != (self.max_bin-1):
                            e -= (self.coupling_matrix_[j*(self.max_bin-1) + j_value, k*(self.max_bin-1) + k_value])
                    e -= (self.local_fields_[j*(self.max_bin-1) + j_value])
            energies[i] = e
        return energies

    def _define_cutoff(self):
        energies = self._compute_energy(self.X_)
        energies = np.sort(energies, axis=None)
        self.cutoff_ = energies[int(energies.shape[0]*self.cutoff_quantile)]

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