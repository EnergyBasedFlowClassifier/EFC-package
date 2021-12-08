"""
This is a module that implements the Energy-based Flow Classifier main interface.
"""
import warnings
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import MaxAbsScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from joblib import Parallel, delayed

from ._base import BaseEFC


class EnergyBasedFlowClassifier(ClassifierMixin, BaseEstimator):
    """The Energy-based Flow Classifier algorithm.

    Parameters
    ----------
    pseudocounts : float, default=`0.5`
        The weight of the pseudocounts added to empirical
        frequencies. Must be in the interval `(0,1)`.

    cutoff_quantile : float, default=`0.95`
        The quantile used to define the model's energy threshold.
        It must be in range `(0,1)`.

    n_bins : int, default=`30`
        The number of bins to produce when discretizing data features.
        Using the quantile strategy.

    n_jobs : int, default=None
        The number of parallel jobs to run on :meth:`fit`
        and :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Attributes
    ----------
    max_bin_ : int
        The maximum value of the features in X.

    n_features_in_ : int
        The number of features in X.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    target_type_ : string
        The type of target seen at :meth:`fit` according to
        :meth:`utils.multiclass.type_of_target`.

    base_class_idx_ : int
        The index of the base class passed to
        :meth:`fit` in the classes_ vector. Only used when target is binary.

    estimators_ : list of BaseEFC instances
        The collection of fitted sub-estimators. When the target
        is binary, this collection consists of only one estimator.


    """

    def __init__(self, pseudocounts=0.5, cutoff_quantile=0.95, n_bins=30, n_jobs=None):
        self.pseudocounts = pseudocounts
        self.cutoff_quantile = cutoff_quantile
        self.n_bins = n_bins
        self.n_jobs = n_jobs

    def _more_tags(self):
        return {"poor_score": True}

    def fit(self, X, y, base_class=None, categorical_columns=[]):
        """Fit the Energy-based Flow Classifier model according to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values.

        base_class : int or string, depending on y's dtype
            Only used for binary target. Defines the class that will be used for training among the classes in the target vector. If no class is passed, the first class in the array np.unique(y) will be used.

        categorical_columns : array-like
            Indicates categorical attributes so that they are not normalized and discretized as numeric attributes. These attributes must be encoded before being passed to EFC.

        Returns
        -------
        self : object
            Returns the fitted estimator.
        """
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")

        numeric_transformer = Pipeline(
            steps=[
                ("scaler", MaxAbsScaler()),
                (
                    "discretizer",
                    KBinsDiscretizer(
                        n_bins=self.n_bins, encode="ordinal", strategy="quantile"
                    ),
                ),
            ]
        )

        self.preprocessor_ = ColumnTransformer(
            [("categorical", "passthrough", categorical_columns)],
            remainder=numeric_transformer,
        )

        X, y = check_X_y(X, y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X = self.preprocessor_.fit_transform(X).astype("int64")

        self.max_bin_ = np.max(X) + 1
        self.n_features_in_ = X.shape[1]
        self.target_type_ = type_of_target(y)
        self.classes_, y = np.unique(y, return_inverse=True)

        if self.target_type_ not in ["binary", "multiclass"]:
            raise ValueError("Unknown label type: ")

        if self.target_type_ == "binary":
            if base_class is None:
                self.base_class_idx_ = 0
                train_samples = np.where(y == self.base_class_idx_)[0]
            elif base_class in self.classes_:
                self.base_class_idx_ = np.where(self.classes_ == base_class)[0][0]
                train_samples = np.where(y == self.base_class_idx_)[0]
            else:
                raise ValueError("Base class not in target classes.")

            self.estimators_ = [
                BaseEFC(self.max_bin_, self.pseudocounts, self.cutoff_quantile).fit(
                    X[train_samples, :]
                )
            ]

        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(
                    BaseEFC(self.max_bin_, self.pseudocounts, self.cutoff_quantile).fit
                )(X[np.where(y == idx)[0], :])
                for idx in range(len(self.classes_))
            )

        return self

    def predict(self, X, return_energies=False, unknown_class=False):
        """
        Perform classification on samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples for classification.

        return_energies : boolean, default=False,
            Whether to return the energy vector of samples in X.

        unknown_class : boolean, default=False,
            Whether to use the `unknown` class for samples with low similarity to all training classes. If targets dtype is numeric, the unknown class will be represented by -1.


        Returns
        -------
        y_pred : array-like, shape (n_samples, )
            Class labels for samples in X.

        y_energies : array-like, shape (n_samples, )
            Computed energies for samples in X.
        """

        X = check_array(X)

        check_is_fitted(self)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in predict is different from the number of features in fit."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X = self.preprocessor_.transform(X).astype("int64")

        energies = np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(estimator._compute_energy)(X) for estimator in self.estimators_
            )
        )

        y_energies = np.empty(X.shape[0], dtype="float64")
        y_pred = np.empty(X.shape[0], dtype=self.classes_.dtype)

        if self.target_type_ == "binary":
            for row in range(X.shape[0]):
                y_energies[row] = energies[:, row]
                if energies[:, row] < self.estimators_[0].cutoff_:
                    y_pred[row] = self.classes_[self.base_class_idx_]
                else:
                    y_pred[row] = self.classes_[self.base_class_idx_ - 1]

        else:
            for row in range(X.shape[0]):
                min_energy = np.min(energies[:, row])
                label_idx = np.where(energies[:, row] == min_energy)[0][0]
                y_energies[row] = min_energy
                y_pred[row] = self.classes_[label_idx]
                if unknown_class:
                    if min_energy > self.estimators_[label_idx].cutoff_:
                        if np.issubdtype(self.classes_.dtype, np.number):
                            y_pred[row] = -1
                        else:
                            y_pred[row] = "unknown"

        if return_energies:
            return y_pred, y_energies
        return y_pred
