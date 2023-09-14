"""
================================================================
Energy-based Flow Classifier for anomaly detection
================================================================

An example plot of the energies calculated by the :class:`EnergyBasedFlowClassfifier` for benign and malicious samples of cancer patients.

In this example the EFC is trained only with the bening class, making it an anomaly detector

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from efc import EnergyBasedFlowClassifier

# loading the toy dataset from scikit-learn
X, y = load_breast_cancer(return_X_y=True)

# spliting train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, shuffle=True, test_size=0.3
)


# train and test EFC
clf = EnergyBasedFlowClassifier(n_bins=10, cutoff_quantile=0.99)

# Since the target for this dataset is binary, 
# EFC will be used in its single-class version.
# Therefore, we need to define which class will
# be used for training with the parameter base_class=0
clf.fit(X_train, y_train, base_class=0)
y_pred, y_energies = clf.predict(X_test, return_energies=True)


# ploting energies
benign = np.where(y_test == 0)[0]
malicious = np.where(y_test == 1)[0]

benign_energies = y_energies[benign]
malicious_energies = y_energies[malicious]
cutoff = clf.estimators_[0].cutoff_

bins = np.histogram(y_energies, bins=60)[1]

plt.hist(
    malicious_energies,
    bins,
    facecolor="#006680",
    alpha=0.7,
    ec="white",
    linewidth=0.3,
    label="malicious",
)
plt.hist(
    benign_energies,
    bins,
    facecolor="#b3b3b3",
    alpha=0.7,
    ec="white",
    linewidth=0.3,
    label="benign",
)
plt.axvline(cutoff, color="r", linestyle="dashed", linewidth=1)
plt.legend()

plt.xlabel("Energy", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.show()




