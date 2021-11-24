import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, KBinsDiscretizer
from efc import EnergyBasedFlowClassifier

def run(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, shuffle=True, test_size=0.3)

    #creating a model with class 0 as base class
    clf = EnergyBasedFlowClassifier()
    clf.fit(X_train, y_train, base_class=0)
    y_pred, y_energies_test = clf.predict(X_test, return_energies=True)


    #computing energies of train samples to plot
    X_train = MaxAbsScaler().fit_transform(X_train)
    X_train = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='quantile').fit_transform(X_train).astype("int64")
    y_energies_train = clf.estimators_[0]._compute_energy(X_train)


    #ploting energies 
    class0_idx = np.where(y_test == 0)[0]
    class1_idx = np.where(y_test == 1)[0]
    class0_train_idx = np.where(y_train == 0)[0]

    class0_energies = y_energies_test[class0_idx]
    class1_energies = y_energies_test[class1_idx]
    class0_train_energies = y_energies_train[class0_train_idx]
    cutoff = clf.estimators_[0].cutoff_

    bins = np.histogram(y_energies_test, bins=60)[1]

    plt.hist(class0_train_energies, bins, facecolor="green", alpha=0.5, ec='white', linewidth=0.3, label="class0-train")
    plt.hist(class1_energies, bins,  facecolor="#006680", alpha=0.7, ec='white', linewidth=0.3, label="class1-test")
    plt.hist(class0_energies, bins, facecolor="#b3b3b3", alpha=0.7, ec='white', linewidth=0.3, label="class0-test")
    plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
    plt.legend()
    # plt.xlim([500, 2250])
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    plt.show()


X, y = make_classification(n_samples=1000, n_classes=2)


run(X, y)