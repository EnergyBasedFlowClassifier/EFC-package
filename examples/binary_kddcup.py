"""
=======================================
Using the Energy Based Flow Classifier
=======================================

An example of :class:`efc.energyclassifier.EnergyBasedFlowClassifier`
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import classification_report
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MaxAbsScaler, KBinsDiscretizer, LabelEncoder
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '/home/munak98/Documents/project-template')
from efc import EnergyBasedFlowClassifier

data = pd.read_csv("examples/kddcup.data_10_percent", header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)


#find symbolic and continous features indexes
symbolic, continuous = [], []
for i in range(X_train.shape[1]):
    if is_numeric_dtype(X_train.iloc[:, i]):
        continuous.append(i)
    else:
        symbolic.append(i)
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

#encode symbolic features
if symbolic != []:
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    X_train[:, symbolic] = enc.fit_transform(X_train[:, symbolic])
    X_test[:, symbolic] = enc.transform(X_test[:, symbolic])
    X_test[:, symbolic] = np.nan_to_num(X_test[:, symbolic].astype('float'), nan=np.max(X_test[:, symbolic])+1)

#normalize continuos features
norm = MaxAbsScaler()
X_train[:, continuous] = norm.fit_transform(X_train[:, continuous])
X_test[:, continuous] = norm.transform(X_test[:, continuous])

#discretize continuos features
disc = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='quantile')
X_train[:, continuous] = disc.fit_transform(X_train[:, continuous])
X_test[:, continuous] = disc.transform(X_test[:, continuous]).astype('int64')

#encode_labels
y_train = ["malicious" if value != "normal." else value for value in y_train]
y_test = ["malicious" if value != "normal." else value for value in y_test]


#trai and test EFC
cls = EnergyBasedFlowClassifier()
cls.fit(X_train, y_train, base_class="normal.")
y_pred, y_energies = cls.predict(X_test, return_energies=True)


#plot energies
normal_idx = np.where(y_test == np.array('normal.'))[0]
malicious_idx = np.where(y_test == np.array('malicious'))[0]

normal_energies = y_energies[normal_idx]
malicious_energies = y_energies[malicious_idx]
cutoff = cls.estimators_[0].cutoff_

bins = np.histogram(y_energies, bins=60)[1]

plt.hist(malicious_energies, bins,  facecolor="#006680", alpha=0.7, ec='white', linewidth=0.3, label="malicious")
plt.hist(normal_energies, bins, facecolor="#b3b3b3", alpha=0.7, ec='white', linewidth=0.3, label="normal")
plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
plt.legend()
plt.xlabel("Energy", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.show()
