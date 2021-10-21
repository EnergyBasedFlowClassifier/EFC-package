"""
=======================================
Using the Energy Based Flow Classifier
=======================================

An example plot of :class:`efc.energyclassifier.EnergyBasedFlowClassifier`
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import classification_report
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MaxAbsScaler, KBinsDiscretizer, LabelEncoder
import sys
sys.path.insert(0, '/home/munak98/Documents/project-template')
from efc import EnergyBasedFlowClassifier


X, y = load_breast_cancer(return_X_y=True)

# split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)


#find symbolic and continous features indexes
symbolic, continuous = [], []
for i in range(X_train.shape[1]):
    if is_numeric_dtype(X_train[:, i]):
        continuous.append(i)
    else:
        symbolic.append(i)


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

#encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


cls = EnergyBasedFlowClassifier()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)

print(classification_report(y_test, y_pred))
