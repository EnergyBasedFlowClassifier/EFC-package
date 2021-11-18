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
from sklearn.preprocessing import OrdinalEncoder, MaxAbsScaler, KBinsDiscretizer
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '/home/munak98/Documents/project-template')
from efc import EnergyBasedFlowClassifier

subclasses = [['normal.'], ['back.', 'smurf.', 'teardrop.', 'neptune.', 'land.', 'pod.'],
['ipsweep.',  'portsweep.',  'satan.',  'nmap.'], ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'spy.',
 'warezclient.', 'warezmaster.', 'phf.'], ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']]

classes = ["Normal", "DoS", "Probe", "U2R", "R2L"]


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
disc = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
X_train[:, continuous] = disc.fit_transform(X_train[:, continuous])
X_test[:, continuous] = disc.transform(X_test[:, continuous]).astype('int64')

#encode_labels
for idx in range(len(subclasses)):
    y_train = [classes[idx] if value in subclasses[idx] else value for value in y_train]
    y_test = [classes[idx] if value in subclasses[idx] else value for value in y_test]


#trai and test EFC
cls = EnergyBasedFlowClassifier()
cls.fit(X_train, y_train)
y_pred, y_energies = cls.predict(X_test, return_energies=True)

print(classification_report(y_test, y_pred))

