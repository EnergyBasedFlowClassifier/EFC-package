import numpy as np
import sys
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import classification_report
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MaxAbsScaler, KBinsDiscretizer, LabelEncoder
import sys, math
from _template import EnergyBasedFlowClassifier

# malicious_names = [['normal.'], ['back.', 'smurf.', 'teardrop.', 'neptune.', 'land.', 'pod.'],
# ['ipsweep.',  'portsweep.',  'satan.',  'nmap.'], ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'spy.',
#  'warezclient.', 'warezmaster.', 'phf.'], ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']]


# dataset = pd.read_csv("Multi-class EFC/KDDCup99/Data/kddcup.data_10_percent", sep = ',', header=None)
# dataset = dataset[:dataset.shape[0]//4]


X, y = load_iris(True, True)
dataset = pd.concat([X, y], axis=1)
dataset.dropna(inplace=True)

#find symbolic and continous features indexes
symbolic, continuous = [], []
for i in range(dataset.shape[1]-1):
    if is_numeric_dtype(dataset.iloc[:, i]):
        continuous.append(i)
    else:
        symbolic.append(i)

print(symbolic)
print(continuous)

# split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.33, random_state=42, shuffle=True, stratify=dataset.iloc[:, -1])

#encode symbolic features
if symbolic != []:
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    X_train.iloc[:, symbolic] = enc.fit_transform(X_train.iloc[:, symbolic])
    X_test.iloc[:, symbolic] = enc.transform(X_test.iloc[:, symbolic])
    X_test.iloc[:, symbolic] = np.nan_to_num(X_test.iloc[:, symbolic].astype('float'), nan=np.max(X_test.iloc[:, symbolic])+1)

#normalize continuos features
norm = MaxAbsScaler()
X_train.iloc[:, continuous] = norm.fit_transform(X_train.iloc[:, continuous])
X_test.iloc[:, continuous] = norm.transform(X_test.iloc[:, continuous])

#discretize continuos features
disc = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='quantile')
X_train.iloc[:, continuous] = disc.fit_transform(X_train.iloc[:, continuous])
X_test.iloc[:, continuous] = disc.transform(X_test.iloc[:, continuous])

#encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# encode labels
# for idx in range(len(malicious_names)):
#     y_train = [idx if value in malicious_names[idx] else value for value in y_train]
#     y_test = [idx if value in malicious_names[idx] else value for value in y_test]

X_train = np.array(X_train, dtype='int64')
X_test = np.array(X_test, dtype='int64')
y_train = np.array(y_train)
y_test = np.array(y_test)

cls = EnergyBasedFlowClassifier(max_bin=66, pseudocounts=0.5, n_jobs=-1)
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)

print(classification_report(y_test, y_pred))

