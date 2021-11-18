import numpy as np
import warnings

from seaborn import load_dataset
from matplotlib import pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import KBinsDiscretizer
from efc import EnergyBasedFlowClassifier

#load dataset
data = load_dataset("titanic")
data.replace([np.inf, -np.inf], np.nan)
data.dropna(inplace=True)

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

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
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    X_train[:, continuous] = disc.fit_transform(X_train[:, continuous])
    X_test[:, continuous] = disc.transform(X_test[:, continuous]).astype('int64')

#train and test EFC
cls = EnergyBasedFlowClassifier(cutoff_quantile=0.99)
cls.fit(X_train, y_train, base_class=1)
y_pred, y_energies = cls.predict(X_test, return_energies=True)

#plot energies
survived = np.where(y_test == 0)[0]
died = np.where(y_test == 1)[0]

normal_energies = y_energies[survived]
malicious_energies = y_energies[died]
cutoff = cls.estimators_[0].cutoff_

bins = np.histogram(y_energies, bins=60)[1]

plt.hist(malicious_energies, bins,  facecolor="#006680", alpha=0.7, ec='white', linewidth=0.3, label="Survived")
plt.hist(normal_energies, bins, facecolor="#b3b3b3", alpha=0.7, ec='white', linewidth=0.3, label="Died")
plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
plt.legend()
plt.xlabel("Energy", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.show()
