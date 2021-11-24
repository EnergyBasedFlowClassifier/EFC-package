import numpy as np

from seaborn import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from efc import EnergyBasedFlowClassifier

#load dataset
data = load_dataset("titanic")
data.columns = range(data.shape[1])

data.replace([np.inf, -np.inf], np.nan)
data.dropna(inplace=True)

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)


#encode categorical features
categorical = X_train.select_dtypes(exclude=np.number).columns.values
if categorical.size != 0:
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    X_train.loc[:, categorical] = enc.fit_transform(X_train.loc[:, categorical])
    X_test.loc[:, categorical] = enc.transform(X_test.loc[:, categorical])
    X_test.loc[:, categorical] = np.nan_to_num(X_test.loc[:, categorical].astype('float'), nan=np.max(X_test.loc[:, categorical])+1)

categorical = [x-1 for x in categorical]

#train and test EFC
cls = EnergyBasedFlowClassifier(cutoff_quantile=0.99)
cls.fit(X_train, y_train, base_class=1, categorical_columns=categorical)
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
