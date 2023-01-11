"""
=========================================
Understanding EFC's predictions
=========================================

Breaking down the energy value of a record by 
accessing the local field and coupling values of this record.

In this example the EFC is trained with binary targets.

"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from efc import EnergyBasedFlowClassifier

# loading the toy dataset from scikit-learn
X, y = load_breast_cancer(return_X_y=True)

# spliting train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, shuffle=True, test_size=0.3
)


# train EFC
clf = EnergyBasedFlowClassifier(n_bins=10, cutoff_quantile=0.99)
clf.fit(X_train, y_train, base_class=0)

# call the fitted preprocessor to transform just one record
x = clf.preprocessor_.transform(X_test[1, :].reshape(1, -1)).astype("int64")

# call the _breakdown_energy method of the base estimator on this record
sample_energy, sample_fields, sample_couplings = clf.estimators_[0]._breakdown_energy(x)

# access the desired features
print(f"Total energy for sample {x[0]}: ", sample_energy)

print(f"Local field of feature 25 with value {x[0, 25]}: ", sample_fields[25])

print(f"Coupling of features 10 and 5 with values {x[0, 10], x[0, 5]}: ", sample_couplings[10,5])

