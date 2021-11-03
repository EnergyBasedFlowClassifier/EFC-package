import pytest

from sklearn.utils.estimator_checks import check_estimator

import sys
sys.path.insert(0, '/home/munak98/Documents/project-template')
from efc._energyclassifier import EnergyBasedFlowClassifier



@pytest.mark.parametrize(
    "estimator",
    [EnergyBasedFlowClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator, generate_only=True)


for idx, (estimator, func) in enumerate(test_all_estimators(EnergyBasedFlowClassifier())):
    if idx != 20:
        print(idx)
        print(func)
        func(estimator)