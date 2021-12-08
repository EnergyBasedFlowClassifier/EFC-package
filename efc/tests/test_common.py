import pytest

from sklearn.utils.estimator_checks import check_estimator
from efc import EnergyBasedFlowClassifier


@pytest.mark.parametrize("estimator", [EnergyBasedFlowClassifier()])
def test_all_estimators(estimator):
    return check_estimator(estimator)
