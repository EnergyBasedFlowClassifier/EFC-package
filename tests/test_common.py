import pytest

from sklearn.utils.estimator_checks import check_estimator
from efc import EnergyBasedFlowClassifier



@pytest.mark.parametrize(
    "estimator",
    [EnergyBasedFlowClassifier()]
)

# @pytest.mark.skip(reason="test 20 impossible to pass")
def test_all_estimators(estimator):
    return check_estimator(estimator)


# for idx, (estimator, func) in enumerate(test_all_estimators(EnergyBasedFlowClassifier())):
#     if idx not in []:
#         print(idx)
#         print(func)
#         func(estimator)