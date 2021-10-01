import pytest

from sklearn.utils.estimator_checks import check_estimator

from efctemplate import TemplateEstimator
from efctemplate import TemplateClassifier
from efctemplate import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
