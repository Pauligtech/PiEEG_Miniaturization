import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from moabb.pipelines.utils import FilterBank

clf = SVC(kernel="linear")
fb = FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=6)))
# pipe = make_pipeline(fb, SelectKBest(score_func=mutual_info_classif, k=10), clf)
pipe = make_pipeline(fb, clf)

# this is what will be loaded
PIPELINE = {
    "name": "FBCSP + SVM",
    "paradigms": ["FilterBankMotorImagery"],
    "pipeline": pipe,
}
