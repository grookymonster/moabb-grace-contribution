"""
Cross-Dataset Motor Imagery Classification
===========================================

This example shows how to train on one dataset and test on another using
:class:`moabb.evaluations.CrossDatasetEvaluation`.

The evaluation automatically aligns channels and resampling rates across
datasets via ``paradigm.match_all``, fits each pipeline once on all
training data, and reports per-subject per-session scores on the test
dataset.
"""

from collections import OrderedDict

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from moabb.datasets import BNCI2014_001, Zhou2016
from moabb.evaluations import CrossDatasetEvaluation
from moabb.paradigms import LeftRightImagery


# %%
# Define paradigm and datasets
# -----------------------------
paradigm = LeftRightImagery()

train_dataset = BNCI2014_001()
test_dataset = Zhou2016()

# %%
# Create pipeline
# ---------------
pipelines = OrderedDict()
pipelines["TS+LDA"] = make_pipeline(
    Covariances(estimator="oas"),
    TangentSpace(),
    LDA(),
)

# %%
# Run evaluation
# ---------------
evaluation = CrossDatasetEvaluation(
    paradigm=paradigm,
    train_datasets=train_dataset,
    test_datasets=test_dataset,
)

results = evaluation.process(pipelines)
print(results[["pipeline", "dataset", "subject", "session", "score"]])
