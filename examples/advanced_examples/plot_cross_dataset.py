"""
Cross-dataset motor imagery classification
===========================================

This example shows how to train on one dataset (BNCI2014_001) and
test on another (Zhou2016) using ``CrossDatasetEvaluation``.
Channel alignment and resampling are handled automatically.
"""

import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from moabb import set_log_level
from moabb.datasets import BNCI2014001, Zhou2016
from moabb.evaluations import CrossDatasetEvaluation
from moabb.paradigms import LeftRightImagery

set_log_level("WARNING")

paradigm = LeftRightImagery()

train_dataset = BNCI2014001()
test_dataset = Zhou2016()

pipelines = {
    "CSP+LDA": make_pipeline(Covariances("oas"), CSP(nfilter=6), LDA()),
}

evaluation = CrossDatasetEvaluation(
    paradigm=paradigm,
    train_datasets=train_dataset,
    test_datasets=test_dataset,
)

results = evaluation.process(pipelines)

print(results[["dataset", "subject", "session", "score"]])

fig, ax = plt.subplots(figsize=(8, 5))
results.boxplot(column="score", by="pipeline", ax=ax)
ax.set_title("Cross-dataset: BNCI2014_001 -> Zhou2016")
ax.set_ylabel("Score")
plt.suptitle("")
plt.tight_layout()
plt.show()
