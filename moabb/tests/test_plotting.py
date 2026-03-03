import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.pyplot import Figure


matplotlib.use("Agg")

from moabb.analysis.plotting import (
    _get_dataset_parameters,
    distribution_plot,
    paired_plot,
    score_plot,
)
from moabb.datasets.utils import dataset_list


@pytest.mark.parametrize(
    "dataset_class",
    [pytest.param(d, id=d.__name__) for d in dataset_list],
)
def test_get_dataset_parameters(dataset_class):
    if "Fake" in dataset_class.__name__:
        pytest.skip(
            f"Skipping test for {dataset_class.__name__} as it is a fake dataset."
        )
    dataset = dataset_class()
    dataset_name, paradigm, n_subjects, n_sessions, n_trials, trial_len = (
        _get_dataset_parameters(dataset)
    )
    assert isinstance(dataset_name, str)
    assert isinstance(paradigm, str)
    assert isinstance(n_subjects, int)
    assert isinstance(n_sessions, int)
    assert isinstance(n_trials, int)
    assert isinstance(trial_len, float)


def _make_results_df(n_subjects=5, n_pipelines=2, dataset_names=None, seed=42):
    """Create a synthetic results DataFrame mimicking Results.to_dataframe()."""
    rng = np.random.RandomState(seed)
    dataset_names = dataset_names or ["Dataset0", "Dataset1"]
    pipeline_names = [f"Pipeline{i}" for i in range(n_pipelines)]
    rows = []
    for ds in dataset_names:
        for pipe in pipeline_names:
            for subj in range(1, n_subjects + 1):
                for sess in ["0", "1"]:
                    rows.append(
                        {
                            "dataset": ds,
                            "pipeline": pipe,
                            "subject": subj,
                            "session": sess,
                            "score": rng.uniform(0.4, 1.0),
                            "time": rng.uniform(0.1, 2.0),
                            "n_samples": 100,
                            "n_channels": 10,
                            "n_samples_test": 50,
                            "n_classes": 2,
                        }
                    )
    return pd.DataFrame(rows)


def test_score_plot():
    data = _make_results_df()
    fig, color_dict = score_plot(data, chance_level=0.5)
    assert isinstance(fig, Figure)
    assert isinstance(color_dict, dict)


def test_distribution_plot():
    data = _make_results_df()
    fig, color_dict = distribution_plot(data, chance_level=0.25)
    assert isinstance(fig, Figure)
    assert isinstance(color_dict, dict)


def test_score_plot_auto_chance_level():
    data = _make_results_df()
    fig, color_dict = score_plot(data, chance_level="auto")
    assert isinstance(fig, Figure)
    assert isinstance(color_dict, dict)


def test_paired_plot():
    data = _make_results_df()
    fig = paired_plot(data, "Pipeline0", "Pipeline1", chance_level=0.5)
    assert isinstance(fig, Figure)
