import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.pyplot import Figure


matplotlib.use("Agg")

from moabb.analysis.plotting import (
    _get_dataset_parameters,
    _resolve_chance_levels,
    distribution_plot,
    paired_plot,
    score_plot,
)
from moabb.analysis.style import MOABB_PALETTE, apply_moabb_style
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


def _make_results_df(
    n_subjects=5,
    n_datasets=2,
    n_pipelines=2,
    dataset_names=None,
    pipeline_names=None,
    seed=42,
):
    """Create a synthetic results DataFrame mimicking Results.to_dataframe()."""
    rng = np.random.RandomState(seed)
    dataset_names = dataset_names or [f"Dataset{i}" for i in range(n_datasets)]
    pipeline_names = pipeline_names or [f"Pipeline{i}" for i in range(n_pipelines)]

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
                        }
                    )
    return pd.DataFrame(rows)


class TestResolveChanceLevels:
    def test_none_defaults_to_half(self):
        data = _make_results_df(dataset_names=["A", "B"])
        theoretical, adjusted = _resolve_chance_levels(data, None)
        assert theoretical == {"A": 0.5, "B": 0.5}
        assert adjusted is None

    def test_float_uniform(self):
        data = _make_results_df(dataset_names=["A", "B"])
        theoretical, adjusted = _resolve_chance_levels(data, 0.25)
        assert theoretical == {"A": 0.25, "B": 0.25}
        assert adjusted is None

    def test_dict_simple(self):
        data = _make_results_df(dataset_names=["A", "B"])
        theoretical, adjusted = _resolve_chance_levels(data, {"A": 0.5, "B": 0.25})
        assert theoretical == {"A": 0.5, "B": 0.25}
        assert adjusted is None

    def test_dict_from_get_chance_levels(self):
        data = _make_results_df(dataset_names=["A", "B"])
        chance_dict = {
            "A": {"theoretical": 0.5, "adjusted": {0.05: 0.58, 0.01: 0.62}},
            "B": {"theoretical": 0.25, "adjusted": {0.05: 0.32}},
        }
        theoretical, adjusted = _resolve_chance_levels(data, chance_dict)
        assert theoretical["A"] == 0.5
        assert theoretical["B"] == 0.25
        assert adjusted is not None
        assert adjusted["A"][0.05] == 0.58
        assert adjusted["B"][0.05] == 0.32

    def test_missing_dataset_defaults_to_half(self):
        data = _make_results_df(dataset_names=["A", "B"])
        theoretical, _ = _resolve_chance_levels(data, {"A": 0.3})
        assert theoretical["A"] == 0.3
        assert theoretical["B"] == 0.5

    def test_invalid_type_raises(self):
        data = _make_results_df()
        with pytest.raises(TypeError):
            _resolve_chance_levels(data, "invalid")


class TestScorePlot:
    def test_default_returns_figure(self):
        data = _make_results_df()
        fig, color_dict = score_plot(data)
        assert isinstance(fig, Figure)
        assert isinstance(color_dict, dict)

    def test_backward_compat_no_chance_level(self):
        data = _make_results_df()
        fig, color_dict = score_plot(data)
        assert isinstance(fig, Figure)

    def test_with_float_chance_level(self):
        data = _make_results_df()
        fig, color_dict = score_plot(data, chance_level=0.25)
        assert isinstance(fig, Figure)

    def test_with_dict_chance_level(self):
        data = _make_results_df(dataset_names=["DS1", "DS2"])
        fig, color_dict = score_plot(data, chance_level={"DS1": 0.5, "DS2": 0.25})
        assert isinstance(fig, Figure)

    def test_with_get_chance_levels_dict(self):
        data = _make_results_df(dataset_names=["DS1", "DS2"])
        chance_dict = {
            "DS1": {
                "theoretical": 0.5,
                "adjusted": {0.05: 0.58, 0.01: 0.62},
            },
            "DS2": {
                "theoretical": 0.25,
                "adjusted": {0.05: 0.32},
            },
        }
        fig, color_dict = score_plot(data, chance_level=chance_dict)
        assert isinstance(fig, Figure)

    def test_horizontal_orientation(self):
        data = _make_results_df()
        fig, _ = score_plot(data, orientation="horizontal", chance_level=0.25)
        assert isinstance(fig, Figure)

    def test_with_pipelines_filter(self):
        data = _make_results_df(pipeline_names=["A", "B", "C"])
        fig, color_dict = score_plot(data, pipelines=["A", "B"])
        assert isinstance(fig, Figure)
        assert "C" not in color_dict


class TestDistributionPlot:
    def test_returns_figure(self):
        data = _make_results_df()
        fig, color_dict = distribution_plot(data)
        assert isinstance(fig, Figure)
        assert isinstance(color_dict, dict)

    def test_with_chance_level(self):
        data = _make_results_df()
        fig, color_dict = distribution_plot(data, chance_level=0.25)
        assert isinstance(fig, Figure)

    def test_horizontal(self):
        data = _make_results_df()
        fig, _ = distribution_plot(data, orientation="h")
        assert isinstance(fig, Figure)

    def test_with_dict_chance_level(self):
        data = _make_results_df(dataset_names=["DS1", "DS2"])
        chance_dict = {
            "DS1": {"theoretical": 0.5, "adjusted": {0.05: 0.58}},
            "DS2": {"theoretical": 0.25},
        }
        fig, _ = distribution_plot(data, chance_level=chance_dict)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self):
        data = _make_results_df()
        fig, _ = distribution_plot(data, figsize=(14, 10))
        assert isinstance(fig, Figure)

    def test_with_pipelines_filter(self):
        data = _make_results_df(pipeline_names=["A", "B", "C"])
        fig, color_dict = distribution_plot(data, pipelines=["A"])
        assert isinstance(fig, Figure)


class TestPairedPlot:
    def test_default(self):
        data = _make_results_df(pipeline_names=["Alg1", "Alg2"])
        fig = paired_plot(data, "Alg1", "Alg2")
        assert isinstance(fig, Figure)

    def test_with_chance_level_float(self):
        data = _make_results_df(pipeline_names=["Alg1", "Alg2"])
        fig = paired_plot(data, "Alg1", "Alg2", chance_level=0.25)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # Scores displayed as percentages: 0.25 -> 25
        assert ax.get_xlim()[0] == pytest.approx(25, abs=5)
        assert ax.get_ylim()[0] == pytest.approx(25, abs=5)

    def test_with_chance_level_dict(self):
        data = _make_results_df(
            pipeline_names=["Alg1", "Alg2"],
            dataset_names=["DS1", "DS2"],
        )
        fig = paired_plot(data, "Alg1", "Alg2", chance_level={"DS1": 0.5, "DS2": 0.25})
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # min chance level is 0.25 -> 25%
        assert ax.get_xlim()[0] == pytest.approx(25, abs=5)

    def test_crosshair_lines_present(self):
        data = _make_results_df(pipeline_names=["Alg1", "Alg2"])
        fig = paired_plot(data, "Alg1", "Alg2", chance_level=0.25)
        ax = fig.axes[0]
        # At least 3 Line2D objects: diagonal + horizontal + vertical crosshair
        from matplotlib.lines import Line2D

        lines = [c for c in ax.get_children() if isinstance(c, Line2D)]
        assert len(lines) >= 3

    def test_shaded_band_with_adjusted_levels(self):
        data = _make_results_df(
            pipeline_names=["Alg1", "Alg2"],
            dataset_names=["DS1", "DS2"],
        )
        chance = {
            "DS1": {"theoretical": 0.5, "adjusted": {0.05: 0.6}},
            "DS2": {"theoretical": 0.25, "adjusted": {0.05: 0.35}},
        }
        fig = paired_plot(data, "Alg1", "Alg2", chance_level=chance)
        ax = fig.axes[0]
        from matplotlib.patches import Patch, Rectangle

        patches = [
            c
            for c in ax.get_children()
            if isinstance(c, (Patch, Rectangle))
            and c.get_alpha() is not None
            and c.get_alpha() < 0.15
        ]
        # Should have at least 2 shaded patches (axhspan + axvspan)
        assert len(patches) >= 2

    def test_no_band_without_adjusted_levels(self):
        data = _make_results_df(pipeline_names=["Alg1", "Alg2"])
        fig = paired_plot(data, "Alg1", "Alg2", chance_level=0.25)
        ax = fig.axes[0]
        from matplotlib.patches import Patch, Rectangle

        patches = [
            c
            for c in ax.get_children()
            if isinstance(c, (Patch, Rectangle))
            and c.get_alpha() is not None
            and c.get_alpha() < 0.15
        ]
        assert len(patches) == 0


class TestMoabbStyle:
    def test_moabb_palette_length(self):
        assert len(MOABB_PALETTE) == 6

    def test_score_plot_has_accent_line(self):
        data = _make_results_df()
        fig, _ = score_plot(data)
        assert len(fig.patches) > 0

    def test_score_plot_spines(self):
        data = _make_results_df()
        fig, _ = score_plot(data)
        ax = fig.axes[0]
        assert ax.spines["bottom"].get_visible()
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert not ax.spines["left"].get_visible()

    def test_apply_moabb_style_title(self):
        fig, ax = matplotlib.pyplot.subplots()
        apply_moabb_style(ax, title="Test Title", subtitle="Test Sub")
        texts = [t.get_text() for t in fig.texts]
        assert "Test Title" in texts
        assert "Test Sub" in texts
        matplotlib.pyplot.close(fig)
