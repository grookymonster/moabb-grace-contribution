"""Tests for moabb.analysis.chance_level module."""

import pytest

from moabb.analysis.chance_level import (
    adjusted_chance_level,
    get_chance_levels,
    theoretical_chance_level,
)


class TestTheoreticalChanceLevel:
    def test_binary(self):
        assert theoretical_chance_level(2) == 0.5

    def test_four_class(self):
        assert theoretical_chance_level(4) == 0.25

    def test_three_class(self):
        assert abs(theoretical_chance_level(3) - 1 / 3) < 1e-10

    def test_invalid_n_classes(self):
        with pytest.raises(ValueError):
            theoretical_chance_level(1)
        with pytest.raises(ValueError):
            theoretical_chance_level(0)


class TestAdjustedChanceLevel:
    def test_exceeds_theoretical(self):
        # Adjusted threshold should be higher than theoretical chance
        assert adjusted_chance_level(2, 20, 0.05) > 0.5

    def test_exceeds_theoretical_four_class(self):
        assert adjusted_chance_level(4, 50, 0.05) > 0.25

    def test_approaches_theoretical_large_n(self):
        # With many trials, adjusted level should approach theoretical
        adj = adjusted_chance_level(2, 10000, 0.05)
        assert abs(adj - 0.5) < 0.02

    def test_stricter_alpha_higher_threshold(self):
        # Stricter significance should require higher accuracy
        adj_05 = adjusted_chance_level(2, 50, 0.05)
        adj_01 = adjusted_chance_level(2, 50, 0.01)
        assert adj_01 > adj_05

    def test_fewer_trials_higher_threshold(self):
        # Fewer trials should require higher accuracy
        adj_small = adjusted_chance_level(2, 20, 0.05)
        adj_large = adjusted_chance_level(2, 200, 0.05)
        assert adj_small > adj_large

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            adjusted_chance_level(1, 100, 0.05)
        with pytest.raises(ValueError):
            adjusted_chance_level(2, 0, 0.05)
        with pytest.raises(ValueError):
            adjusted_chance_level(2, 100, 0.0)
        with pytest.raises(ValueError):
            adjusted_chance_level(2, 100, 1.0)


class TestGetChanceLevels:
    def _make_mock_dataset(self, name, event_id, summary_table=None, paradigm="imagery"):
        """Create a minimal mock dataset object."""

        class MockDataset:
            pass

        ds = MockDataset()
        ds.__class__ = type(name, (), {})
        ds.__class__.__name__ = name
        ds.event_id = event_id
        ds.paradigm = paradigm
        if summary_table is not None:
            ds._summary_table = summary_table
        return ds

    def test_theoretical_only(self):
        ds = self._make_mock_dataset(
            "BinaryDS", {"left": 1, "right": 2}
        )
        levels = get_chance_levels([ds])
        assert "BinaryDS" in levels
        assert levels["BinaryDS"]["theoretical"] == 0.5
        assert "adjusted" not in levels["BinaryDS"]

    def test_theoretical_four_class(self):
        ds = self._make_mock_dataset(
            "FourClassDS",
            {"a": 1, "b": 2, "c": 3, "d": 4},
        )
        levels = get_chance_levels([ds])
        assert levels["FourClassDS"]["theoretical"] == 0.25

    def test_multiple_datasets(self):
        ds1 = self._make_mock_dataset("DS1", {"a": 1, "b": 2})
        ds2 = self._make_mock_dataset("DS2", {"a": 1, "b": 2, "c": 3, "d": 4})
        levels = get_chance_levels([ds1, ds2])
        assert levels["DS1"]["theoretical"] == 0.5
        assert levels["DS2"]["theoretical"] == 0.25

    def test_with_alpha_and_summary_table(self):
        summary = {"#Trials / class": "50", "#Classes": "2"}
        ds = self._make_mock_dataset(
            "TestDS",
            {"a": 1, "b": 2},
            summary_table=summary,
            paradigm="imagery",
        )
        levels = get_chance_levels([ds], alpha=0.05)
        assert "adjusted" in levels["TestDS"]
        assert 0.05 in levels["TestDS"]["adjusted"]
        assert levels["TestDS"]["adjusted"][0.05] > 0.5

    def test_with_multiple_alphas(self):
        summary = {"#Trials / class": "50", "#Classes": "2"}
        ds = self._make_mock_dataset(
            "TestDS",
            {"a": 1, "b": 2},
            summary_table=summary,
            paradigm="imagery",
        )
        levels = get_chance_levels([ds], alpha=[0.05, 0.01, 0.001])
        adj = levels["TestDS"]["adjusted"]
        assert 0.05 in adj
        assert 0.01 in adj
        assert 0.001 in adj
        # Stricter alpha -> higher threshold
        assert adj[0.01] > adj[0.05]
        assert adj[0.001] > adj[0.01]

    def test_no_summary_table_warns(self):
        ds = self._make_mock_dataset("NoTable", {"a": 1, "b": 2})
        levels = get_chance_levels([ds], alpha=0.05)
        # Should still have theoretical, but no adjusted
        assert levels["NoTable"]["theoretical"] == 0.5
        assert "adjusted" not in levels["NoTable"]

    def test_invalid_alpha(self):
        ds = self._make_mock_dataset("DS", {"a": 1, "b": 2})
        with pytest.raises(ValueError):
            get_chance_levels([ds], alpha=0.0)
        with pytest.raises(ValueError):
            get_chance_levels([ds], alpha=1.0)
