"""Tests for moabb.analysis.chance_level module."""

import pandas as pd
import pytest

from moabb.analysis.chance_level import (
    adjusted_chance_level,
    chance_levels_from_dataframe,
    get_chance_levels,
    theoretical_chance_level,
)


def test_theoretical_chance_level():
    assert theoretical_chance_level(2) == 0.5
    assert theoretical_chance_level(4) == 0.25
    with pytest.raises(ValueError):
        theoretical_chance_level(1)


def test_adjusted_chance_level():
    # Adjusted threshold should exceed theoretical
    assert adjusted_chance_level(2, 20, 0.05) > 0.5
    # Approaches theoretical with many trials
    assert abs(adjusted_chance_level(2, 10000, 0.05) - 0.5) < 0.02
    # Stricter alpha -> higher threshold
    assert adjusted_chance_level(2, 50, 0.01) > adjusted_chance_level(2, 50, 0.05)
    with pytest.raises(ValueError):
        adjusted_chance_level(1, 100, 0.05)


def test_get_chance_levels():
    class MockDataset:
        pass

    ds = MockDataset()
    ds.__class__ = type("BinaryDS", (), {})
    ds.code = "BinaryDS"
    ds.event_id = {"left": 1, "right": 2}
    ds.paradigm = "imagery"
    ds._summary_table = {"#Trials / class": "50", "#Classes": "2"}

    levels = get_chance_levels([ds], alpha=[0.05, 0.01])
    assert levels["BinaryDS"]["theoretical"] == 0.5
    assert levels["BinaryDS"]["adjusted"][0.05] > 0.5
    assert levels["BinaryDS"]["adjusted"][0.01] > levels["BinaryDS"]["adjusted"][0.05]


def test_chance_levels_from_dataframe():
    data = pd.DataFrame(
        {
            "dataset": ["DS_A", "DS_A", "DS_B", "DS_B"],
            "n_samples_test": [50, 50, 100, 100],
            "n_classes": [2, 2, 4, 4],
        }
    )
    levels = chance_levels_from_dataframe(data, alpha=0.05)
    assert levels["DS_A"]["theoretical"] == 0.5
    assert levels["DS_A"]["adjusted"][0.05] > 0.5
    assert levels["DS_B"]["theoretical"] == 0.25
