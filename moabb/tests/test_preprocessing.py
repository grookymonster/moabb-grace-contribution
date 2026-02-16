"""Tests for FixedPipeline step manipulation methods."""

import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from moabb.datasets.bids_interface import StepType
from moabb.datasets.preprocessing import FixedPipeline


class _DummyTransformer(TransformerMixin, BaseEstimator):
    """Minimal transformer for testing."""

    def __init__(self, name="dummy"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


def _make_pipeline(*step_types):
    """Create a FixedPipeline from a sequence of StepType values."""
    steps = [
        (st, _DummyTransformer(name=f"{st.value}_{i}")) for i, st in enumerate(step_types)
    ]
    return FixedPipeline(steps)


# ── find_steps ────────────────────────────────────────────────────────────────


class TestFindSteps:
    def test_multiple_matches(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS, StepType.RAW)
        result = pipe.find_steps(StepType.RAW)
        assert len(result) == 2
        assert result[0][0] == 0
        assert result[1][0] == 2

    def test_single_match(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS, StepType.ARRAY)
        result = pipe.find_steps(StepType.EPOCHS)
        assert len(result) == 1
        assert result[0][0] == 1

    def test_no_matches(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS)
        result = pipe.find_steps(StepType.ARRAY)
        assert result == []

    def test_returns_correct_transformers(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS)
        result = pipe.find_steps(StepType.EPOCHS)
        assert result[0][1] is pipe.steps[1][1]


# ── insert_step ───────────────────────────────────────────────────────────────


class TestInsertStep:
    def test_insert_after(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS, StepType.ARRAY)
        new_t = _DummyTransformer("new")
        pipe.insert_step(StepType.EPOCHS, new_t, after=StepType.EPOCHS)
        assert len(pipe.steps) == 4
        assert pipe.steps[2] == (StepType.EPOCHS, new_t)

    def test_insert_after_uses_last_match(self):
        pipe = _make_pipeline(
            StepType.RAW, StepType.EPOCHS, StepType.EPOCHS, StepType.ARRAY
        )
        new_t = _DummyTransformer("new")
        pipe.insert_step(StepType.RAW, new_t, after=StepType.EPOCHS)
        assert pipe.steps[3] == (StepType.RAW, new_t)
        assert len(pipe.steps) == 5

    def test_insert_before(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS, StepType.ARRAY)
        new_t = _DummyTransformer("new")
        pipe.insert_step(StepType.RAW, new_t, before=StepType.EPOCHS)
        assert len(pipe.steps) == 4
        assert pipe.steps[1] == (StepType.RAW, new_t)

    def test_insert_before_uses_first_match(self):
        pipe = _make_pipeline(
            StepType.RAW, StepType.EPOCHS, StepType.EPOCHS, StepType.ARRAY
        )
        new_t = _DummyTransformer("new")
        pipe.insert_step(StepType.RAW, new_t, before=StepType.EPOCHS)
        assert pipe.steps[1] == (StepType.RAW, new_t)
        assert len(pipe.steps) == 5

    def test_insert_at_index(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS)
        new_t = _DummyTransformer("new")
        pipe.insert_step(StepType.ARRAY, new_t, index=0)
        assert pipe.steps[0] == (StepType.ARRAY, new_t)
        assert len(pipe.steps) == 3

    def test_insert_at_end_index(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS)
        new_t = _DummyTransformer("new")
        pipe.insert_step(StepType.ARRAY, new_t, index=len(pipe.steps))
        assert pipe.steps[-1] == (StepType.ARRAY, new_t)

    def test_error_after_not_found(self):
        pipe = _make_pipeline(StepType.RAW)
        with pytest.raises(ValueError, match="No steps of type"):
            pipe.insert_step(StepType.EPOCHS, _DummyTransformer(), after=StepType.EPOCHS)

    def test_error_before_not_found(self):
        pipe = _make_pipeline(StepType.RAW)
        with pytest.raises(ValueError, match="No steps of type"):
            pipe.insert_step(StepType.EPOCHS, _DummyTransformer(), before=StepType.ARRAY)

    def test_error_no_positioning_arg(self):
        pipe = _make_pipeline(StepType.RAW)
        with pytest.raises(ValueError, match="Exactly one"):
            pipe.insert_step(StepType.RAW, _DummyTransformer())

    def test_error_multiple_positioning_args(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS)
        with pytest.raises(ValueError, match="Exactly one"):
            pipe.insert_step(
                StepType.RAW,
                _DummyTransformer(),
                after=StepType.RAW,
                before=StepType.EPOCHS,
            )

    def test_chaining(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS)
        result = pipe.insert_step(StepType.ARRAY, _DummyTransformer(), index=1)
        assert result is pipe


# ── remove_step ───────────────────────────────────────────────────────────────


class TestRemoveStep:
    def test_remove_by_index(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS, StepType.ARRAY)
        pipe.remove_step(index=1)
        assert len(pipe.steps) == 2
        assert pipe.steps[0][0] == StepType.RAW
        assert pipe.steps[1][0] == StepType.ARRAY

    def test_remove_by_step_type_single(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS, StepType.ARRAY)
        pipe.remove_step(step_type=StepType.EPOCHS)
        assert len(pipe.steps) == 2
        assert all(st != StepType.EPOCHS for st, _ in pipe.steps)

    def test_remove_by_step_type_multiple(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS, StepType.RAW, StepType.ARRAY)
        pipe.remove_step(step_type=StepType.RAW)
        assert len(pipe.steps) == 2
        assert all(st != StepType.RAW for st, _ in pipe.steps)

    def test_error_step_type_not_found(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS)
        with pytest.raises(ValueError, match="No steps of type"):
            pipe.remove_step(step_type=StepType.ARRAY)

    def test_error_would_empty_pipeline_by_type(self):
        pipe = _make_pipeline(StepType.RAW, StepType.RAW)
        with pytest.raises(ValueError, match="Cannot remove all steps"):
            pipe.remove_step(step_type=StepType.RAW)

    def test_error_would_empty_pipeline_by_index(self):
        pipe = _make_pipeline(StepType.RAW)
        with pytest.raises(ValueError, match="Cannot remove all steps"):
            pipe.remove_step(index=0)

    def test_error_no_arg(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS)
        with pytest.raises(ValueError, match="Exactly one"):
            pipe.remove_step()

    def test_error_multiple_args(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS)
        with pytest.raises(ValueError, match="Exactly one"):
            pipe.remove_step(index=0, step_type=StepType.RAW)

    def test_chaining(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS, StepType.ARRAY)
        result = pipe.remove_step(index=1)
        assert result is pipe

    def test_chaining_combined(self):
        pipe = _make_pipeline(StepType.RAW, StepType.EPOCHS, StepType.ARRAY)
        new_t = _DummyTransformer("inserted")
        pipe.remove_step(index=1).insert_step(StepType.EPOCHS, new_t, index=1)
        assert len(pipe.steps) == 3
        assert pipe.steps[1] == (StepType.EPOCHS, new_t)
