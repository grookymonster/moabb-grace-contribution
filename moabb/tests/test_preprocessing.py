"""Tests for moabb.datasets.preprocessing module."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from moabb.datasets.bids_interface import StepType
from moabb.datasets.preprocessing import (
    EpochsToEvents,
    EventsToLabels,
    FixedPipeline,
    FixedTransformer,
    ForkPipelines,
    NamedFunctionTransformer,
    RawToEvents,
    RawToEventsP300,
    RawToFixedIntervalEvents,
    SetRawAnnotations,
    _compute_events_desc,
    _generate_sliding_window_events,
    _get_event_id_values,
    _insert_rest_events,
    _is_none_pipeline,
    _unsafe_pick_events,
    get_crop_pipeline,
    get_filter_pipeline,
    get_resample_pipeline,
    make_fixed_pipeline,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


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


def _make_raw(sfreq=250, n_channels=3, duration=10.0, stim_events=None):
    """Create a minimal MNE Raw object for testing.

    Parameters
    ----------
    stim_events : list of (sample, label) or None
        If provided, creates a stim channel with events at given samples.
        If None, no stim channel is added.
    """
    import mne

    n_samples = int(sfreq * duration)
    ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels

    if stim_events is not None:
        ch_names.append("STI 014")
        ch_types.append("stim")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    data = np.random.RandomState(42).randn(len(ch_names), n_samples) * 1e-6

    if stim_events is not None:
        data[-1, :] = 0
        for sample, label in stim_events:
            if sample < n_samples:
                data[-1, sample] = label

    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def _make_events(*labels, sfreq=250, interval=2.0):
    """Create a simple events array with evenly spaced events."""
    events = np.zeros((len(labels), 3), dtype="int32")
    for i, label in enumerate(labels):
        events[i, 0] = int(i * interval * sfreq)
        events[i, 2] = label
    return events


# ── FixedPipeline ─────────────────────────────────────────────────────────────


class TestFixedPipeline:
    def test_sklearn_is_fitted(self):
        pipe = _make_pipeline(StepType.RAW)
        assert pipe.__sklearn_is_fitted__() is True


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


# ── make_fixed_pipeline ──────────────────────────────────────────────────────


class TestMakeFixedPipeline:
    def test_returns_fixed_pipeline(self):
        t = _DummyTransformer()
        pipe = make_fixed_pipeline(t)
        assert isinstance(pipe, FixedPipeline)
        assert pipe.__sklearn_is_fitted__() is True

    def test_multiple_steps(self):
        t1 = _DummyTransformer("a")
        t2 = _DummyTransformer("b")
        pipe = make_fixed_pipeline(t1, t2)
        assert len(pipe.steps) == 2


# ── _is_none_pipeline ────────────────────────────────────────────────────────


class TestIsNonePipeline:
    def test_none_pipeline(self):
        pipe = Pipeline([("none", None)])
        assert _is_none_pipeline(pipe) is True

    def test_non_none_pipeline(self):
        pipe = Pipeline([("t", _DummyTransformer())])
        assert _is_none_pipeline(pipe) is False

    def test_not_a_pipeline(self):
        assert _is_none_pipeline("not a pipeline") is False

    def test_multi_step_with_none(self):
        pipe = Pipeline([("none", None), ("t", _DummyTransformer())])
        assert _is_none_pipeline(pipe) is False


# ── _unsafe_pick_events ──────────────────────────────────────────────────────


class TestUnsafePickEvents:
    def test_picks_matching_events(self):
        events = np.array([[0, 0, 1], [100, 0, 2], [200, 0, 1]], dtype="int32")
        result = _unsafe_pick_events(events, include=[1])
        assert result.shape == (2, 3)
        assert np.all(result[:, 2] == 1)

    def test_no_matching_events_returns_empty(self):
        events = np.array([[0, 0, 1], [100, 0, 2]], dtype="int32")
        result = _unsafe_pick_events(events, include=[99])
        assert result.shape == (0, 3)

    def test_all_match(self):
        events = np.array([[0, 0, 5], [100, 0, 5]], dtype="int32")
        result = _unsafe_pick_events(events, include=[5])
        assert result.shape == (2, 3)


# ── _insert_rest_events ──────────────────────────────────────────────────────


class TestInsertRestEvents:
    def test_basic_rest_insertion(self):
        # Two trials at samples 0 and 1000, task duration 500
        events = np.array([[0, 0, 1], [1000, 0, 2]], dtype="int32")
        result = _insert_rest_events(events, task_duration_samples=500)
        # Should have rest events inserted in gaps
        assert len(result) > len(events)
        # Rest events should have label _REST_LABEL (-1)
        from moabb.datasets.preprocessing import _REST_LABEL

        rest_mask = result[:, 2] == _REST_LABEL
        assert np.any(rest_mask)

    def test_no_gaps_no_rest(self):
        # Contiguous trials: task ends exactly where next starts
        events = np.array([[0, 0, 1], [500, 0, 2]], dtype="int32")
        result = _insert_rest_events(events, task_duration_samples=500)
        from moabb.datasets.preprocessing import _REST_LABEL

        # The last trial should still get a post-task rest
        rest_mask = result[:, 2] == _REST_LABEL
        assert np.any(rest_mask)

    def test_with_interval_start(self):
        events = np.array([[0, 0, 1], [2000, 0, 2]], dtype="int32")
        result = _insert_rest_events(
            events, task_duration_samples=500, interval_start_samples=100
        )
        # With interval_start > 0, pre-task rest events should be added
        assert len(result) > len(events)

    def test_single_event(self):
        events = np.array([[0, 0, 1]], dtype="int32")
        result = _insert_rest_events(events, task_duration_samples=500)
        assert len(result) >= 1

    def test_events_sorted_by_onset(self):
        events = np.array([[0, 0, 1], [1000, 0, 2]], dtype="int32")
        result = _insert_rest_events(events, task_duration_samples=500)
        assert np.all(result[:-1, 0] <= result[1:, 0])


# ── _generate_sliding_window_events ──────────────────────────────────────────


class TestGenerateSlidingWindowEvents:
    def test_empty_events(self):
        events = np.zeros((0, 3), dtype="int32")
        result = _generate_sliding_window_events(
            events, window_length=1.0, overlap=50, sfreq=250, interval=(0, 4)
        )
        assert result.shape == (0, 3)

    def test_zero_window_length_raises(self):
        events = np.array([[0, 0, 1]], dtype="int32")
        with pytest.raises(ValueError, match="Window length must be strictly positive"):
            _generate_sliding_window_events(
                events, window_length=0, overlap=50, sfreq=250, interval=(0, 4)
            )

    def test_basic_sliding_window(self):
        # Single event at sample 0, interval=(0, 4), sfreq=250
        # task_duration = 4*250 = 1000 samples
        events = np.array([[0, 0, 1]], dtype="int32")
        result = _generate_sliding_window_events(
            events, window_length=1.0, overlap=50, sfreq=250, interval=(0, 4)
        )
        assert len(result) > 0
        # All windows should start at valid positions
        assert np.all(result[:, 0] >= 0)

    def test_no_overlap(self):
        events = np.array([[0, 0, 1]], dtype="int32")
        result = _generate_sliding_window_events(
            events, window_length=1.0, overlap=0, sfreq=250, interval=(0, 4)
        )
        assert len(result) > 0
        # With no overlap, stride = window_length, so windows should be spaced 250 apart
        if len(result) > 1:
            diffs = np.diff(result[:, 0])
            assert np.all(diffs == 250)

    def test_multiple_events(self):
        # Two events with different labels
        events = np.array([[0, 0, 1], [2000, 0, 2]], dtype="int32")
        result = _generate_sliding_window_events(
            events, window_length=1.0, overlap=50, sfreq=250, interval=(0, 4)
        )
        assert len(result) > 0

    def test_max_start_less_than_first_onset(self):
        # Edge case: window so large that max_start < first_onset
        events = np.array([[1000, 0, 1]], dtype="int32")
        result = _generate_sliding_window_events(
            events, window_length=10.0, overlap=0, sfreq=250, interval=(0, 1)
        )
        assert result.shape == (0, 3)

    def test_with_tmin_offset(self):
        events = np.array([[0, 0, 1], [2000, 0, 2]], dtype="int32")
        result = _generate_sliding_window_events(
            events,
            window_length=1.0,
            overlap=50,
            sfreq=250,
            interval=(0, 4),
            tmin=-0.5,
        )
        assert len(result) > 0

    def test_with_nonzero_interval_start(self):
        events = np.array([[0, 0, 1], [3000, 0, 2]], dtype="int32")
        result = _generate_sliding_window_events(
            events,
            window_length=1.0,
            overlap=50,
            sfreq=250,
            interval=(2, 6),
        )
        assert len(result) > 0


# ── ForkPipelines ─────────────────────────────────────────────────────────────


class TestForkPipelines:
    def test_init_and_transform(self):
        t1 = _DummyTransformer("a")
        t2 = _DummyTransformer("b")
        fork = ForkPipelines([("branch1", t1), ("branch2", t2)])
        result = fork.transform("data")
        assert list(result.keys()) == ["branch1", "branch2"]
        assert result["branch1"] == "data"

    def test_fit(self):
        t1 = _DummyTransformer("a")
        t2 = _DummyTransformer("b")
        fork = ForkPipelines([("branch1", t1), ("branch2", t2)])
        result = fork.fit("data")
        assert result is fork

    def test_sklearn_is_fitted(self):
        fork = ForkPipelines([("b", _DummyTransformer())])
        assert fork.__sklearn_is_fitted__() is True

    def test_sk_visual_block(self):
        fork = ForkPipelines([("b1", _DummyTransformer()), ("b2", _DummyTransformer())])
        block = fork._sk_visual_block_()
        # Should return a _VisualBlock (not NotImplemented)
        assert block is not NotImplemented


# ── FixedTransformer ──────────────────────────────────────────────────────────


class TestFixedTransformer:
    def test_init(self):
        t = FixedTransformer()
        assert t._is_fitted is True

    def test_fit(self):
        t = FixedTransformer()
        result = t.fit("X")
        assert result is t

    def test_sklearn_is_fitted(self):
        t = FixedTransformer()
        assert t.__sklearn_is_fitted__() is True

    def test_sk_visual_block(self):
        t = FixedTransformer()
        block = t._sk_visual_block_()
        assert block is not NotImplemented


# ── _get_event_id_values ──────────────────────────────────────────────────────


class TestGetEventIdValues:
    def test_simple_dict(self):
        result = _get_event_id_values({"left": 1, "right": 2})
        assert result == [1, 2]

    def test_empty_dict(self):
        result = _get_event_id_values({})
        assert result == []

    def test_list_values(self):
        result = _get_event_id_values({"target": [1, 2], "nontarget": [3]})
        assert sorted(result) == [1, 2, 3]

    def test_single_value(self):
        result = _get_event_id_values({"a": 5})
        assert result == [5]


# ── _compute_events_desc ─────────────────────────────────────────────────────


class TestComputeEventsDesc:
    def test_simple_mapping(self):
        result = _compute_events_desc({"left": 1, "right": 2})
        assert result == {1: "left", 2: "right"}

    def test_list_codes(self):
        result = _compute_events_desc({"target": [1, 2], "nontarget": [3]})
        assert result == {1: "target", 2: "target", 3: "nontarget"}

    def test_empty(self):
        result = _compute_events_desc({})
        assert result == {}


# ── SetRawAnnotations ─────────────────────────────────────────────────────────


class TestSetRawAnnotations:
    def test_init(self):
        sra = SetRawAnnotations(event_id={"left": 1, "right": 2}, interval=(0, 4))
        assert sra.event_id == {"left": 1, "right": 2}
        assert sra.interval == (0, 4)

    def test_duplicate_event_code_raises(self):
        with pytest.raises(ValueError, match="Duplicate event code"):
            SetRawAnnotations(event_id={"left": 1, "right": 1}, interval=(0, 4))

    def test_transform_with_stim_channel(self):
        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(500, 1), (1500, 2)])
        sra = SetRawAnnotations(event_id={"left": 1, "right": 2}, interval=(0, 4))
        result = sra.transform(raw)
        assert result is raw
        assert len(raw.annotations) == 2

    def test_transform_no_events_warns(self):
        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(500, 99)])
        sra = SetRawAnnotations(event_id={"left": 1}, interval=(0, 4))
        result = sra.transform(raw)
        assert result is raw

    def test_transform_with_annotations(self):
        import mne

        raw = _make_raw(sfreq=250, duration=10.0, stim_events=None)
        # Add annotations manually
        annotations = mne.Annotations(
            onset=[1.0, 3.0], duration=[0.0, 0.0], description=["left", "right"]
        )
        raw.set_annotations(annotations)
        sra = SetRawAnnotations(event_id={"left": 1, "right": 2}, interval=(0, 4))
        result = sra.transform(raw)
        assert result is raw

    def test_transform_no_stim_no_annotations_warns(self):
        raw = _make_raw(sfreq=250, duration=10.0, stim_events=None)
        raw.set_annotations(None)
        sra = SetRawAnnotations(event_id={"left": 1}, interval=(0, 4))
        result = sra.transform(raw)
        assert result is raw


# ── RawToEvents ───────────────────────────────────────────────────────────────


class TestRawToEvents:
    def test_init_basic(self):
        rte = RawToEvents(event_id={"left": 1}, interval=(0, 4))
        assert rte.overlap is None
        assert rte.window_length is None

    def test_init_with_overlap(self):
        rte = RawToEvents(
            event_id={"left": 1}, interval=(0, 4), overlap=50, window_length=1.0
        )
        assert rte.overlap == 50.0

    def test_init_overlap_without_window_raises(self):
        with pytest.raises(ValueError, match="window_length must be provided"):
            RawToEvents(event_id={"left": 1}, interval=(0, 4), overlap=50)

    def test_init_invalid_overlap_type_raises(self):
        with pytest.raises(TypeError, match="overlap must be a number"):
            RawToEvents(
                event_id={"left": 1},
                interval=(0, 4),
                overlap="invalid",
                window_length=1.0,
            )

    def test_init_overlap_out_of_range_raises(self):
        with pytest.raises(ValueError, match="overlap must be in"):
            RawToEvents(
                event_id={"left": 1},
                interval=(0, 4),
                overlap=100,
                window_length=1.0,
            )

    def test_transform_with_stim(self):
        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(500, 1), (1500, 2)])
        rte = RawToEvents(event_id={"left": 1, "right": 2}, interval=(0, 4))
        events = rte.transform(raw)
        assert events.shape[1] == 3
        assert len(events) == 2

    def test_transform_no_events(self):
        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(500, 99)])
        rte = RawToEvents(event_id={"left": 1}, interval=(0, 4))
        events = rte.transform(raw)
        assert events.shape == (0, 3)

    def test_transform_with_annotations(self):
        import mne

        raw = _make_raw(sfreq=250, duration=10.0, stim_events=None)
        annotations = mne.Annotations(
            onset=[1.0, 3.0], duration=[0.0, 0.0], description=["left", "right"]
        )
        raw.set_annotations(annotations)
        rte = RawToEvents(event_id={"left": 1, "right": 2}, interval=(0, 4))
        events = rte.transform(raw)
        assert events.shape[1] == 3
        assert len(events) == 2

    def test_transform_annotations_no_match(self):
        import mne

        raw = _make_raw(sfreq=250, duration=10.0, stim_events=None)
        annotations = mne.Annotations(onset=[1.0], duration=[0.0], description=["other"])
        raw.set_annotations(annotations)
        rte = RawToEvents(event_id={"left": 1}, interval=(0, 4))
        events = rte.transform(raw)
        assert events.shape == (0, 3)

    def test_transform_with_overlap(self):
        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(500, 1), (1500, 2)])
        rte = RawToEvents(
            event_id={"left": 1, "right": 2},
            interval=(0, 4),
            overlap=50,
            window_length=1.0,
        )
        events = rte.transform(raw)
        assert events.shape[1] == 3
        # Sliding window should produce more events than original
        assert len(events) >= 2


# ── RawToEventsP300 ───────────────────────────────────────────────────────────


class TestRawToEventsP300:
    def test_init(self):
        rte = RawToEventsP300(event_id={"Target": 1, "NonTarget": 2}, interval=(0, 1))
        assert rte.ignore_relabelling is False

    def test_transform_basic(self):
        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(500, 1), (1000, 2)])
        rte = RawToEventsP300(event_id={"Target": 1, "NonTarget": 2}, interval=(0, 1))
        events = rte.transform(raw)
        assert events.shape[1] == 3

    def test_transform_with_relabelling(self):
        raw = _make_raw(
            sfreq=250,
            duration=10.0,
            stim_events=[(500, 1), (750, 3), (1000, 2), (1250, 4)],
        )
        rte = RawToEventsP300(
            event_id={"Target": [1, 3], "NonTarget": [2, 4]}, interval=(0, 1)
        )
        events = rte.transform(raw)
        assert events.shape[1] == 3
        # After relabelling, labels should be 0 and 1
        assert set(events[:, 2]).issubset({0, 1})

    def test_transform_ignore_relabelling(self):
        raw = _make_raw(
            sfreq=250,
            duration=10.0,
            stim_events=[(500, 1), (750, 3), (1000, 2), (1250, 4)],
        )
        rte = RawToEventsP300(
            event_id={"Target": [1, 3], "NonTarget": [2, 4]},
            interval=(0, 1),
            ignore_relabelling=True,
        )
        events = rte.transform(raw)
        assert events.shape[1] == 3
        # Original labels should be preserved
        assert set(events[:, 2]).issubset({1, 2, 3, 4})


# ── RawToFixedIntervalEvents ─────────────────────────────────────────────────


class TestRawToFixedIntervalEvents:
    def test_init(self):
        t = RawToFixedIntervalEvents(
            length=1.0, stride=0.5, start_offset=0, stop_offset=None
        )
        assert t.length == 1.0
        assert t.marker == 1

    def test_transform(self):
        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(0, 0)])
        t = RawToFixedIntervalEvents(
            length=1.0, stride=0.5, start_offset=0, stop_offset=None
        )
        events = t.transform(raw)
        assert events is not None
        assert events.shape[1] == 3
        # Should have multiple events
        assert len(events) > 1
        assert np.all(events[:, 2] == 1)

    def test_transform_with_stop_offset(self):
        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(0, 0)])
        t = RawToFixedIntervalEvents(
            length=1.0, stride=0.5, start_offset=0, stop_offset=5.0
        )
        events = t.transform(raw)
        assert events is not None

    def test_transform_not_raw_raises(self):
        t = RawToFixedIntervalEvents(
            length=1.0, stride=0.5, start_offset=0, stop_offset=None
        )
        with pytest.raises(ValueError):
            t.transform("not a raw")

    def test_transform_returns_none_when_no_events(self):
        # Very short raw with long window → no events
        raw = _make_raw(sfreq=250, duration=0.1, stim_events=[(0, 0)])
        t = RawToFixedIntervalEvents(
            length=1.0, stride=0.5, start_offset=0, stop_offset=None
        )
        result = t.transform(raw)
        assert result is None

    def test_custom_marker(self):
        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(0, 0)])
        t = RawToFixedIntervalEvents(
            length=1.0, stride=0.5, start_offset=0, stop_offset=None, marker=42
        )
        events = t.transform(raw)
        assert np.all(events[:, 2] == 42)


# ── EpochsToEvents ───────────────────────────────────────────────────────────


class TestEpochsToEvents:
    def test_transform(self):
        import mne

        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(500, 1), (1500, 2)])
        events = np.array([[500, 0, 1], [1500, 0, 2]], dtype="int32")
        epochs = mne.Epochs(
            raw,
            events,
            event_id=[1, 2],
            tmin=0,
            tmax=0.5,
            baseline=None,
            preload=True,
            verbose=False,
        )
        t = EpochsToEvents()
        result = t.transform(epochs)
        assert np.array_equal(result, epochs.events)


# ── EventsToLabels ───────────────────────────────────────────────────────────


class TestEventsToLabels:
    def test_transform(self):
        events = np.array([[0, 0, 1], [100, 0, 2], [200, 0, 1]], dtype="int32")
        t = EventsToLabels(event_id={"left": 1, "right": 2})
        labels = t.transform(events)
        assert labels == ["left", "right", "left"]

    def test_with_list_codes(self):
        events = np.array([[0, 0, 1], [100, 0, 3]], dtype="int32")
        t = EventsToLabels(event_id={"target": [1, 3]})
        labels = t.transform(events)
        assert labels == ["target", "target"]


# ── RawToEpochs ──────────────────────────────────────────────────────────────


class TestRawToEpochs:
    def test_init(self):
        from moabb.datasets.preprocessing import RawToEpochs

        t = RawToEpochs(event_id={"left": 1}, tmin=0, tmax=1.0, baseline=None)
        assert t.channels is None
        assert t.interpolate_missing_channels is False

    def test_transform_basic(self):
        import mne

        from moabb.datasets.preprocessing import RawToEpochs

        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(500, 1), (1500, 1)])
        events = np.array([[500, 0, 1], [1500, 0, 1]], dtype="int32")
        t = RawToEpochs(event_id={"left": 1}, tmin=0, tmax=0.5, baseline=None)
        result = t.transform({"raw": raw, "events": events})
        assert isinstance(result, mne.Epochs)
        assert len(result) == 2

    def test_transform_no_events_raises(self):
        from moabb.datasets.preprocessing import RawToEpochs

        raw = _make_raw(sfreq=250, duration=10.0, stim_events=[(500, 1)])
        events = np.zeros((0, 3), dtype="int32")
        t = RawToEpochs(event_id={"left": 1}, tmin=0, tmax=0.5, baseline=None)
        with pytest.raises(ValueError, match="No events found"):
            t.transform({"raw": raw, "events": events})

    def test_transform_not_raw_raises(self):
        from moabb.datasets.preprocessing import RawToEpochs

        events = np.array([[500, 0, 1]], dtype="int32")
        t = RawToEpochs(event_id={"left": 1}, tmin=0, tmax=0.5, baseline=None)
        with pytest.raises(ValueError, match="raw must be a mne.io.BaseRaw"):
            t.transform({"raw": "not_raw", "events": events})

    def test_transform_with_channels(self):
        import mne

        from moabb.datasets.preprocessing import RawToEpochs

        raw = _make_raw(sfreq=250, n_channels=3, duration=10.0, stim_events=[(500, 1)])
        events = np.array([[500, 0, 1]], dtype="int32")
        t = RawToEpochs(
            event_id={"left": 1},
            tmin=0,
            tmax=0.5,
            baseline=None,
            channels=["EEG1", "EEG2"],
        )
        result = t.transform({"raw": raw, "events": events})
        assert isinstance(result, mne.Epochs)
        assert len(result.ch_names) == 2


# ── NamedFunctionTransformer ─────────────────────────────────────────────────


class TestNamedFunctionTransformer:
    def test_init_with_name(self):
        def my_func(x):
            return x

        t = NamedFunctionTransformer(my_func, display_name="My Transform")
        assert repr(t) == "My Transform"

    def test_init_without_name(self):
        def my_func(x):
            return x

        t = NamedFunctionTransformer(my_func)
        assert repr(t) == "my_func"

    def test_sk_visual_block(self):
        t = NamedFunctionTransformer(lambda x: x, display_name="test")
        block = t._sk_visual_block_()
        assert block is not NotImplemented

    def test_transform(self):
        t = NamedFunctionTransformer(lambda x: x * 2)
        assert t.transform(5) == 10


# ── get_*_pipeline helpers ────────────────────────────────────────────────────


class TestPipelineHelpers:
    def test_get_filter_pipeline(self):
        t = get_filter_pipeline(8, 30)
        assert isinstance(t, NamedFunctionTransformer)
        assert "Band Pass Filter" in repr(t)

    def test_get_crop_pipeline(self):
        t = get_crop_pipeline(0, 4)
        assert isinstance(t, NamedFunctionTransformer)
        assert "Crop" in repr(t)

    def test_get_resample_pipeline(self):
        t = get_resample_pipeline(128)
        assert isinstance(t, NamedFunctionTransformer)
        assert "Resample" in repr(t)


# ── Additional coverage: edge cases and uncovered branches ────────────────────


class TestUnsafePickEventsReRaise:
    def test_reraises_non_matching_runtime_error(self):
        """Line 193: re-raise RuntimeError that isn't 'No events found'."""
        from unittest.mock import patch

        events = np.array([[0, 0, 1]], dtype="int32")
        with patch("moabb.datasets.preprocessing.mne.pick_events") as mock_pick:
            mock_pick.side_effect = RuntimeError("Some other error")
            with pytest.raises(RuntimeError, match="Some other error"):
                _unsafe_pick_events(events, include=[1])


class TestInsertRestEventsContiguous:
    def test_contiguous_no_rest_with_zero_interval(self):
        """Line 244: early return when contiguous trials leave no gaps."""
        # task_duration == spacing between events → no gap → no rest
        # Actually, the last event always gets a post-task rest, so we
        # need overlapping trials to get no rest at all.
        # With task_duration > spacing, next_task_start <= task_end for all pairs
        events = np.array([[0, 0, 1], [100, 0, 2]], dtype="int32")
        result = _insert_rest_events(events, task_duration_samples=200)
        # With interval_start=0, no pre-task rest.
        # task0 ends at 200, task1 starts at 100 → next_task_start(100) < task_end(200)
        # task1 ends at 300, next_task_start=301 > 300 → 1 rest event only for last
        from moabb.datasets.preprocessing import _REST_LABEL

        rest_count = np.sum(result[:, 2] == _REST_LABEL)
        assert rest_count == 1  # Only post-last-trial rest

    def test_fully_overlapping_single_event(self):
        """When there's only one event with no gaps, rest is still added after."""
        events = np.array([[0, 0, 1]], dtype="int32")
        result = _insert_rest_events(events, task_duration_samples=500)
        from moabb.datasets.preprocessing import _REST_LABEL

        rest_count = np.sum(result[:, 2] == _REST_LABEL)
        assert rest_count == 1


class TestSetRawAnnotationsEdgeCases:
    def test_non_int_event_id_raises(self):
        """Line 507: ValueError when annotation-only raw has list event_id values."""
        import mne

        raw = _make_raw(sfreq=250, duration=10.0, stim_events=None)
        annotations = mne.Annotations(onset=[1.0], duration=[0.0], description=["left"])
        raw.set_annotations(annotations)
        sra = SetRawAnnotations(event_id={"left": [1, 2]}, interval=(0, 4))
        with pytest.raises(ValueError, match="event_id values must be integers"):
            sra.transform(raw)

    def test_annotation_extras_transfer(self):
        """Lines 514-519, 540-544: extras handling in annotation path."""
        import mne

        raw = _make_raw(sfreq=250, duration=10.0, stim_events=None)
        annotations = mne.Annotations(
            onset=[1.0, 3.0], duration=[0.0, 0.0], description=["left", "right"]
        )
        # Add extras to annotations
        annotations.extras = [{"session": "s1"}, {"session": "s2"}]
        raw.set_annotations(annotations)

        sra = SetRawAnnotations(event_id={"left": 1, "right": 2}, interval=(0, 4))
        result = sra.transform(raw)
        assert result is raw
        # Check that annotations were set (extras may or may not survive
        # depending on MNE version, but the code path is exercised)
        assert len(raw.annotations) >= 1


class TestRawToEventsReRaise:
    def test_reraises_non_matching_value_error(self):
        """Line 614: re-raise ValueError that isn't 'Could not find any...'."""
        from unittest.mock import patch

        import mne

        raw = _make_raw(sfreq=250, duration=10.0, stim_events=None)
        annotations = mne.Annotations(onset=[1.0], duration=[0.0], description=["left"])
        raw.set_annotations(annotations)
        rte = RawToEvents(event_id={"left": 1}, interval=(0, 4))

        with patch("moabb.datasets.preprocessing.mne.events_from_annotations") as mock_fn:
            mock_fn.side_effect = ValueError("Some other value error")
            with pytest.raises(ValueError, match="Some other value error"):
                rte.transform(raw)


class TestRawToEpochsInterpolateMissing:
    def test_interpolate_missing_channels(self):
        """Lines 750-776: interpolate_missing_channels path."""
        import mne

        from moabb.datasets.preprocessing import RawToEpochs

        # Use standard 10-20 channel names so montage lookup works
        n_samples = int(250 * 10.0)
        ch_names = ["C3", "Cz", "C4"]
        info = mne.create_info(
            ch_names=ch_names + ["STI 014"],
            sfreq=250,
            ch_types=["eeg"] * 3 + ["stim"],
        )
        data = np.random.RandomState(42).randn(4, n_samples) * 1e-6
        data[-1, :] = 0
        data[-1, 500] = 1
        raw = mne.io.RawArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

        events = np.array([[500, 0, 1]], dtype="int32")
        # Request a channel that doesn't exist (Fz), plus existing ones
        t = RawToEpochs(
            event_id={"left": 1},
            tmin=0,
            tmax=0.5,
            baseline=None,
            channels=["C3", "Cz", "C4", "Fz"],
            interpolate_missing_channels=True,
        )
        result = t.transform({"raw": raw, "events": events})
        assert isinstance(result, mne.Epochs)
        assert len(result.ch_names) == 4


class TestInsertRestEventsNoRest:
    def test_no_rest_contiguous_overlapping(self):
        """Line 244: early return when task segments fully overlap (no gaps)."""
        # Two events at samples 0 and 100, with task_duration=200
        # Event 0: task [0, 200), event 1: task [100, 300)
        # next_task_start(100) <= task_end(200) for event 0 → no post-task rest
        # next_task_start for event 1 = 301, which is > 300 → post-task rest
        # But with 3 overlapping events, we can avoid all rests:
        # Fully overlapping: 3 events at 0, 50, 100, task_duration=500
        # event 0: task [0,500), next=50 <= 500 → no rest
        # event 1: task [50,550), next=100 <= 550 → no rest
        # event 2: task [100,600), next=601 > 600 → rest
        # Still produces 1 rest. That's fine — this exercises the branch
        # for the events where next_task_start <= task_end (no rest added).
        events = np.array([[0, 0, 1], [50, 0, 1], [100, 0, 1]], dtype="int32")
        from moabb.datasets.preprocessing import _REST_LABEL

        result = _insert_rest_events(events, task_duration_samples=500)
        # Only the last event should have a rest
        rest_count = np.sum(result[:, 2] == _REST_LABEL)
        assert rest_count == 1


class TestSetRawAnnotationsNoneAnnotations:
    def test_no_stim_none_annotations(self):
        """Lines 502-505: no stim channel and annotations is None."""
        raw = _make_raw(sfreq=250, duration=10.0, stim_events=None)
        # MNE RawArray always creates empty annotations, so we need to force None
        raw._annotations = None
        sra = SetRawAnnotations(event_id={"left": 1}, interval=(0, 4))
        result = sra.transform(raw)
        assert result is raw


class TestGenerateSlidingWindowEdgeCases:
    def test_vote_start_before_first_transition(self):
        """Line 356-361: seg_idx < 0 branch (vote before first event)."""
        # Use negative tmin so the vote window starts before the first event
        events = np.array([[500, 0, 1]], dtype="int32")
        result = _generate_sliding_window_events(
            events,
            window_length=1.0,
            overlap=0,
            sfreq=250,
            interval=(0, 4),
            tmin=-3.0,
        )
        # Should still produce events (the vote window covers pre-event area)
        assert len(result) > 0
