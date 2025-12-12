"""Tests for lazy metadata cache functionality."""

import tempfile
from pathlib import Path

import pytest

from moabb.datasets.fake import FakeDataset
from moabb.datasets.metadata_cache import (
    FixedIntervalTrialEstimator,
    MetadataCache,
    RunInfo,
    SessionInfo,
    SubjectInfo,
    _compute_params_hash,
)
from moabb.evaluations.splitters import (
    CrossSessionSplitter,
    CrossSubjectSplitter,
    UnifiedSplitter,
    WithinSessionSplitter,
)
from moabb.paradigms.fixed_interval_windows import FixedIntervalWindowsProcessing
from moabb.paradigms.motor_imagery import MotorImagery


class TestMetadataCache:
    """Tests for MetadataCache class."""

    def test_create_empty_cache(self):
        """Test creating an empty cache."""
        cache = MetadataCache(dataset_code="TestDataset")
        assert cache.dataset_code == "TestDataset"
        assert cache.n_subjects == 0
        assert cache.subject_list == []

    def test_add_subjects_sessions_runs(self):
        """Test building cache structure manually."""
        cache = MetadataCache(dataset_code="TestDataset")

        # Add a subject with sessions and runs
        run1 = RunInfo(run_id="0", n_trials=100, duration=60.0)
        session1 = SessionInfo(session_id="0", runs={"0": run1})
        subject1 = SubjectInfo(subject_id=1, sessions={"0": session1})
        cache.subjects[1] = subject1

        assert cache.n_subjects == 1
        assert cache.subject_list == [1]
        assert cache.subjects[1].n_sessions == 1
        assert cache.subjects[1].sessions["0"].runs["0"].n_trials == 100

    def test_to_metadata_df(self):
        """Test converting cache to DataFrame."""
        cache = MetadataCache(dataset_code="TestDataset")

        # Add two subjects with two sessions each
        for subj_id in [1, 2]:
            sessions = {}
            for sess_id in ["0", "1"]:
                run = RunInfo(run_id="0", n_trials=50)
                sessions[sess_id] = SessionInfo(session_id=sess_id, runs={"0": run})
            cache.subjects[subj_id] = SubjectInfo(subject_id=subj_id, sessions=sessions)

        df = cache.to_metadata_df()
        assert len(df) == 4  # 2 subjects * 2 sessions * 1 run
        assert "subject" in df.columns
        assert "session" in df.columns
        assert "run" in df.columns
        assert "n_trials" in df.columns

    def test_to_metadata_df_expand_trials(self):
        """Test expanding trials in DataFrame."""
        cache = MetadataCache(dataset_code="TestDataset")

        run = RunInfo(run_id="0", n_trials=10)
        session = SessionInfo(session_id="0", runs={"0": run})
        cache.subjects[1] = SubjectInfo(subject_id=1, sessions={"0": session})

        df = cache.to_metadata_df(expand_trials=True)
        assert len(df) == 10  # One row per trial

    def test_json_serialization(self):
        """Test saving and loading cache to/from JSON."""
        cache = MetadataCache(dataset_code="TestDataset")

        run = RunInfo(run_id="0", n_trials=100, events={"left": 50, "right": 50})
        session = SessionInfo(session_id="0", runs={"0": run})
        cache.subjects[1] = SubjectInfo(subject_id=1, sessions={"0": session})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            cache.to_json(path)

            loaded = MetadataCache.from_json(path)
            assert loaded.dataset_code == "TestDataset"
            assert loaded.n_subjects == 1
            assert loaded.subjects[1].sessions["0"].runs["0"].n_trials == 100
            assert loaded.subjects[1].sessions["0"].runs["0"].events == {
                "left": 50,
                "right": 50,
            }

    def test_params_hash(self):
        """Test parameter hash computation."""
        hash1 = _compute_params_hash({"a": 1, "b": 2})
        hash2 = _compute_params_hash({"b": 2, "a": 1})  # Same params, different order
        hash3 = _compute_params_hash({"a": 1, "b": 3})  # Different params

        assert hash1 == hash2  # Order shouldn't matter
        assert hash1 != hash3  # Different params should give different hash


class TestFixedIntervalTrialEstimator:
    """Tests for trial count estimation."""

    def test_basic_estimation(self):
        """Test basic trial count estimation."""
        estimator = FixedIntervalTrialEstimator(
            length=5.0, stride=10.0, start_offset=0.0, stop_offset=None
        )

        # 100s recording with 5s windows and 10s stride
        # Starting at t=0: windows at 0-5, 10-15, 20-25, ..., 90-95 = 10 windows
        assert estimator.estimate_n_trials(100.0) == 10

    def test_with_offsets(self):
        """Test estimation with start/stop offsets."""
        estimator = FixedIntervalTrialEstimator(
            length=5.0, stride=10.0, start_offset=10.0, stop_offset=10.0
        )

        # 100s recording, but effective duration is 100 - 10 - 10 = 80s
        # Windows at 10-15, 20-25, ..., 80-85 = 8 windows
        assert estimator.estimate_n_trials(100.0) == 8

    def test_short_recording(self):
        """Test with recording shorter than window length."""
        estimator = FixedIntervalTrialEstimator(
            length=10.0, stride=5.0, start_offset=0.0, stop_offset=None
        )

        assert estimator.estimate_n_trials(5.0) == 0  # Too short


class TestFixedIntervalWindowsProcessing:
    """Tests for FixedIntervalWindowsProcessing trial estimation."""

    def test_estimate_n_trials(self):
        """Test estimate_n_trials method."""
        paradigm = FixedIntervalWindowsProcessing(
            fmin=7, fmax=45, length=5.0, stride=10.0
        )

        assert paradigm.estimate_n_trials(100.0) == 10

    def test_get_trial_count_estimator(self):
        """Test get_trial_count_estimator method."""
        paradigm = FixedIntervalWindowsProcessing(
            fmin=7, fmax=45, length=5.0, stride=10.0, start_offset=5.0
        )

        estimator = paradigm.get_trial_count_estimator()
        assert isinstance(estimator, FixedIntervalTrialEstimator)
        assert estimator.length == 5.0
        assert estimator.stride == 10.0
        assert estimator.start_offset == 5.0


class TestSplittersWithCache:
    """Tests for splitters using metadata cache."""

    @pytest.fixture
    def fake_cache(self):
        """Create a fake metadata cache."""
        cache = MetadataCache(dataset_code="FakeDataset")

        # 5 subjects, 2 sessions each, 50 trials per session
        for subj_id in range(1, 6):
            sessions = {}
            for sess_idx in range(2):
                sess_id = str(sess_idx)
                run = RunInfo(run_id="0", n_trials=50)
                sessions[sess_id] = SessionInfo(session_id=sess_id, runs={"0": run})
            cache.subjects[subj_id] = SubjectInfo(subject_id=subj_id, sessions=sessions)

        return cache

    def test_within_session_n_splits(self, fake_cache):
        """Test WithinSessionSplitter.get_n_splits_from_cache."""
        splitter = WithinSessionSplitter(n_folds=5)
        n_splits = splitter.get_n_splits_from_cache(fake_cache)

        # 5 subjects * 2 sessions * 5 folds = 50 splits
        assert n_splits == 50

    def test_cross_session_n_splits(self, fake_cache):
        """Test CrossSessionSplitter.get_n_splits_from_cache."""
        splitter = CrossSessionSplitter()
        n_splits = splitter.get_n_splits_from_cache(fake_cache)

        # 5 subjects * 2 sessions (LOSO) = 10 splits
        assert n_splits == 10

    def test_cross_subject_n_splits(self, fake_cache):
        """Test CrossSubjectSplitter.get_n_splits_from_cache."""
        splitter = CrossSubjectSplitter()
        n_splits = splitter.get_n_splits_from_cache(fake_cache)

        # 5 subjects (LOSO) = 5 splits
        assert n_splits == 5


class TestUnifiedSplitter:
    """Tests for UnifiedSplitter."""

    @pytest.fixture
    def fake_cache(self):
        """Create a fake metadata cache."""
        cache = MetadataCache(dataset_code="FakeDataset")

        for subj_id in range(1, 4):  # 3 subjects
            sessions = {}
            for sess_idx in range(2):  # 2 sessions each
                sess_id = str(sess_idx)
                run = RunInfo(run_id="0", n_trials=30)
                sessions[sess_id] = SessionInfo(session_id=sess_id, runs={"0": run})
            cache.subjects[subj_id] = SubjectInfo(subject_id=subj_id, sessions=sessions)

        return cache

    def test_within_session_splits(self, fake_cache):
        """Test UnifiedSplitter for within-session evaluation."""
        splitter = UnifiedSplitter("within_session", n_folds=3)

        n_splits = splitter.get_n_splits_from_cache(fake_cache)
        assert n_splits == 18  # 3 subjects * 2 sessions * 3 folds

        splits = list(splitter.generate_splits_from_cache(fake_cache))
        assert len(splits) == 18
        assert all(s.evaluation_type == "within_session" for s in splits)

    def test_cross_session_splits(self, fake_cache):
        """Test UnifiedSplitter for cross-session evaluation."""
        splitter = UnifiedSplitter("cross_session")

        n_splits = splitter.get_n_splits_from_cache(fake_cache)
        assert n_splits == 6  # 3 subjects * 2 sessions (LOSO)

        splits = list(splitter.generate_splits_from_cache(fake_cache))
        assert len(splits) == 6
        assert all(s.evaluation_type == "cross_session" for s in splits)

    def test_cross_subject_splits(self, fake_cache):
        """Test UnifiedSplitter for cross-subject evaluation."""
        splitter = UnifiedSplitter("cross_subject")

        n_splits = splitter.get_n_splits_from_cache(fake_cache)
        assert n_splits == 3  # 3 subjects (LOSO)

        splits = list(splitter.generate_splits_from_cache(fake_cache))
        assert len(splits) == 3
        assert all(s.evaluation_type == "cross_subject" for s in splits)

    def test_split_info_required_subjects(self, fake_cache):
        """Test SplitInfo.get_required_subjects method."""
        splitter = UnifiedSplitter("cross_subject")
        splits = list(splitter.generate_splits_from_cache(fake_cache))

        # For cross-subject, each split needs all subjects
        for split in splits:
            required = split.get_required_subjects()
            assert set(required) == {1, 2, 3}


class TestFakeDatasetIntegration:
    """Integration tests using FakeDataset."""

    def test_get_metadata_cache(self):
        """Test getting metadata cache from FakeDataset."""
        dataset = FakeDataset()
        cache = dataset.get_metadata_cache()

        assert cache.dataset_code == dataset.code
        assert cache.n_subjects == len(dataset.subject_list)

    def test_paradigm_get_metadata_lazy(self):
        """Test paradigm's get_metadata_lazy method."""
        dataset = FakeDataset()
        paradigm = MotorImagery()

        df = paradigm.get_metadata_lazy(dataset)

        assert "subject" in df.columns
        assert "session" in df.columns
        assert set(df["subject"].unique()) == set(dataset.subject_list)
