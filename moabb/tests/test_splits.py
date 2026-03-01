import numpy as np
import pytest
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.utils import check_random_state

import pandas as pd

from moabb.datasets.fake import FakeDataset
from moabb.evaluations.splitters import (
    CrossDatasetSplitter,
    CrossSessionSplitter,
    WithinSessionSplitter,
)
from moabb.paradigms.motor_imagery import FakeImageryParadigm


@pytest.fixture
def data():
    dataset = FakeDataset(
        ["left_hand", "right_hand"], n_subjects=5, seed=12, n_sessions=5
    )
    paradigm = FakeImageryParadigm()
    return paradigm.get_data(dataset=dataset)


# Split done for the Within Session evaluation
def eval_split_within_session(shuffle, random_state, data):
    _, y, metadata = data
    rng = check_random_state(random_state) if shuffle else None

    all_index = metadata.index.values
    subjects = metadata["subject"].unique()
    if shuffle:
        rng.shuffle(subjects)

    for i, subject in enumerate(subjects):
        subject_mask = metadata["subject"] == subject

        subject_indices = all_index[subject_mask]
        subject_metadata = metadata[subject_mask]
        sessions = subject_metadata["session"].unique()
        y_subject = y[subject_mask]

        if shuffle:
            rng.shuffle(sessions)

        for session in sessions:
            session_mask = subject_metadata["session"] == session
            indices = subject_indices[session_mask]
            metadata_ = subject_metadata[session_mask]
            y_ = y_subject[session_mask]

            cv = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=rng)

            for idx_train, idx_test in cv.split(metadata_, y_):
                yield indices[idx_train], indices[idx_test]


def eval_split_cross_session(shuffle, random_state, data):
    _, y, metadata = data

    rng = check_random_state(random_state) if shuffle else None

    subjects = metadata["subject"].unique()

    for subject in subjects:
        subject_mask = metadata["subject"] == subject
        subject_metadata = metadata[subject_mask]
        subject_sessions = subject_metadata["session"].unique()

        if shuffle:
            splitter = GroupShuffleSplit(n_splits=len(subject_sessions), random_state=rng)
        else:
            splitter = LeaveOneGroupOut()

        for train_ix, test_ix in splitter.split(
            X=subject_metadata, y=y[subject_mask], groups=subject_metadata["session"]
        ):
            yield subject_metadata.index[train_ix], subject_metadata.index[test_ix]


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_within_session_compatibility(shuffle, random_state, data):
    _, y, metadata = data

    split = WithinSessionSplitter(n_folds=5, shuffle=shuffle, random_state=random_state)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        eval_split_within_session(shuffle=shuffle, random_state=random_state, data=data),
        split.split(y, metadata),
    ):
        # Check if the output is the same as the input
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


def test_is_shuffling(data):
    X, y, metadata = data

    split = WithinSessionSplitter(n_folds=5, shuffle=False)
    split_shuffle = WithinSessionSplitter(n_folds=5, shuffle=True, random_state=3)

    for (train, test), (train_shuffle, test_shuffle) in zip(
        split.split(y, metadata), split_shuffle.split(y, metadata)
    ):
        # Check if the output is the same as the input
        assert not np.array_equal(train, train_shuffle)
        assert not np.array_equal(test, test_shuffle)


@pytest.mark.parametrize("splitter", [WithinSessionSplitter, CrossSessionSplitter])
def test_custom_inner_cv(
    splitter,
    data,
):
    X, y, metadata = data
    # Use a custom inner cv
    split = splitter(cv_class=TimeSeriesSplit, max_train_size=2)

    for train, test in split.split(y, metadata):
        # Check if the output is the same as the input
        assert len(train) <= 2  # Due to TimeSeriesSplit constraints
        assert len(test) >= 20


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_cross_session(shuffle, random_state, data):
    _, y, metadata = data

    if shuffle:
        split = CrossSessionSplitter(
            shuffle=shuffle, random_state=random_state, cv_class=GroupShuffleSplit
        )
    else:
        split = CrossSessionSplitter(shuffle=shuffle, random_state=random_state)

    for idx_train_splitter, idx_test_splitter in split.split(y, metadata):
        # Check if the output is the same as the input
        session_train = metadata.iloc[idx_train_splitter]["session"].unique()
        session_test = metadata.iloc[idx_test_splitter]["session"].unique()
        assert np.intersect1d(session_train, session_test).size == 0
        assert (
            np.union1d(session_train, session_test).size
            == metadata["session"].unique().size
        )


@pytest.mark.parametrize("shuffle, random_state", [(False, None), (True, 0), (True, 42)])
def test_cross_session_compatibility(shuffle, random_state, data):
    _, y, metadata = data

    if shuffle:
        splitter = CrossSessionSplitter(
            shuffle=shuffle, random_state=random_state, cv_class=GroupShuffleSplit
        )
    else:
        splitter = CrossSessionSplitter(shuffle=shuffle, random_state=random_state)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        eval_split_cross_session(shuffle=shuffle, random_state=random_state, data=data),
        splitter.split(y, metadata),
    ):
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


def test_cross_session_is_shuffling_and_order(data):
    _, y, metadata = data

    splitter_no_shuffle = CrossSessionSplitter(shuffle=False)
    splitter_shuffle = CrossSessionSplitter(
        shuffle=True, random_state=3, cv_class=GroupShuffleSplit
    )

    splits_no_shuffle = list(splitter_no_shuffle.split(y, metadata))
    splits_shuffle = list(splitter_shuffle.split(y, metadata))

    train_diff = []
    test_diff = []

    # For tracking session order differences
    session_orders_no_shuffle = []
    session_orders_shuffle = []

    for i, ((train_ns, test_ns), (train_s, test_s)) in enumerate(
        zip(splits_no_shuffle, splits_shuffle)
    ):
        print(f"\nFold {i}:")

        # Get session ordering for non-shuffled and shuffled
        train_ns_sessions = metadata.iloc[train_ns]["session"].unique()
        test_ns_sessions = metadata.iloc[test_ns]["session"].unique()
        train_s_sessions = metadata.iloc[train_s]["session"].unique()
        test_s_sessions = metadata.iloc[test_s]["session"].unique()

        print(f"Train no shuffle sessions: {train_ns_sessions}")
        print(f"Test no shuffle sessions : {test_ns_sessions}")
        print(f"Train shuffled sessions  : {train_s_sessions}")
        print(f"Test shuffle sessions    : {test_s_sessions}")

        # Track if indices are the same
        train_diff.append(np.array_equal(train_ns, train_s))
        test_diff.append(np.array_equal(test_ns, test_s))

        # Track session orders
        session_orders_no_shuffle.append(
            (list(train_ns_sessions), list(test_ns_sessions))
        )
        session_orders_shuffle.append((list(train_s_sessions), list(test_s_sessions)))

    # Check if indices are different in at least some folds
    assert not all(train_diff), "All train indices are identical despite shuffle"
    assert not all(test_diff), "All test indices are identical despite shuffle"

    # Check if session ordering is different
    session_order_differences = [
        not (
            np.array_equal(no_shuffle[0], shuffle[0])
            and np.array_equal(no_shuffle[1], shuffle[1])
        )
        for no_shuffle, shuffle in zip(session_orders_no_shuffle, session_orders_shuffle)
    ]

    assert any(session_order_differences), (
        "Session ordering is identical in all folds despite shuffle. "
        "When shuffle=True, we expect some difference in the session ordering."
    )


def test_cross_session_unique_subjects(data):
    _, y, metadata = data

    splitter_shuffle = CrossSessionSplitter(
        shuffle=True, random_state=3, cv_class=GroupShuffleSplit
    )
    splits_shuffle = list(splitter_shuffle.split(y, metadata))

    # Check if session splits are different across subjects
    subject_session_patterns = {}
    for i, (train_idx, test_idx) in enumerate(splits_shuffle):
        subject = metadata.iloc[train_idx]["subject"].iloc[
            0
        ]  # Get the subject for this fold
        if subject not in subject_session_patterns:
            subject_session_patterns[subject] = []

        train_sessions = set(metadata.iloc[train_idx]["session"].unique())
        test_sessions = set(metadata.iloc[test_idx]["session"].unique())
        subject_session_patterns[subject].append((train_sessions, test_sessions))

    # Verify that at least some subjects have different session splitting patterns
    pattern_differences = []
    subjects = list(subject_session_patterns.keys())
    for sub1, sub2 in zip(subjects, subjects[1:]):
        # Compare patterns for each subject pair
        patterns_differ = False
        for (train1, test1), (train2, test2) in zip(
            subject_session_patterns[sub1], subject_session_patterns[sub2]
        ):
            if train1 != train2 or test1 != test2:
                patterns_differ = True
                break
        pattern_differences.append(patterns_differ)

    assert any(
        pattern_differences
    ), "Session splitting patterns are identical across all subjects"


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_cross_session_unique_sessions(shuffle, random_state, data):
    _, y, metadata = data
    if shuffle:
        split = CrossSessionSplitter(
            shuffle=shuffle, random_state=random_state, cv_class=GroupShuffleSplit
        )
    else:
        split = CrossSessionSplitter(shuffle=shuffle, random_state=random_state)

    splits = list(split.split(y, metadata))

    for i, (train, test) in enumerate(splits):
        train_sessions = metadata.iloc[train]["session"].unique()
        test_sessions = metadata.iloc[test]["session"].unique()
        assert not np.intersect1d(
            train_sessions, test_sessions
        ).size, f"Fold {i} train and test sessions overlap"


@pytest.mark.parametrize("shuffle", [True, False])
def test_cross_session_get_n_splits(data, shuffle):
    _, y, metadata = data
    if shuffle:
        split = CrossSessionSplitter(shuffle=shuffle, cv_class=GroupShuffleSplit)
    else:
        split = CrossSessionSplitter()

    n_splits = split.get_n_splits(metadata)
    assert n_splits == 5 * 5  # 5 subjects, 5 sessions each


def test_if_split_is_not_random(data):
    _, y, metadata = data

    split = CrossSessionSplitter(
        shuffle=True, random_state=42, cv_class=GroupShuffleSplit
    )

    splits = list(split.split(y, metadata))
    splits_2 = list(split.split(y, metadata))

    for (train, test), (train_2, test_2) in zip(splits, splits_2):
        print(f"Train: {train}")
        print(f"Test: {test}")
        print(f"Train 2: {train_2}")
        print(f"Test 2: {test_2}")
        assert np.array_equal(train, train_2)
        assert np.array_equal(test, test_2)


@pytest.mark.parametrize(
    "cv_class",
    [
        LeaveOneGroupOut,
        TimeSeriesSplit,
        GroupKFold,
        LeaveOneOut,
        LeavePGroupsOut,
        LeavePOut,
    ],
)
def test_raise_error_on_invalid_cv_class(cv_class):
    with pytest.raises(ValueError):
        CrossSessionSplitter(shuffle=True, cv_class=cv_class)


@pytest.mark.parametrize(
    "cv_class",
    [
        GroupShuffleSplit,
        StratifiedKFold,
        KFold,
        RepeatedKFold,
        RepeatedStratifiedKFold,
        ShuffleSplit,
        StratifiedGroupKFold,
        StratifiedShuffleSplit,
    ],
)
def test_cross_session_splitter_without_error(
    cv_class,
):
    splitter = CrossSessionSplitter(shuffle=True, cv_class=cv_class)
    assert splitter is not None
    assert isinstance(splitter, CrossSessionSplitter)


# ---------- CrossDatasetSplitter tests ----------


@pytest.fixture
def cross_dataset_data():
    """Create synthetic metadata for cross-dataset splitter tests."""
    # Simulate two training datasets and one test dataset
    n_train1 = 40  # 2 subjects x 20 samples
    n_train2 = 30  # 1 subject x 30 samples
    n_test = 40  # 2 subjects x 20 samples

    y = np.concatenate(
        [
            np.tile([0, 1], n_train1 // 2),
            np.tile([0, 1], n_train2 // 2),
            np.tile([0, 1], n_test // 2),
        ]
    )

    metadata = pd.DataFrame(
        {
            "dataset": (
                ["ds_train_1"] * n_train1
                + ["ds_train_2"] * n_train2
                + ["ds_test"] * n_test
            ),
            "subject": (
                [1] * 20 + [2] * 20 + [1] * 30 + [1] * 20 + [2] * 20
            ),
            "session": (
                ["s1"] * 10 + ["s2"] * 10 + ["s1"] * 10 + ["s2"] * 10
                + ["s1"] * 15 + ["s2"] * 15
                + ["s1"] * 10 + ["s2"] * 10 + ["s1"] * 10 + ["s2"] * 10
            ),
        }
    )

    train_codes = ["ds_train_1", "ds_train_2"]
    test_codes = ["ds_test"]
    return y, metadata, train_codes, test_codes


def test_cross_dataset_split_count(cross_dataset_data):
    """Test that the splitter yields one split per test subject."""
    y, metadata, train_codes, test_codes = cross_dataset_data
    splitter = CrossDatasetSplitter(train_codes, test_codes)
    splits = list(splitter.split(y, metadata))
    # ds_test has 2 subjects, so 2 splits
    assert len(splits) == 2


def test_cross_dataset_train_from_train_dataset(cross_dataset_data):
    """Test that training indices come only from training datasets."""
    y, metadata, train_codes, test_codes = cross_dataset_data
    splitter = CrossDatasetSplitter(train_codes, test_codes)

    for train_idx, _ in splitter.split(y, metadata):
        train_datasets = metadata.iloc[train_idx]["dataset"].unique()
        for ds in train_datasets:
            assert ds in train_codes, f"Train index from non-train dataset: {ds}"


def test_cross_dataset_test_from_test_dataset(cross_dataset_data):
    """Test that test indices come only from test datasets."""
    y, metadata, train_codes, test_codes = cross_dataset_data
    splitter = CrossDatasetSplitter(train_codes, test_codes)

    for _, test_idx in splitter.split(y, metadata):
        test_datasets = metadata.iloc[test_idx]["dataset"].unique()
        for ds in test_datasets:
            assert ds in test_codes, f"Test index from non-test dataset: {ds}"


def test_cross_dataset_get_n_splits(cross_dataset_data):
    """Test that get_n_splits returns the correct count."""
    y, metadata, train_codes, test_codes = cross_dataset_data
    splitter = CrossDatasetSplitter(train_codes, test_codes)
    assert splitter.get_n_splits(y, metadata) == 2


def test_cross_dataset_test_single_subject_per_split(cross_dataset_data):
    """Test that each split tests exactly one subject."""
    y, metadata, train_codes, test_codes = cross_dataset_data
    splitter = CrossDatasetSplitter(train_codes, test_codes)

    for _, test_idx in splitter.split(y, metadata):
        subjects = metadata.iloc[test_idx]["subject"].unique()
        assert len(subjects) == 1, f"Expected 1 test subject, got {len(subjects)}"


def test_cross_dataset_train_indices_consistent(cross_dataset_data):
    """Test that train indices are the same across all splits."""
    y, metadata, train_codes, test_codes = cross_dataset_data
    splitter = CrossDatasetSplitter(train_codes, test_codes)

    train_indices_list = [train_idx for train_idx, _ in splitter.split(y, metadata)]
    for train_idx in train_indices_list:
        np.testing.assert_array_equal(train_idx, train_indices_list[0])


def test_cross_dataset_multiple_test_datasets():
    """Test splitter with multiple test datasets."""
    y = np.tile([0, 1], 50)
    metadata = pd.DataFrame(
        {
            "dataset": ["train_ds"] * 40 + ["test_ds_1"] * 30 + ["test_ds_2"] * 30,
            "subject": (
                [1] * 20 + [2] * 20
                + [1] * 15 + [2] * 15
                + [1] * 15 + [2] * 15
            ),
            "session": ["s1"] * 100,
        }
    )
    splitter = CrossDatasetSplitter(["train_ds"], ["test_ds_1", "test_ds_2"])
    splits = list(splitter.split(y, metadata))
    # 2 subjects in test_ds_1 + 2 subjects in test_ds_2 = 4 splits
    assert len(splits) == 4
    assert splitter.get_n_splits(y, metadata) == 4
