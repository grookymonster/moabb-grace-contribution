import inspect
import logging

from sklearn.model_selection import (
    BaseCrossValidator,
    LeaveOneGroupOut,
    StratifiedKFold,
)
from sklearn.utils import check_random_state


log = logging.getLogger(__name__)


class WithinSessionSplitter(BaseCrossValidator):
    """Data splitter for within session evaluation.

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets for each subject in each session. This splitter
    assumes that all data from all subjects is already known and loaded.

    .. image:: https://raw.githubusercontent.com/NeuroTechX/moabb/refs/heads/develop/docs/source/images/withinsess.png
        :alt: The schematic diagram of the WithinSession split
        :align: center

    The inner cross-validation strategy can be changed by passing the
    `cv_class` and `cv_kwargs` arguments. By default, it uses StratifiedKFold.

    Parameters
    ----------
    n_folds : int, default=5
        Number of folds. Must be at least 2. If
    shuffle : bool, default=True
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of splits. Only used when `shuffle` is True.
        Pass an int for reproducible output across multiple function calls.
    cv_class: cros-validation class, default=StratifiedKFold
        Inner cross-validation strategy for splitting the sessions.
    cv_kwargs: dict
        Additional arguments to pass to the inner cross-validation strategy.

    """

    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = None,
        cv_class: type[BaseCrossValidator] = StratifiedKFold,
        **cv_kwargs,
    ):
        self.cv_class = cv_class
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.cv_kwargs = cv_kwargs
        self._cv_kwargs = dict(**cv_kwargs)

        self.random_state = random_state
        self._rng = check_random_state(random_state) if shuffle else None

        if not shuffle and random_state is not None:
            raise ValueError("random_state should be None when shuffle is False")

        # Create a dictionary of parameters by adding arguments only if they
        # are part of the inner cross-validation strategy's signature
        params = inspect.signature(self.cv_class).parameters
        for p, v in [
            ("n_splits", n_folds),
            ("shuffle", shuffle),
            ("random_state", self._rng),
        ]:
            if p in params:
                self._cv_kwargs[p] = v

    def get_n_splits(self, metadata):
        num_sessions_subjects = metadata.groupby(["subject", "session"]).ngroups
        return self.n_folds * num_sessions_subjects

    def get_n_splits_from_cache(self, cache):
        """Compute number of splits from metadata cache without loading data.

        Parameters
        ----------
        cache : MetadataCache
            Metadata cache containing dataset structure.

        Returns
        -------
        int
            Total number of cross-validation splits.

        Notes
        -----
        .. versionadded:: 1.2.0
        """
        # Count total (subject, session) pairs
        n_session_subject_pairs = sum(
            subject_info.n_sessions for subject_info in cache.subjects.values()
        )
        return self.n_folds * n_session_subject_pairs

    def split(self, y, metadata):
        all_index = metadata.index.values

        # Shuffle subjects if required
        subjects = metadata["subject"].unique()
        if self.shuffle:
            self._rng.shuffle(subjects)

        for subject in subjects:
            subject_mask = metadata["subject"] == subject
            subject_indices = all_index[subject_mask]
            subject_metadata = metadata[subject_mask]
            y_subject = y[subject_mask]

            # Shuffle sessions if required
            sessions = subject_metadata["session"].unique()

            if self.shuffle:
                self._rng.shuffle(sessions)

            for session in sessions:
                session_mask = subject_metadata["session"] == session
                indices = subject_indices[session_mask]
                y_session = y_subject[session_mask]

                # Instantiate a new internal splitter for each session
                splitter = self.cv_class(**self._cv_kwargs)

                # Split using the current instance of StratifiedKFold by default
                for train_ix, test_ix in splitter.split(indices, y_session):

                    yield indices[train_ix], indices[test_ix]


class CrossSessionSplitter(BaseCrossValidator):
    """Data splitter for cross session evaluation.

    This splitter enables cross-session evaluation by performing a Leave-One-Session-Out (LOSO)
    cross-validation on data from each subject.

    It assumes that the entire metainformation across all subjects is already loaded.

    Unlike the `CrossSessionEvaluation` class from `moabb.evaluation`, which manages
    the complete evaluation process end-to-end, this splitter is solely responsible
    for dividing the data into training and testing sets based on sessions.

    .. image:: https://raw.githubusercontent.com/NeuroTechX/moabb/refs/heads/develop/docs/source/images/crosssess.jpg
        :alt: The schematic diagram of the CrossSession split
        :align: center

    The inner cross-validation strategy can be changed by passing the
    `cv_class` and `cv_kwargs` arguments. By default, it uses LeaveOneGroupOut,
    which performs Leave-One-Session-Out cross-validation.

    Parameters
    ----------
    cv_class: cross-validation class, default=LeaveOneGroupOut
        Inner cross-validation strategy for splitting the sessions of one subject.
        LeaveOneGroupOut is the most common default.
    shuffle: bool, default=False
        Whether to shuffle the session order for each subject. It can only be
        used when changing the `cv_class` to a class compatible with `shuffle`.
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the inner cross-validation when `shuffle` is True.
        Pass an int for reproducible output across multiple function calls.
        For `cv_class` accepting `random_state`, they are provided with a shared rng.
    cv_kwargs: dict
        Additional arguments to pass to the inner cross-validation strategy.

    Yields
    ------
    train : ndarray
        The training set indices for that split.

    test : ndarray
        The testing set indices for that split.
    """

    def __init__(
        self,
        cv_class: type[BaseCrossValidator] = LeaveOneGroupOut,
        shuffle: bool = False,
        random_state: int = None,
        **cv_kwargs,
    ):
        self.cv_class = cv_class
        self.cv_kwargs = cv_kwargs
        self._cv_kwargs = dict(**cv_kwargs)

        self.shuffle = shuffle
        self.random_state = random_state

        params = inspect.signature(self.cv_class).parameters
        # When shuffle=True, only allow cv_classes that explicitly support shuffling
        # (i.e., have a 'shuffle' parameter)
        # changing here
        params_key = params.keys()

        if shuffle and ("shuffle" not in params_key and "random_state" not in params_key):
            raise ValueError(
                f"Shuffling is not supported for {cv_class.__name__}. "
                "Choose a different `cv_class` or use `shuffle=False`. "
                "Example of `cv_class`: `GroupShuffleSplit`: "
                "CrossSessionSplitter(shuffle=True, random_state=42, cv_class=GroupShuffleSplit)"
            )

        if not shuffle and "shuffle" in params and random_state is not None:
            raise ValueError(
                "`random_state` should be None when `shuffle` is False for {cv_class.__name__}"
            )

        self._need_rng = "random_state" in params and (shuffle or "shuffle" not in params)

        if "shuffle" in params:
            self._cv_kwargs["shuffle"] = shuffle

    def get_n_splits(self, metadata):
        """
        Return the number of splits for the cross-validation.

        The number of splits is the number of subjects times the number of splits
        of the inner cross-validation strategy.

        We try to keep the same behaviour as the sklearn cross-validation classes.

        Parameters
        ----------
        metadata: pd.DataFrame
            The metadata containing the subject and session information.

        Returns
        -------
        n_splits: int
            The number of splits for the cross-validation
        """
        subjects = metadata["subject"].unique()
        n_splits = 0
        for subject in subjects:
            subject_metadata = metadata.query("subject == @subject")
            sessions = subject_metadata["session"].unique()

            if len(sessions) <= 1:
                continue  # Skip subjects with only one session

            splitter = self.cv_class(**self._cv_kwargs)
            n_splits += splitter.get_n_splits(
                subject_metadata, groups=subject_metadata["session"]
            )
        return n_splits

    def get_n_splits_from_cache(self, cache):
        """Compute number of splits from metadata cache without loading data.

        For cross-session evaluation, the number of splits depends on:
        - Number of subjects with >= 2 sessions
        - The inner CV strategy (default: LeaveOneGroupOut = n_sessions per subject)

        Parameters
        ----------
        cache : MetadataCache
            Metadata cache containing dataset structure.

        Returns
        -------
        int
            Total number of cross-validation splits.

        Notes
        -----
        .. versionadded:: 1.2.0
        """
        n_splits = 0
        for subject_info in cache.subjects.values():
            n_sessions = subject_info.n_sessions
            if n_sessions <= 1:
                continue  # Skip subjects with only one session

            # For LeaveOneGroupOut, n_splits = n_sessions
            # For other CV classes, we'd need to estimate differently
            if self.cv_class == LeaveOneGroupOut:
                n_splits += n_sessions
            else:
                # Create a dummy splitter to estimate
                splitter = self.cv_class(**self._cv_kwargs)
                # Try to get n_splits with n_groups
                try:
                    # Some splitters have get_n_splits that takes groups
                    n_splits += splitter.get_n_splits(
                        X=range(n_sessions), groups=range(n_sessions)
                    )
                except TypeError:
                    # Fall back to assuming LOSO behavior
                    n_splits += n_sessions
        return n_splits

    def split(self, y, metadata):
        # here, I am getting the index across all the subject
        all_index = metadata.index.values
        # I check how many subjects are here:
        subjects = metadata["subject"].unique()

        # To make sure that when I shuffle the subject, I shuffle the same way
        # the session when the object is created
        cv_kwargs = {**self._cv_kwargs}  # Copy the original kwargs
        if self._need_rng:
            cv_kwargs["random_state"] = check_random_state(self.random_state)

        # For each subject I am creating the mask to select the subject metainformation.
        for subject in subjects:
            # Creating the subject_mask
            subject_mask = metadata["subject"] == subject
            # from all the index, I am getting the trial index
            subject_indices = all_index[subject_mask]
            # Here, I am getting the metainformation to use the column session
            subject_metadata = metadata[subject_mask]
            # getting the label at subject level
            y_subject = y[subject_mask]
            # check the number of sessions and check how many sessions we
            # have!
            sessions = subject_metadata["session"].unique()

            if len(sessions) <= 1:
                log.info(
                    f"Skipping subject {subject}: Only one session available"
                    f"Cross-session evaluation requires at least two sessions."
                )
                continue  # Skip subjects with only one session

            # by default, I am using LeaveOneGroupOut
            splitter = self.cv_class(**cv_kwargs)

            # Yield the splits for a given subject
            for train_session_idx, test_session_idx in splitter.split(
                X=subject_indices, y=y_subject, groups=subject_metadata["session"]
            ):
                # returning the index
                yield subject_indices[train_session_idx], subject_indices[
                    test_session_idx
                ]


class CrossSubjectSplitter(BaseCrossValidator):
    """Data splitter for cross subject evaluation.

    This splitter enables cross-subject evaluation by performing a Leave-One-Session-Out (LOSO)
    cross-validation on the dataset.

    It assumes that the entire metainformation across all subjects is already loaded.

    Unlike the `CrossSubjectEvaluation` class from `moabb.evaluation`, which manages
    the complete evaluation process end-to-end, this splitter is solely responsible
    for dividing the data into training and testing sets based on subjects.

    .. image:: https://raw.githubusercontent.com/NeuroTechX/moabb/refs/heads/develop/docs/source/images/crosssubj.png
        :alt: The schematic diagram of the CrossSubject split
        :align: center

    The splitting strategy for the subjects can be changed by passing the
    `cv_class` and `cv_kwargs` arguments. By default, it uses LeaveOneGroupOut,
    which performs Leave-One-Subject-Out cross-validation.

    Parameters
    ----------
    cv_class: cross-validation class, default=LeaveOneGroupOut
        Cross-validation strategy for splitting the subjects between train and test sets.
        By default, use LeaveOneGroupOut, which keeps one subject as a test.
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the cross-validation.
        Pass an int for reproducible output across multiple calls.
    cv_kwargs: dict
        Additional arguments to pass to the inner cross-validation strategy.

    Yields
    ------
    train : ndarray
        The training set indices for that split.

    test : ndarray
        The testing set indices for that split.
    """

    def __init__(
        self,
        cv_class: type[BaseCrossValidator] = LeaveOneGroupOut,
        random_state: int = None,
        **cv_kwargs,
    ):
        self.cv_class = cv_class
        self.cv_kwargs = cv_kwargs
        self._cv_kwargs = dict(**cv_kwargs)

        params = inspect.signature(self.cv_class).parameters
        if "random_state" in params:
            self._cv_kwargs["random_state"] = random_state

    def get_n_splits(self, metadata):
        """
        Return the number of splits for the cross-validation.

        The number of splits is the number of subjects times the number of splits
        of the inner cross-validation strategy.

        We try to keep the same behaviour as the sklearn cross-validation classes.

        Parameters
        ----------
        metadata: pd.DataFrame
            The metadata containing the subject and session information.

        Returns
        -------
        n_splits: int
            The number of splits for the cross-validation
        """

        splitter = self.cv_class(**self._cv_kwargs)
        n_splits = splitter.get_n_splits(metadata.index, groups=metadata["subject"])
        return n_splits

    def get_n_splits_from_cache(self, cache):
        """Compute number of splits from metadata cache without loading data.

        For cross-subject evaluation with LeaveOneGroupOut (default),
        the number of splits equals the number of subjects.

        Parameters
        ----------
        cache : MetadataCache
            Metadata cache containing dataset structure.

        Returns
        -------
        int
            Total number of cross-validation splits.

        Notes
        -----
        .. versionadded:: 1.2.0
        """
        n_subjects = cache.n_subjects

        # For LeaveOneGroupOut, n_splits = n_subjects
        if self.cv_class == LeaveOneGroupOut:
            return n_subjects
        else:
            # Create a dummy splitter to estimate
            splitter = self.cv_class(**self._cv_kwargs)
            try:
                return splitter.get_n_splits(
                    X=range(n_subjects), groups=range(n_subjects)
                )
            except TypeError:
                # Fall back to assuming LOSO behavior
                return n_subjects

    def split(self, y, metadata):
        # here, I am getting the index across all the subject
        all_index = metadata.index.values

        splitter = self.cv_class(**self._cv_kwargs)

        # Yield the splits for the entire dataset
        for train_session_idx, test_session_idx in splitter.split(
            X=all_index, y=y, groups=metadata["subject"]
        ):
            # returning the index
            yield all_index[train_session_idx], all_index[test_session_idx]


class SplitInfo:
    """Information about a single cross-validation split.

    This class encapsulates all information needed to execute a single
    split without requiring the full metadata to be present.

    Parameters
    ----------
    split_idx : int
        Index of this split (0-based).
    evaluation_type : str
        Type of evaluation: "within_session", "cross_session", or "cross_subject".
    subjects : list[int]
        Subject(s) involved in this split.
    sessions : dict[int, list[str]] | None
        For within/cross-session: mapping of subject to session IDs.
    train_subjects : list[int] | None
        For cross-subject: subjects in training set.
    test_subjects : list[int] | None
        For cross-subject: subjects in test set.
    train_sessions : list[str] | None
        For cross-session: sessions in training set (per subject).
    test_sessions : list[str] | None
        For cross-session: sessions in test set (per subject).
    fold_idx : int | None
        For within-session: fold index within the session.

    Attributes
    ----------
    split_idx : int
    evaluation_type : str
    subjects : list[int]
    sessions : dict[int, list[str]] | None
    train_subjects : list[int] | None
    test_subjects : list[int] | None
    train_sessions : list[str] | None
    test_sessions : list[str] | None
    fold_idx : int | None

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(
        self,
        split_idx: int,
        evaluation_type: str,
        subjects: list,
        sessions: dict = None,
        train_subjects: list = None,
        test_subjects: list = None,
        train_sessions: list = None,
        test_sessions: list = None,
        fold_idx: int = None,
    ):
        self.split_idx = split_idx
        self.evaluation_type = evaluation_type
        self.subjects = subjects
        self.sessions = sessions
        self.train_subjects = train_subjects
        self.test_subjects = test_subjects
        self.train_sessions = train_sessions
        self.test_sessions = test_sessions
        self.fold_idx = fold_idx

    def __repr__(self):
        return (
            f"SplitInfo(idx={self.split_idx}, type={self.evaluation_type}, "
            f"subjects={self.subjects})"
        )

    def get_required_subjects(self) -> list:
        """Get list of subjects that need to be loaded for this split."""
        if self.evaluation_type == "cross_subject":
            # Need all subjects (train + test)
            return list(set((self.train_subjects or []) + (self.test_subjects or [])))
        else:
            # For within/cross-session, subjects list contains required subjects
            return self.subjects


class UnifiedSplitter:
    """Unified interface for all evaluation splitter types.

    This class wraps WithinSessionSplitter, CrossSessionSplitter, and
    CrossSubjectSplitter to provide a consistent interface for computing
    splits from metadata cache and generating SplitInfo objects.

    Parameters
    ----------
    evaluation_type : str
        Type of evaluation: "within_session", "cross_session", or "cross_subject".
    n_folds : int, default=5
        Number of folds for within-session evaluation.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    random_state : int | None, default=None
        Random state for reproducibility.
    cv_class : type | None, default=None
        Custom cross-validation class. If None, uses default for evaluation type.
    **cv_kwargs
        Additional arguments passed to the CV class.

    Examples
    --------
    >>> splitter = UnifiedSplitter("within_session", n_folds=5)
    >>> n_splits = splitter.get_n_splits_from_cache(cache)
    >>> for split_info in splitter.generate_splits_from_cache(cache):
    ...     subjects_to_load = split_info.get_required_subjects()

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    EVALUATION_TYPES = ("within_session", "cross_session", "cross_subject")

    def __init__(
        self,
        evaluation_type: str,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = None,
        cv_class: type = None,
        **cv_kwargs,
    ):
        if evaluation_type not in self.EVALUATION_TYPES:
            raise ValueError(
                f"evaluation_type must be one of {self.EVALUATION_TYPES}, "
                f"got {evaluation_type!r}"
            )

        self.evaluation_type = evaluation_type
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.cv_class = cv_class
        self.cv_kwargs = cv_kwargs

        # Create the underlying splitter
        self._splitter = self._create_splitter()

    def _create_splitter(self):
        """Create the appropriate splitter based on evaluation type."""
        if self.evaluation_type == "within_session":
            kwargs = {
                "n_folds": self.n_folds,
                "shuffle": self.shuffle,
                "random_state": self.random_state,
                **self.cv_kwargs,
            }
            if self.cv_class is not None:
                kwargs["cv_class"] = self.cv_class
            return WithinSessionSplitter(**kwargs)

        elif self.evaluation_type == "cross_session":
            # CrossSessionSplitter doesn't support shuffle with default LOSO
            kwargs = {
                "random_state": self.random_state,
                **self.cv_kwargs,
            }
            # Only pass shuffle if explicitly set and cv_class supports it
            if self.cv_class is not None:
                kwargs["cv_class"] = self.cv_class
                kwargs["shuffle"] = self.shuffle
            return CrossSessionSplitter(**kwargs)

        else:  # cross_subject
            kwargs = {
                "random_state": self.random_state,
                **self.cv_kwargs,
            }
            if self.cv_class is not None:
                kwargs["cv_class"] = self.cv_class
            return CrossSubjectSplitter(**kwargs)

    def get_n_splits(self, metadata):
        """Get number of splits from metadata DataFrame.

        Parameters
        ----------
        metadata : pd.DataFrame
            Metadata with subject, session columns.

        Returns
        -------
        int
            Number of splits.
        """
        return self._splitter.get_n_splits(metadata)

    def get_n_splits_from_cache(self, cache):
        """Get number of splits from metadata cache without loading data.

        Parameters
        ----------
        cache : MetadataCache
            Metadata cache containing dataset structure.

        Returns
        -------
        int
            Number of splits.
        """
        return self._splitter.get_n_splits_from_cache(cache)

    def split(self, y, metadata):
        """Generate train/test indices from data.

        Parameters
        ----------
        y : array-like
            Labels.
        metadata : pd.DataFrame
            Metadata with subject, session columns.

        Yields
        ------
        train : ndarray
            Training indices.
        test : ndarray
            Test indices.
        """
        yield from self._splitter.split(y, metadata)

    def generate_splits_from_cache(self, cache):
        """Generate SplitInfo objects from metadata cache.

        This method generates split information without requiring the actual
        data to be loaded. Each SplitInfo contains enough information to
        determine which subjects need to be loaded for that split.

        Parameters
        ----------
        cache : MetadataCache
            Metadata cache containing dataset structure.

        Yields
        ------
        SplitInfo
            Information about each split.

        Notes
        -----
        This is the key method for lazy parallelization: it allows computing
        all splits upfront based only on the dataset structure, then each
        split can be executed independently with only the required data loaded.
        """
        split_idx = 0

        if self.evaluation_type == "within_session":
            # Within-session: n_folds splits per (subject, session)
            for subject_id, subject_info in cache.subjects.items():
                for session_id in subject_info.session_ids:
                    for fold_idx in range(self.n_folds):
                        yield SplitInfo(
                            split_idx=split_idx,
                            evaluation_type=self.evaluation_type,
                            subjects=[subject_id],
                            sessions={subject_id: [session_id]},
                            fold_idx=fold_idx,
                        )
                        split_idx += 1

        elif self.evaluation_type == "cross_session":
            # Cross-session: LOSO per subject (leave one session out)
            for subject_id, subject_info in cache.subjects.items():
                session_ids = subject_info.session_ids
                if len(session_ids) <= 1:
                    continue  # Skip subjects with only one session

                for test_session in session_ids:
                    train_sessions = [s for s in session_ids if s != test_session]
                    yield SplitInfo(
                        split_idx=split_idx,
                        evaluation_type=self.evaluation_type,
                        subjects=[subject_id],
                        sessions={subject_id: session_ids},
                        train_sessions=train_sessions,
                        test_sessions=[test_session],
                    )
                    split_idx += 1

        else:  # cross_subject
            # Cross-subject: LOSO (leave one subject out)
            subject_ids = cache.subject_list
            for test_subject in subject_ids:
                train_subjects = [s for s in subject_ids if s != test_subject]
                yield SplitInfo(
                    split_idx=split_idx,
                    evaluation_type=self.evaluation_type,
                    subjects=subject_ids,
                    train_subjects=train_subjects,
                    test_subjects=[test_subject],
                )
                split_idx += 1
