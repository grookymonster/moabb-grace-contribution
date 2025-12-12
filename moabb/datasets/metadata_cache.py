"""Lazy Metadata Cache for MOABB.

This module provides utilities for extracting and caching dataset metadata
without loading the raw EEG data. This enables efficient parallelization
by determining split structures upfront.

The metadata cache stores:
- Subject/session/run structure from BIDS paths
- Event counts per run from events.tsv files
- Recording durations for trial estimation

This information is sufficient to compute cross-validation splits
without the expensive operation of loading and preprocessing raw data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd


if TYPE_CHECKING:
    from moabb.datasets.base import BaseDataset

log = logging.getLogger(__name__)

# Remote URL for pre-generated metadata caches
METADATA_CACHE_URL = (
    "https://raw.githubusercontent.com/NeuroTechX/moabb-metadata-cache/main"
)


def _compute_params_hash(params: dict) -> str:
    """Compute a hash of paradigm parameters for cache invalidation.

    Parameters
    ----------
    params : dict
        Dictionary of paradigm parameters.

    Returns
    -------
    str
        8-character hash string.
    """
    params_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


@dataclass
class RunInfo:
    """Information about a single run.

    Parameters
    ----------
    run_id : str
        The run identifier (e.g., "0", "1", "0train").
    n_trials : int | None
        Number of trials/epochs in this run, if known.
    duration : float | None
        Recording duration in seconds, if known.
    events : dict[str, int] | None
        Event counts by event type, if known.
    fpath : str | None
        Path to the raw data file.
    """

    run_id: str
    n_trials: int | None = None
    duration: float | None = None
    events: dict[str, int] | None = None
    fpath: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "n_trials": self.n_trials,
            "duration": self.duration,
            "events": self.events,
            "fpath": self.fpath,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunInfo":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            n_trials=data.get("n_trials"),
            duration=data.get("duration"),
            events=data.get("events"),
            fpath=data.get("fpath"),
        )


@dataclass
class SessionInfo:
    """Information about a single session.

    Parameters
    ----------
    session_id : str
        The session identifier (e.g., "0", "1", "0train").
    runs : dict[str, RunInfo]
        Dictionary mapping run_id to RunInfo.
    """

    session_id: str
    runs: dict[str, RunInfo] = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        """Number of runs in this session."""
        return len(self.runs)

    @property
    def total_trials(self) -> int | None:
        """Total trials across all runs, or None if any run has unknown count."""
        trials = [r.n_trials for r in self.runs.values()]
        if any(t is None for t in trials):
            return None
        return sum(trials)

    @property
    def total_duration(self) -> float | None:
        """Total duration across all runs, or None if any run has unknown duration."""
        durations = [r.duration for r in self.runs.values()]
        if any(d is None for d in durations):
            return None
        return sum(durations)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "runs": {k: v.to_dict() for k, v in self.runs.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionInfo":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            runs={k: RunInfo.from_dict(v) for k, v in data.get("runs", {}).items()},
        )


@dataclass
class SubjectInfo:
    """Information about a single subject.

    Parameters
    ----------
    subject_id : int
        The subject identifier.
    sessions : dict[str, SessionInfo]
        Dictionary mapping session_id to SessionInfo.
    """

    subject_id: int
    sessions: dict[str, SessionInfo] = field(default_factory=dict)

    @property
    def n_sessions(self) -> int:
        """Number of sessions for this subject."""
        return len(self.sessions)

    @property
    def session_ids(self) -> list[str]:
        """List of session IDs."""
        return list(self.sessions.keys())

    @property
    def total_trials(self) -> int | None:
        """Total trials across all sessions, or None if any session has unknown count."""
        trials = [s.total_trials for s in self.sessions.values()]
        if any(t is None for t in trials):
            return None
        return sum(trials)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "subject_id": self.subject_id,
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubjectInfo":
        """Create from dictionary."""
        return cls(
            subject_id=data["subject_id"],
            sessions={
                k: SessionInfo.from_dict(v) for k, v in data.get("sessions", {}).items()
            },
        )


@dataclass
class MetadataCache:
    """Cached metadata for a dataset.

    This class stores the structure of a dataset (subjects, sessions, runs)
    along with event counts and recording durations, enabling split computation
    without loading raw EEG data.

    Parameters
    ----------
    dataset_code : str
        The unique identifier for the dataset.
    subjects : dict[int, SubjectInfo]
        Dictionary mapping subject_id to SubjectInfo.
    paradigm : str | None
        The paradigm type (e.g., "imagery", "p300").
    params_hash : str | None
        Hash of paradigm parameters for cache invalidation.
    moabb_version : str | None
        MOABB version used to generate this cache.

    Examples
    --------
    >>> cache = MetadataCache.from_dataset(dataset)
    >>> metadata_df = cache.to_metadata_df()
    >>> n_splits = splitter.get_n_splits_from_cache(cache)
    """

    dataset_code: str
    subjects: dict[int, SubjectInfo] = field(default_factory=dict)
    paradigm: str | None = None
    params_hash: str | None = None
    moabb_version: str | None = None

    @property
    def subject_list(self) -> list[int]:
        """List of subject IDs."""
        return sorted(self.subjects.keys())

    @property
    def n_subjects(self) -> int:
        """Number of subjects."""
        return len(self.subjects)

    def get_sessions_for_subject(self, subject: int) -> list[str]:
        """Get session IDs for a specific subject."""
        if subject not in self.subjects:
            return []
        return self.subjects[subject].session_ids

    def get_n_sessions_per_subject(self) -> dict[int, int]:
        """Get number of sessions per subject."""
        return {s: info.n_sessions for s, info in self.subjects.items()}

    def get_min_sessions(self) -> int:
        """Get minimum number of sessions across all subjects."""
        if not self.subjects:
            return 0
        return min(info.n_sessions for info in self.subjects.values())

    def to_metadata_df(self, expand_trials: bool = False) -> pd.DataFrame:
        """Convert to a metadata DataFrame.

        Parameters
        ----------
        expand_trials : bool, default=False
            If True, expand rows to have one row per trial.
            If False, have one row per run with n_trials column.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: subject, session, run, [n_trials].
            If expand_trials=True, n_trials column is omitted and
            there's one row per trial.
        """
        records = []
        for subject_id, subject_info in self.subjects.items():
            for session_id, session_info in subject_info.sessions.items():
                for run_id, run_info in session_info.runs.items():
                    if expand_trials and run_info.n_trials is not None:
                        # Expand to one row per trial
                        for _ in range(run_info.n_trials):
                            records.append(
                                {
                                    "subject": subject_id,
                                    "session": session_id,
                                    "run": run_id,
                                }
                            )
                    else:
                        records.append(
                            {
                                "subject": subject_id,
                                "session": session_id,
                                "run": run_id,
                                "n_trials": run_info.n_trials,
                                "duration": run_info.duration,
                            }
                        )
        return pd.DataFrame(records)

    def get_trial_index(self) -> pd.DataFrame:
        """Get a DataFrame with one row per trial, suitable for splitting.

        This is used by splitters to determine train/test indices.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: subject, session, run.
            One row per trial, with sequential integer index.
        """
        return self.to_metadata_df(expand_trials=True).reset_index(drop=True)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        import moabb

        return {
            "dataset_code": self.dataset_code,
            "subjects": {str(k): v.to_dict() for k, v in self.subjects.items()},
            "paradigm": self.paradigm,
            "params_hash": self.params_hash,
            "moabb_version": self.moabb_version or moabb.__version__,
        }

    def to_json(self, path: str | Path) -> None:
        """Save cache to JSON file.

        Parameters
        ----------
        path : str | Path
            Path to save the JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info(f"Saved metadata cache to {path}")

    @classmethod
    def from_dict(cls, data: dict) -> "MetadataCache":
        """Create from dictionary."""
        return cls(
            dataset_code=data["dataset_code"],
            subjects={
                int(k): SubjectInfo.from_dict(v)
                for k, v in data.get("subjects", {}).items()
            },
            paradigm=data.get("paradigm"),
            params_hash=data.get("params_hash"),
            moabb_version=data.get("moabb_version"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "MetadataCache":
        """Load cache from JSON file.

        Parameters
        ----------
        path : str | Path
            Path to the JSON file.

        Returns
        -------
        MetadataCache
            Loaded cache instance.
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dataset(
        cls,
        dataset: "BaseDataset",
        paradigm_params: dict | None = None,
        include_events: bool = True,
        include_durations: bool = True,
    ) -> "MetadataCache":
        """Create metadata cache from a dataset without loading raw data.

        This method extracts the dataset structure from BIDS paths and
        optionally parses events.tsv files for event counts.

        Parameters
        ----------
        dataset : BaseDataset
            The dataset to extract metadata from.
        paradigm_params : dict | None
            Paradigm parameters for hash computation.
        include_events : bool, default=True
            Whether to parse events.tsv files for event counts.
        include_durations : bool, default=True
            Whether to extract recording durations from sidecar files.

        Returns
        -------
        MetadataCache
            Cache containing dataset structure and metadata.
        """
        import moabb
        from moabb.datasets.base import BaseBIDSDataset

        cache = cls(
            dataset_code=dataset.code,
            paradigm=dataset.paradigm,
            params_hash=_compute_params_hash(paradigm_params or {}),
            moabb_version=moabb.__version__,
        )

        if isinstance(dataset, BaseBIDSDataset):
            cache._populate_from_bids(
                dataset,
                include_events=include_events,
                include_durations=include_durations,
            )
        else:
            cache._populate_from_legacy(dataset)

        return cache

    def _populate_from_bids(
        self,
        dataset: "BaseDataset",
        include_events: bool = True,
        include_durations: bool = True,
    ) -> None:
        """Populate cache from BIDS dataset structure.

        Parameters
        ----------
        dataset : BaseBIDSDataset
            The BIDS dataset.
        include_events : bool
            Whether to parse events.tsv files.
        include_durations : bool
            Whether to extract recording durations.
        """
        from moabb.datasets.bids_interface import run_bids_to_moabb

        # Get all BIDS paths without loading data
        try:
            bids_paths = dataset.bids_paths(subject=None)
        except Exception as e:
            log.warning(f"Could not get BIDS paths for all subjects: {e}")
            # Fall back to per-subject approach
            bids_paths = []
            for subject in dataset.subject_list:
                try:
                    bids_paths.extend(dataset.bids_paths(subject=subject))
                except Exception as e2:
                    log.warning(f"Could not get BIDS paths for subject {subject}: {e2}")

        for bids_path in bids_paths:
            # Extract subject, session, run from BIDS path
            try:
                subject_id = int(bids_path.subject)
            except (TypeError, ValueError):
                log.warning(f"Invalid subject ID in path: {bids_path}")
                continue

            session_id = bids_path.session or "0"
            run_id = run_bids_to_moabb(bids_path)

            # Ensure subject exists
            if subject_id not in self.subjects:
                self.subjects[subject_id] = SubjectInfo(subject_id=subject_id)

            # Ensure session exists
            if session_id not in self.subjects[subject_id].sessions:
                self.subjects[subject_id].sessions[session_id] = SessionInfo(
                    session_id=session_id
                )

            # Create run info
            run_info = RunInfo(
                run_id=run_id,
                fpath=str(bids_path.fpath) if bids_path.fpath else None,
            )

            # Parse events.tsv if available
            if include_events:
                events = self._parse_events_tsv(bids_path, dataset.event_id)
                if events:
                    run_info.events = events
                    run_info.n_trials = sum(events.values())

            # Get duration from sidecar JSON if available
            if include_durations:
                duration = self._get_duration_from_sidecar(bids_path)
                if duration is not None:
                    run_info.duration = duration

            self.subjects[subject_id].sessions[session_id].runs[run_id] = run_info

    def _populate_from_legacy(self, dataset: "BaseDataset") -> None:
        """Populate cache from legacy (non-BIDS) dataset structure.

        For legacy datasets, we construct the structure from subject_list
        and n_sessions, but we don't have event counts without loading data.

        Parameters
        ----------
        dataset : BaseDataset
            The legacy dataset.
        """
        for subject_id in dataset.subject_list:
            subject_info = SubjectInfo(subject_id=subject_id)

            # Create sessions based on n_sessions
            for session_idx in range(dataset.n_sessions):
                session_id = str(session_idx)
                session_info = SessionInfo(session_id=session_id)

                # Create a single run per session (we don't know the actual structure)
                run_info = RunInfo(run_id="0")
                session_info.runs["0"] = run_info

                subject_info.sessions[session_id] = session_info

            self.subjects[subject_id] = subject_info

    def _parse_events_tsv(
        self, bids_path: Any, event_id: dict[str, int]
    ) -> dict[str, int] | None:
        """Parse events.tsv file to count events.

        Parameters
        ----------
        bids_path : BIDSPath
            The BIDS path for the raw data file.
        event_id : dict[str, int]
            Mapping of event names to codes.

        Returns
        -------
        dict[str, int] | None
            Event counts by event name, or None if file not found.
        """
        try:
            # Construct events.tsv path
            events_path = bids_path.copy().update(suffix="events", extension=".tsv")
            if not events_path.fpath.exists():
                return None

            # Read events file
            events_df = pd.read_csv(events_path.fpath, sep="\t")

            if "trial_type" not in events_df.columns:
                # Try alternative column names
                for col in ["value", "event_type", "stim_type"]:
                    if col in events_df.columns:
                        events_df["trial_type"] = events_df[col]
                        break
                else:
                    return None

            # Count events matching event_id
            event_counts = {}
            for event_name in event_id.keys():
                count = (events_df["trial_type"] == event_name).sum()
                if count > 0:
                    event_counts[event_name] = int(count)

            return event_counts if event_counts else None

        except Exception as e:
            log.debug(f"Could not parse events.tsv for {bids_path}: {e}")
            return None

    def _get_duration_from_sidecar(self, bids_path: Any) -> float | None:
        """Get recording duration from BIDS sidecar JSON.

        Parameters
        ----------
        bids_path : BIDSPath
            The BIDS path for the raw data file.

        Returns
        -------
        float | None
            Recording duration in seconds, or None if not available.
        """
        try:
            # Try to find sidecar JSON
            json_path = bids_path.copy().update(extension=".json")
            if not json_path.fpath.exists():
                return None

            with open(json_path.fpath) as f:
                sidecar = json.load(f)

            # Try different field names
            for field in ["RecordingDuration", "Duration", "recording_duration"]:
                if field in sidecar:
                    return float(sidecar[field])

            return None

        except Exception as e:
            log.debug(f"Could not get duration from sidecar for {bids_path}: {e}")
            return None

    def update_trial_counts(
        self,
        estimator: "TrialCountEstimator",
    ) -> None:
        """Update trial counts using a trial count estimator.

        This is useful for paradigms like FixedIntervalWindows where
        trial counts can be estimated from recording duration.

        Parameters
        ----------
        estimator : TrialCountEstimator
            Object with estimate_n_trials(duration: float) -> int method.
        """
        for subject_info in self.subjects.values():
            for session_info in subject_info.sessions.values():
                for run_info in session_info.runs.values():
                    if run_info.n_trials is None and run_info.duration is not None:
                        run_info.n_trials = estimator.estimate_n_trials(run_info.duration)


class TrialCountEstimator:
    """Base class for estimating trial counts from recording duration.

    Subclasses should implement estimate_n_trials().
    """

    def estimate_n_trials(self, duration: float) -> int:
        """Estimate the number of trials from recording duration.

        Parameters
        ----------
        duration : float
            Recording duration in seconds.

        Returns
        -------
        int
            Estimated number of trials.
        """
        raise NotImplementedError


class FixedIntervalTrialEstimator(TrialCountEstimator):
    """Estimate trials for fixed interval windows paradigm.

    Parameters
    ----------
    length : float
        Length of each epoch in seconds.
    stride : float
        Stride between epochs in seconds.
    start_offset : float
        Start offset from beginning of recording in seconds.
    stop_offset : float | None
        Stop offset from end of recording in seconds.
    """

    def __init__(
        self,
        length: float,
        stride: float,
        start_offset: float = 0.0,
        stop_offset: float | None = None,
    ):
        self.length = length
        self.stride = stride
        self.start_offset = start_offset
        self.stop_offset = stop_offset

    def estimate_n_trials(self, duration: float) -> int:
        """Estimate number of fixed-interval epochs.

        Parameters
        ----------
        duration : float
            Recording duration in seconds.

        Returns
        -------
        int
            Estimated number of epochs.
        """
        effective_duration = duration - self.start_offset
        if self.stop_offset is not None:
            effective_duration -= self.stop_offset

        if effective_duration < self.length:
            return 0

        # Number of complete epochs that fit
        n_trials = int((effective_duration - self.length) / self.stride) + 1
        return max(0, n_trials)


def get_local_cache_path(dataset_code: str, params_hash: str | None = None) -> Path:
    """Get the local path for a dataset's metadata cache.

    Parameters
    ----------
    dataset_code : str
        The dataset code.
    params_hash : str | None
        Optional paradigm parameters hash.

    Returns
    -------
    Path
        Path to the local cache file.
    """
    from moabb.datasets import download as dl

    cache_dir = Path(dl.get_dataset_path("metadata_cache", None))
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{dataset_code}"
    if params_hash:
        filename += f"_{params_hash}"
    filename += ".json"

    return cache_dir / filename


def fetch_metadata_cache(
    dataset: "BaseDataset",
    paradigm_params: dict | None = None,
    force_update: bool = False,
) -> MetadataCache:
    """Fetch or generate metadata cache for a dataset.

    This function:
    1. Checks for a local cache file
    2. If not found, attempts to fetch from remote URL
    3. If fetch fails, generates cache from dataset

    Parameters
    ----------
    dataset : BaseDataset
        The dataset to get cache for.
    paradigm_params : dict | None
        Paradigm parameters for cache key.
    force_update : bool, default=False
        If True, regenerate cache even if it exists.

    Returns
    -------
    MetadataCache
        The metadata cache for the dataset.
    """
    import moabb

    params_hash = _compute_params_hash(paradigm_params or {})
    local_path = get_local_cache_path(dataset.code, params_hash)

    # Check local cache
    if not force_update and local_path.exists():
        try:
            cache = MetadataCache.from_json(local_path)
            # Validate version
            if cache.moabb_version == moabb.__version__:
                log.info(f"Loaded metadata cache from {local_path}")
                return cache
            else:
                log.info(
                    f"Cache version mismatch ({cache.moabb_version} vs {moabb.__version__}), regenerating"
                )
        except Exception as e:
            log.warning(f"Failed to load local cache: {e}")

    # Try to fetch from remote
    if not force_update:
        try:
            cache = _fetch_remote_cache(dataset.code, params_hash)
            if cache is not None:
                cache.to_json(local_path)
                return cache
        except Exception as e:
            log.debug(f"Failed to fetch remote cache: {e}")

    # Generate from dataset
    log.info(f"Generating metadata cache for {dataset.code}")
    cache = MetadataCache.from_dataset(dataset, paradigm_params)
    cache.to_json(local_path)
    return cache


def _fetch_remote_cache(
    dataset_code: str, params_hash: str | None = None
) -> MetadataCache | None:
    """Attempt to fetch metadata cache from remote URL.

    Parameters
    ----------
    dataset_code : str
        The dataset code.
    params_hash : str | None
        Optional paradigm parameters hash.

    Returns
    -------
    MetadataCache | None
        The fetched cache, or None if not available.
    """
    import urllib.error
    import urllib.request

    filename = f"{dataset_code}"
    if params_hash:
        filename += f"_{params_hash}"
    filename += ".json"

    url = f"{METADATA_CACHE_URL}/{filename}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            return MetadataCache.from_dict(data)
    except urllib.error.URLError:
        return None
    except Exception as e:
        log.debug(f"Failed to fetch from {url}: {e}")
        return None
