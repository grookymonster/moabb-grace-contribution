# Lazy Metadata & Unified Parallel Evaluation for MOABB

## Problem Statement

I have a plan to improve the parallelization of Moabb.

To do this, I'm first trying to unify the evaluation, ensuring it's more agnostic whether we're doing it at the level of one model per subject with cross-validation (within-session), using one model per session and testing on another (cross-session), or cross-subject.

The problem is that I can't determine how many times I'll need to split without consuming the dataset beforehand, and I'd like to devise a way to do this without consuming the raw data (X), given that this is costly for large datasets, i.e., based on events or other information.

You can assume that all datasets are BIDS here.

## Solution: Lazy Metadata Cache System

**TL;DR:** Create a lazy metadata system that extracts subject/session/run structure and event counts from BIDS paths and `events.tsv` files without loading raw EEG data. Pre-generate `metadata_cache.json` files for all datasets with default parameters, hosted remotely. Enable unified evaluation that computes all splits upfront, then parallelizes by split rather than by dataset.

## Implementation Status

### ✅ Completed Components

#### 1. MetadataCache System (`moabb/datasets/metadata_cache.py`)

- `MetadataCache` dataclass with hierarchical structure: `subjects -> sessions -> runs`
- Each `RunInfo` contains: `run_id`, `n_trials`, `duration`, `events`, `fpath`
- Methods: `to_json()`, `from_json()`, `to_metadata_df()`, `get_trial_index()`
- `FixedIntervalTrialEstimator` for estimating trial counts from recording duration
- `fetch_metadata_cache()` for local/remote cache retrieval with fallback generation

#### 2. BaseDataset Integration (`moabb/datasets/base.py`)

- Added `get_metadata_cache()` method that returns `MetadataCache` without loading raw data
- For BIDS datasets: extracts structure from `bids_paths()` and parses `events.tsv` files
- For legacy datasets: constructs structure from `subject_list` and `n_sessions`

#### 3. FixedIntervalWindows Trial Estimation (`moabb/paradigms/fixed_interval_windows.py`)

- Added `estimate_n_trials(duration)` method using formula: `int((duration - start_offset - length) / stride) + 1`
- Added `get_trial_count_estimator()` returning configured `FixedIntervalTrialEstimator`

#### 4. Splitter Extensions (`moabb/evaluations/splitters.py`)

- Added `get_n_splits_from_cache(cache)` to all three splitters:
  - `WithinSessionSplitter`: `n_folds * n_session_subject_pairs`
  - `CrossSessionSplitter`: `sum(n_sessions per subject with >= 2 sessions)`
  - `CrossSubjectSplitter`: `n_subjects`

#### 5. UnifiedSplitter (`moabb/evaluations/splitters.py`)

- New class wrapping all splitter types with unified interface
- `generate_splits_from_cache(cache)` yields `SplitInfo` objects
- `SplitInfo` contains all information needed to execute a split without full metadata

#### 6. BaseParadigm Lazy Methods (`moabb/paradigms/base.py`)

- Added `get_paradigm_params()` for cache key computation
- Added `get_metadata_lazy(dataset)` returning DataFrame with subject/session/run
- Added `get_metadata_cache(dataset)` returning full `MetadataCache` object

#### 7. BaseEvaluation Lazy Methods (`moabb/evaluations/base.py`)

- Added `get_evaluation_type()` returning evaluation type string
- Added `get_metadata_cache(dataset)` delegating to paradigm
- Added `get_n_splits_lazy(dataset)` computing splits without loading data
- Added `get_all_splits_info(dataset)` yielding `SplitInfo` objects

## Usage Examples

```python
from moabb.datasets.fake import FakeDataset
from moabb.paradigms.motor_imagery import MotorImagery
from moabb.evaluations import WithinSessionEvaluation
from moabb.evaluations.splitters import UnifiedSplitter

# Get metadata cache without loading raw data
dataset = FakeDataset()
cache = dataset.get_metadata_cache()
print(f"Subjects: {cache.subject_list}")
print(f"Structure:\n{cache.to_metadata_df()}")

# Compute splits from cache
splitter = UnifiedSplitter("within_session", n_folds=5)
n_splits = splitter.get_n_splits_from_cache(cache)
print(f"Total splits: {n_splits}")

# Get split info for parallel execution
for split_info in splitter.generate_splits_from_cache(cache):
    subjects_needed = split_info.get_required_subjects()
    print(f"Split {split_info.split_idx}: needs subjects {subjects_needed}")

# Use in evaluation
paradigm = MotorImagery()
eval = WithinSessionEvaluation(paradigm=paradigm, datasets=[dataset])
print(f"Lazy split count: {eval.get_n_splits_lazy(dataset)}")
```

## Next Steps

### Remote Cache Hosting
- Create `NeuroTechX/moabb-metadata-cache` repository
- Add CI to regenerate caches on MOABB releases
- Implement cache fetching from remote URL

### Full Parallel Refactoring
- Modify `BaseEvaluation.process()` to:
  1. Compute all splits upfront using `get_all_splits_info()`
  2. Parallelize by split rather than by dataset
  3. Load only required subjects per split

### Cache Generation Script
- Create `scripts/generate_metadata_cache.py`
- Iterate all datasets and generate caches with default parameters

## Further Considerations

1. **Handling non-BIDS legacy datasets**: Currently uses `subject_list` + `n_sessions` as fallback. For full trial counts, datasets need to be converted to BIDS.

2. **Stratification for WithinSession CV**: BIDS `events.tsv` contains event labels—verify `trial_type` values match `event_id` keys. For stratified splits, the lazy metadata needs per-class counts.

3. **Cache invalidation strategy**: Includes `moabb_version`, `paradigm_class`, `paradigm_params_hash`. Regenerates if any mismatch.

4. **Recording duration**: BIDS sidecar JSON files contain `RecordingDuration` field—used for trial estimation in `FixedIntervalWindows` paradigm.

5. **Hosting location**: Recommend separate GitHub repo `NeuroTechX/moabb-metadata-cache` with CI that regenerates on MOABB releases.
