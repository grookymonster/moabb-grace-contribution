#!/usr/bin/env python3
"""Generate summary CSV tables from DATASET_METADATA_CATALOG.

This script auto-generates the 5 paradigm-specific CSV summary tables
(imagery, p300, ssvep, cvep, rstate) from the metadata catalog without
downloading any data.

Usage
-----
    python scripts/generate_summary_tables.py [--output-dir moabb/datasets/]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Overrides for values that metadata cannot represent
# (free-text, complex per-subject variations, etc.)
# Keyed by (dataset_name, column_name) -> str
# ---------------------------------------------------------------------------
_OVERRIDES = {
    # --- Imagery ---
    ("Stieger2021", "#Sessions"): "7 or 11",
    ("BNCI2020_001", "#Chan"): "varies (11-64)",
    ("BNCI2022_001", "#Trials / class"): "varies",
    ("BNCI2022_001", "Total_trials"): "varies",
    ("BNCI2024_001", "#Trials / class"): "varies",
    ("BNCI2024_001", "Total_trials"): "varies",
    ("BNCI2025_001", "#Trials / class"): "varies",
    ("BNCI2025_001", "Total_trials"): "varies",
    ("BNCI2025_002", "#Trials / class"): "varies",
    ("BNCI2025_002", "Total_trials"): "varies",
    ("BNCI2019_001", "#Trials / class"): "varies",
    ("BNCI2019_001", "Total_trials"): "varies",
    # --- P300 ---
    ("BNCI2014_008", "#Trials / class"): "3500 NT / 700 T",
    ("BNCI2014_009", "#Trials / class"): "1440 NT / 288 T",
    ("BNCI2015_003", "#Trials / class"): "1500 NT / 300 T",
    ("BI2013a", "#Sessions"): "8 for subjects 1-7 else 1",
    ("BI2013a", "#Trials / class"): "3200 NT / 640 T",
    ("BI2014a", "#Sessions"): "up to 3",
    ("BI2014a", "#Trials / class"): "990 NT / 198 T",
    ("BI2014b", "#Trials / class"): "200 NT / 40 T",
    ("BI2015a", "#Trials / class"): "4131 NT / 825 T",
    ("BI2015b", "#Trials / class"): "2160 NT / 480 T",
    ("Huebner2017", "#Trials / class"): "364 NT / 112 T",
    ("Huebner2018", "#Trials / class"): "364 NT / 112 T",
    ("Sosulski2019", "#Trials / class"): "7500 NT / 1500 T",
    ("EPFLP300", "#Trials / class"): "2753 NT / 551 T",
    ("Lee2019_ERP", "#Trials / class"): "6900 NT / 1380 T",
    ("ErpCore2021_N170", "#Trials / class"): "240 NT / 80 T",
    ("ErpCore2021_MMN", "#Trials / class"): "800 NT / 200 T",
    ("ErpCore2021_N2pc", "#Trials / class"): "160 NT / 160 T",
    ("ErpCore2021_P3", "#Trials / class"): "160 NT / 40 T",
    ("ErpCore2021_N400", "#Trials / class"): "60 NT / 60 T",
    ("ErpCore2021_ERN", "#Trials / class"): "~400 All",
    ("ErpCore2021_LRP", "#Trials / class"): "~400 All",
    ("Kojima2024A", "#Trials / class"): "~130 NT / ~65 T",
    ("Kojima2024B", "#Trials / class"): "2160 NT / 720 T",
    ("BNCI2015_009", "#Trials / class"): "10071 NT / 2014 T",
    ("BNCI2015_010", "#Trials / class"): "18850 NT / 650 T",
    ("BNCI2015_012", "#Trials / class"): "6075 NT / 759 T",
    ("BNCI2015_013", "#Trials / class"): "809 NT / 235 T",
    ("BNCI2016_002", "#Trials / class"): "varies brake / EMG",
    ("BNCI2020_002", "#Trials / class"): "varies NT / T",
    ("BNCI2015_006", "#Trials / class"): "~875 NT / ~856 T",
    ("BNCI2015_007", "#Trials / class"): "varies NT / T",
    ("BNCI2015_008", "#Trials / class"): "varies NT / T",
    ("RomaniBF2025ERP", "#Sessions"): "up to 3",
    ("RomaniBF2025ERP", "#Trials / class"): "540 NT / 60 T",
    # --- SSVEP ---
    ("Kalunga2016", "#Trials / class"): "16",
    ("MAMEM1", "#Trials / class"): "12-15",
    ("MAMEM2", "#Trials / class"): "20-30",
    ("MAMEM3", "#Trials / class"): "20-30",
    # --- c-VEP ---
    ("MartinezCagigal2023Pary", "Trials length (s)"): "5.3/6.7/10.3/4.0/10.0",
    ("MartinezCagigal2023Pary", "#Trial classes"): "16",
    ("MartinezCagigal2023Pary", "#Trials / class"): "2-30",
    ("MartinezCagigal2023Pary", "#Epochs classes"): "2-11",
    ("MartinezCagigal2023Pary", "#Epochs / class"): "6200-19220",
    ("MartinezCagigal2023Pary", "Codes"): "p-ary m-sequence",
    ("MartinezCagigal2023Pary", "Presentation rate (Hz)"): "120",
    ("MartinezCagigal2023Checker", "#Trial classes"): "16",
    ("MartinezCagigal2023Checker", "#Trials / class"): "2-30",
    ("MartinezCagigal2023Checker", "#Epochs classes"): "2",
    ("MartinezCagigal2023Checker", "#Epochs / class"): "11904/12288",
    ("MartinezCagigal2023Checker", "Codes"): "m-sequence",
    ("MartinezCagigal2023Checker", "Trials length (s)"): "4.2",
    ("MartinezCagigal2023Checker", "Presentation rate (Hz)"): "120",
    ("Thielen2015", "#Trial classes"): "36",
    ("Thielen2015", "#Trials / class"): "3",
    ("Thielen2015", "#Epochs classes"): "2",
    ("Thielen2015", "#Epochs / class"): "27216 NT / 27216 T",
    ("Thielen2015", "Codes"): "Gold codes",
    ("Thielen2015", "Presentation rate (Hz)"): "120",
    ("Thielen2021", "#Trial classes"): "20",
    ("Thielen2021", "#Trials / class"): "5",
    ("Thielen2021", "#Epochs classes"): "2",
    ("Thielen2021", "#Epochs / class"): "18900 NT / 18900 T",
    ("Thielen2021", "Codes"): "Gold codes",
    ("Thielen2021", "Presentation rate (Hz)"): "60",
    ("CastillosCVEP100", "#Trial classes"): "4",
    ("CastillosCVEP100", "#Trials / class"): "15/15/15/15",
    ("CastillosCVEP100", "#Epochs classes"): "2",
    ("CastillosCVEP100", "#Epochs / class"): "3525 NT / 3495 T",
    ("CastillosCVEP100", "Codes"): "m-sequence",
    ("CastillosCVEP100", "Presentation rate (Hz)"): "60",
    ("CastillosCVEP40", "#Trial classes"): "4",
    ("CastillosCVEP40", "#Trials / class"): "15/15/15/15",
    ("CastillosCVEP40", "#Epochs classes"): "2",
    ("CastillosCVEP40", "#Epochs / class"): "3525 NT / 3495 T",
    ("CastillosCVEP40", "Codes"): "m-sequence",
    ("CastillosCVEP40", "Presentation rate (Hz)"): "60",
    ("CastillosBurstVEP40", "#Trial classes"): "4",
    ("CastillosBurstVEP40", "#Trials / class"): "15/15/15/15",
    ("CastillosBurstVEP40", "#Epochs classes"): "2",
    ("CastillosBurstVEP40", "#Epochs / class"): "5820 NT / 1200 T",
    ("CastillosBurstVEP40", "Codes"): "Burst-CVEP",
    ("CastillosBurstVEP40", "Presentation rate (Hz)"): "60",
    ("CastillosBurstVEP100", "#Trial classes"): "4",
    ("CastillosBurstVEP100", "#Trials / class"): "15/15/15/15",
    ("CastillosBurstVEP100", "#Epochs classes"): "2",
    ("CastillosBurstVEP100", "#Epochs / class"): "5820 NT / 1200 T",
    ("CastillosBurstVEP100", "Codes"): "Burst-CVEP",
    ("CastillosBurstVEP100", "Presentation rate (Hz)"): "60",
    # --- Resting state ---
    ("Hinss2021", "#Blocks / class"): "1",
    ("Hinss2021", "Trials length (s)"): "2",
}


def _get_eeg_channels(meta):
    """Return the number of EEG channels (EEG-only convention)."""
    acq = meta.acquisition
    if acq is None:
        return None
    ct = acq.channel_types
    if ct and "eeg" in ct:
        return ct["eeg"]
    return acq.n_channels


def _get_sampling_rate(meta):
    """Return integer sampling rate."""
    if meta.acquisition and meta.acquisition.sampling_rate:
        return int(meta.acquisition.sampling_rate)
    return None


def _get_n_subjects(meta):
    """Return number of subjects."""
    if meta.participants:
        return meta.participants.n_subjects
    return None


def _get_n_classes(meta):
    """Return number of classes."""
    if meta.experiment:
        return meta.experiment.n_classes
    return None


def _get_trial_duration(meta):
    """Return trial duration in seconds."""
    if meta.experiment:
        return meta.experiment.trial_duration
    return None


def _get_sessions(meta):
    """Return sessions per subject."""
    return meta.sessions_per_subject or 1


def _get_runs(meta):
    """Return runs per session."""
    return meta.runs_per_session or 1


def _get_trials_per_class(meta):
    """Try to extract a single per-class trial count from metadata.

    Returns an integer if all classes have the same count, else None.
    """
    # Try experiment.trials_per_class first
    tpc = meta.experiment.trials_per_class if meta.experiment else None
    if isinstance(tpc, dict) and tpc:
        vals = [v for v in tpc.values() if isinstance(v, (int, float))]
        if vals and all(v == vals[0] for v in vals):
            return int(vals[0])

    # Try data_structure.n_trials_per_class
    ds = meta.data_structure
    if ds and isinstance(ds.n_trials_per_class, dict):
        vals = [v for v in ds.n_trials_per_class.values() if isinstance(v, (int, float))]
        if vals and all(v == vals[0] for v in vals):
            return int(vals[0])

    return None


def _get_p300_trials_str(meta):
    """Format P300 trial counts as 'NT_count NT / T_count T'."""
    ds = meta.data_structure
    if ds and isinstance(ds.n_trials_per_class, dict):
        nt_val = None
        t_val = None
        for key, val in ds.n_trials_per_class.items():
            if not isinstance(val, (int, float)):
                continue
            key_lower = key.lower()
            if "non" in key_lower or "nt" == key_lower or "non_target" in key_lower:
                nt_val = int(val)
            elif "target" in key_lower or key_lower == "t":
                t_val = int(val)
        if nt_val is not None and t_val is not None:
            return f"{nt_val} NT / {t_val} T"
    return None


def _apply_override(name, col, value):
    """Return override value if one exists, else the computed value."""
    return _OVERRIDES.get((name, col), value)


def _fmt(val):
    """Format a value for CSV output."""
    if val is None:
        return ""
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


# ---------------------------------------------------------------------------
# Paradigm-specific generators
# ---------------------------------------------------------------------------


def generate_imagery_table(datasets):
    """Generate the Motor Imagery summary table.

    Columns: Dataset, #Subj, #Chan, #Classes, #Trials / class,
             Trials length (s), Freq (Hz), #Sessions, #Runs, Total_trials
    """
    rows = []
    for name, meta in sorted(datasets.items()):
        n_subj = _get_n_subjects(meta)
        n_chan = _apply_override(name, "#Chan", _get_eeg_channels(meta))
        n_classes = _get_n_classes(meta)
        tpc = _apply_override(name, "#Trials / class", _get_trials_per_class(meta))
        trial_dur = _get_trial_duration(meta)
        freq = _get_sampling_rate(meta)
        sessions = _apply_override(name, "#Sessions", _get_sessions(meta))
        runs = _get_runs(meta)

        # Compute total trials
        total = _apply_override(name, "Total_trials", None)
        if total is None:
            try:
                total = (
                    int(n_subj) * int(sessions) * int(runs) * int(tpc) * int(n_classes)
                )
            except (TypeError, ValueError):
                total = "varies"

        rows.append(
            {
                "Dataset": name,
                "#Subj": _fmt(n_subj),
                "#Chan": _fmt(n_chan),
                "#Classes": _fmt(n_classes),
                "#Trials / class": _fmt(tpc),
                "Trials length (s)": _fmt(trial_dur),
                "Freq (Hz)": _fmt(freq),
                "#Sessions": _fmt(sessions),
                "#Runs": _fmt(runs),
                "Total_trials": _fmt(total),
            }
        )
    return pd.DataFrame(rows)


def generate_p300_table(datasets):
    """Generate the P300/ERP summary table.

    Columns: Dataset, #Subj, #Chan, #Trials / class,
             Trials length (s), Freq (Hz), #Sessions
    """
    rows = []
    for name, meta in sorted(datasets.items()):
        n_subj = _get_n_subjects(meta)
        n_chan = _get_eeg_channels(meta)
        trial_dur = _get_trial_duration(meta)
        freq = _get_sampling_rate(meta)
        sessions = _apply_override(name, "#Sessions", _get_sessions(meta))

        # Trial counts: prefer override, then computed NT/T string
        tpc_str = _apply_override(name, "#Trials / class", None)
        if tpc_str is None:
            tpc_str = _get_p300_trials_str(meta)
        if tpc_str is None:
            tpc_str = ""

        rows.append(
            {
                "Dataset": name,
                "#Subj": _fmt(n_subj),
                "#Chan": _fmt(n_chan),
                "#Trials / class": _fmt(tpc_str),
                "Trials length (s)": _fmt(trial_dur),
                "Freq (Hz)": _fmt(freq),
                "#Sessions": _fmt(sessions),
            }
        )
    return pd.DataFrame(rows)


def generate_ssvep_table(datasets):
    """Generate the SSVEP summary table.

    Columns: Dataset, #Subj, #Chan, #Classes, #Trials / class,
             Trials length (s), Freq (Hz), #Sessions
    """
    rows = []
    for name, meta in sorted(datasets.items()):
        n_subj = _get_n_subjects(meta)
        n_chan = _get_eeg_channels(meta)
        n_classes = _get_n_classes(meta)
        tpc = _apply_override(name, "#Trials / class", _get_trials_per_class(meta))
        trial_dur = _get_trial_duration(meta)
        freq = _get_sampling_rate(meta)
        sessions = _get_sessions(meta)

        rows.append(
            {
                "Dataset": name,
                "#Subj": _fmt(n_subj),
                "#Chan": _fmt(n_chan),
                "#Classes": _fmt(n_classes),
                "#Trials / class": _fmt(tpc),
                "Trials length (s)": _fmt(trial_dur),
                "Freq (Hz)": _fmt(freq),
                "#Sessions": _fmt(sessions),
            }
        )
    return pd.DataFrame(rows)


def generate_cvep_table(datasets):
    """Generate the c-VEP summary table.

    Columns: Dataset, #Subj, #Sessions, Freq (Hz), #Chan, Trials length (s),
             #Trial classes, #Trials / class, #Epochs classes, #Epochs / class,
             Codes, Presentation rate (Hz)
    """
    rows = []
    for name, meta in sorted(datasets.items()):
        n_subj = _get_n_subjects(meta)
        sessions = _get_sessions(meta)
        freq = _get_sampling_rate(meta)
        n_chan = _get_eeg_channels(meta)
        trial_dur = _apply_override(name, "Trials length (s)", _get_trial_duration(meta))
        n_classes = _get_n_classes(meta)

        # Most c-VEP fields come from overrides since the schema doesn't
        # have dedicated fields for epoch-level structure
        trial_classes = _apply_override(name, "#Trial classes", _fmt(n_classes))
        tpc = _apply_override(name, "#Trials / class", "")
        epochs_classes = _apply_override(name, "#Epochs classes", "")
        epochs_per_class = _apply_override(name, "#Epochs / class", "")

        # Code type from paradigm_specific
        ps = meta.paradigm_specific
        code_type_raw = ps.code_type if ps else None
        codes = _apply_override(name, "Codes", code_type_raw or "")
        pres_rate = _apply_override(name, "Presentation rate (Hz)", "")

        rows.append(
            {
                "Dataset": name,
                "#Subj": _fmt(n_subj),
                "#Sessions": _fmt(sessions),
                "Freq (Hz)": _fmt(freq),
                "#Chan": _fmt(n_chan),
                "Trials length (s)": _fmt(trial_dur),
                "#Trial classes": _fmt(trial_classes),
                "#Trials / class": _fmt(tpc),
                "#Epochs classes": _fmt(epochs_classes),
                "#Epochs / class": _fmt(epochs_per_class),
                "Codes": _fmt(codes),
                "Presentation rate (Hz)": _fmt(pres_rate),
            }
        )
    return pd.DataFrame(rows)


def generate_rstate_table(datasets):
    """Generate the Resting State summary table.

    Columns: Dataset, #Subj, #Chan, #Classes, #Blocks / class,
             Trials length (s), Freq (Hz), #Sessions
    """
    rows = []
    for name, meta in sorted(datasets.items()):
        n_subj = _get_n_subjects(meta)
        n_chan = _get_eeg_channels(meta)
        n_classes = _get_n_classes(meta)
        trial_dur = _apply_override(name, "Trials length (s)", _get_trial_duration(meta))
        freq = _get_sampling_rate(meta)
        sessions = _get_sessions(meta)

        # Blocks per class from data_structure
        blocks = _apply_override(name, "#Blocks / class", None)
        if blocks is None:
            ds = meta.data_structure
            if ds and ds.n_blocks:
                blocks = ds.n_blocks
            else:
                blocks = ""

        rows.append(
            {
                "Dataset": name,
                "#Subj": _fmt(n_subj),
                "#Chan": _fmt(n_chan),
                "#Classes": _fmt(n_classes),
                "#Blocks / class": _fmt(blocks),
                "Trials length (s)": _fmt(trial_dur),
                "Freq (Hz)": _fmt(freq),
                "#Sessions": _fmt(sessions),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paradigm grouping and main
# ---------------------------------------------------------------------------

# Map metadata paradigm values to CSV file paradigm names
_PARADIGM_MAP = {
    "imagery": "imagery",
    "p300": "p300",
    "ssvep": "ssvep",
    "cvep": "cvep",
    "rstate": "rstate",
}

_GENERATORS = {
    "imagery": generate_imagery_table,
    "p300": generate_p300_table,
    "ssvep": generate_ssvep_table,
    "cvep": generate_cvep_table,
    "rstate": generate_rstate_table,
}

# Datasets to exclude (base classes, not real datasets)
_EXCLUDE = {"ErpCore2021"}


def main(output_dir: str = None):
    from moabb.datasets.metadata import DATASET_METADATA_CATALOG

    output_path = Path(output_dir) if output_dir else Path("moabb/datasets")
    output_path.mkdir(parents=True, exist_ok=True)

    # Group datasets by paradigm
    grouped = {p: {} for p in _GENERATORS}
    for name, meta in DATASET_METADATA_CATALOG.items():
        if name in _EXCLUDE:
            continue
        paradigm = meta.experiment.paradigm if meta.experiment else None
        csv_paradigm = _PARADIGM_MAP.get(paradigm)
        if csv_paradigm and csv_paradigm in grouped:
            grouped[csv_paradigm][name] = meta

    # Generate and write each CSV
    for paradigm, datasets in grouped.items():
        if not datasets:
            print(f"  [{paradigm}] No datasets found, skipping.")
            continue
        generator = _GENERATORS[paradigm]
        df = generator(datasets)
        out_file = output_path / f"summary_{paradigm}.csv"
        df.to_csv(out_file, index=False)
        print(f"  [{paradigm}] Wrote {len(df)} datasets to {out_file}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate summary CSV tables from DATASET_METADATA_CATALOG."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write generated CSVs (default: moabb/datasets/)",
    )
    args = parser.parse_args()
    sys.exit(main(args.output_dir) or 0)
