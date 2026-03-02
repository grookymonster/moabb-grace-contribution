#!/usr/bin/env python3
"""Validate auto-generated summary CSVs against static ground-truth CSVs.

Compares generated CSVs (from generate_summary_tables.py) against the
manually-maintained static CSVs cell-by-cell and produces a divergence report.

Usage
-----
    python scripts/validate_summary_tables.py \\
        --generated-dir /tmp/generated/ \\
        [--static-dir moabb/datasets/] \\
        [--report divergence_report.csv] \\
        [--strict]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


PARADIGMS = ["imagery", "p300", "ssvep", "cvep", "rstate"]

# Column to exclude from comparison (no longer maintained)
_EXCLUDE_COLUMNS = {"PapersWithCode leaderboard"}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize_numeric(s):
    """Try to parse a string as a number for comparison.

    Returns the number if parseable, else None.
    """
    if s is None or s == "":
        return None
    try:
        val = float(s)
        return val
    except (ValueError, TypeError):
        return None


def _normalize_str(s):
    """Normalize a string for comparison: strip, collapse spaces."""
    if s is None:
        return ""
    s = str(s).strip()
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def _values_match(static_val, generated_val):
    """Compare two cell values with tolerance.

    Returns (match: bool, severity: str).
    Severity is one of: 'match', 'format_diff', 'mismatch'.
    """
    s = _normalize_str(static_val)
    g = _normalize_str(generated_val)

    # Exact string match after normalization
    if s == g:
        return True, "match"

    # Both empty
    if not s and not g:
        return True, "match"

    # Numeric comparison with tolerance
    s_num = _normalize_numeric(s)
    g_num = _normalize_numeric(g)
    if s_num is not None and g_num is not None:
        if abs(s_num - g_num) < 0.01:
            # Same value, possibly different format (e.g., "512.0" vs "512")
            if s != g:
                return True, "format_diff"
            return True, "match"

    # Try comparing as ints (e.g., "512" vs "512.0" where one parsed as int)
    if s_num is not None and g_num is not None:
        return False, "mismatch"

    # String mismatch
    return False, "mismatch"


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_paradigm(paradigm, static_dir, generated_dir):
    """Compare one paradigm's CSVs.

    Returns a list of divergence dicts.
    """
    static_file = Path(static_dir) / f"summary_{paradigm}.csv"
    generated_file = Path(generated_dir) / f"summary_{paradigm}.csv"

    if not static_file.exists():
        print(f"  [{paradigm}] Static CSV not found: {static_file}")
        return []
    if not generated_file.exists():
        print(f"  [{paradigm}] Generated CSV not found: {generated_file}")
        return []

    static_df = pd.read_csv(
        static_file, header=0, skipinitialspace=True, dtype=str, keep_default_na=False
    )
    generated_df = pd.read_csv(
        generated_file, header=0, skipinitialspace=True, dtype=str, keep_default_na=False
    )

    # Set Dataset as index
    if "Dataset" in static_df.columns:
        static_df = static_df.set_index("Dataset")
    if "Dataset" in generated_df.columns:
        generated_df = generated_df.set_index("Dataset")

    # Strip whitespace from index
    static_df.index = static_df.index.str.strip()
    generated_df.index = generated_df.index.str.strip()

    # Drop empty-index rows
    static_df = static_df[static_df.index != ""]
    generated_df = generated_df[generated_df.index != ""]

    # Determine columns to compare (exclude dropped columns)
    static_cols = [c for c in static_df.columns if c.strip() not in _EXCLUDE_COLUMNS]
    generated_cols = [
        c for c in generated_df.columns if c.strip() not in _EXCLUDE_COLUMNS
    ]

    # Build column mapping (strip whitespace for alignment)
    static_col_map = {c.strip(): c for c in static_cols}
    generated_col_map = {c.strip(): c for c in generated_cols}
    common_cols = set(static_col_map.keys()) & set(generated_col_map.keys())

    divergences = []

    # Report column differences
    static_only_cols = set(static_col_map.keys()) - set(generated_col_map.keys())
    gen_only_cols = set(generated_col_map.keys()) - set(static_col_map.keys())
    for col in sorted(static_only_cols):
        divergences.append(
            {
                "paradigm": paradigm,
                "dataset": "*",
                "column": col,
                "static_value": "(column exists)",
                "generated_value": "(column missing)",
                "severity": "missing_in_generated",
            }
        )
    for col in sorted(gen_only_cols):
        divergences.append(
            {
                "paradigm": paradigm,
                "dataset": "*",
                "column": col,
                "static_value": "(column missing)",
                "generated_value": "(column exists)",
                "severity": "missing_in_static",
            }
        )

    # Datasets in static but not generated
    for ds in sorted(set(static_df.index) - set(generated_df.index)):
        divergences.append(
            {
                "paradigm": paradigm,
                "dataset": ds,
                "column": "*",
                "static_value": "(exists)",
                "generated_value": "(missing)",
                "severity": "missing_in_generated",
            }
        )

    # Datasets in generated but not static
    for ds in sorted(set(generated_df.index) - set(static_df.index)):
        divergences.append(
            {
                "paradigm": paradigm,
                "dataset": ds,
                "column": "*",
                "static_value": "(missing)",
                "generated_value": "(exists)",
                "severity": "missing_in_static",
            }
        )

    # Cell-by-cell comparison for common datasets and columns
    common_datasets = sorted(set(static_df.index) & set(generated_df.index))
    for ds in common_datasets:
        for col_name in sorted(common_cols):
            s_col = static_col_map[col_name]
            g_col = generated_col_map[col_name]
            s_val = (
                str(static_df.loc[ds, s_col]).strip()
                if s_col in static_df.columns
                else ""
            )
            g_val = (
                str(generated_df.loc[ds, g_col]).strip()
                if g_col in generated_df.columns
                else ""
            )
            match, severity = _values_match(s_val, g_val)
            if not match or severity == "format_diff":
                divergences.append(
                    {
                        "paradigm": paradigm,
                        "dataset": ds,
                        "column": col_name,
                        "static_value": s_val,
                        "generated_value": g_val,
                        "severity": severity,
                    }
                )

    return divergences


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(all_divergences):
    """Print a summary table of divergence counts per paradigm."""
    print("\n" + "=" * 70)
    print("DIVERGENCE SUMMARY")
    print("=" * 70)

    # Group by paradigm and severity
    summary = {}
    for d in all_divergences:
        p = d["paradigm"]
        s = d["severity"]
        summary.setdefault(p, {})
        summary[p][s] = summary[p].get(s, 0) + 1

    header = f"{'Paradigm':<12} {'mismatch':>10} {'format_diff':>12} {'missing_gen':>12} {'missing_static':>14} {'total':>8}"
    print(header)
    print("-" * len(header))

    total_mismatches = 0
    for paradigm in PARADIGMS:
        counts = summary.get(paradigm, {})
        mm = counts.get("mismatch", 0)
        fd = counts.get("format_diff", 0)
        mg = counts.get("missing_in_generated", 0)
        ms = counts.get("missing_in_static", 0)
        total = mm + fd + mg + ms
        total_mismatches += mm
        print(f"{paradigm:<12} {mm:>10} {fd:>12} {mg:>12} {ms:>14} {total:>8}")

    print("-" * len(header))
    total = len(all_divergences)
    print(f"{'TOTAL':<12} {total_mismatches:>10} {'':>12} {'':>12} {'':>14} {total:>8}")
    print()

    return total_mismatches


def print_details(all_divergences, max_rows=50):
    """Print detailed divergence report."""
    mismatches = [d for d in all_divergences if d["severity"] == "mismatch"]
    if mismatches:
        print(f"\nMISMATCHES ({len(mismatches)} total, showing up to {max_rows}):")
        print("-" * 100)
        fmt = "{paradigm:<10} {dataset:<30} {column:<22} {static_value:<20} {generated_value:<20}"
        print(
            fmt.format(
                paradigm="Paradigm",
                dataset="Dataset",
                column="Column",
                static_value="Static",
                generated_value="Generated",
            )
        )
        print("-" * 100)
        for d in mismatches[:max_rows]:
            print(
                fmt.format(
                    paradigm=d["paradigm"],
                    dataset=d["dataset"],
                    column=d["column"],
                    static_value=d["static_value"][:20],
                    generated_value=d["generated_value"][:20],
                )
            )
        if len(mismatches) > max_rows:
            print(f"  ... and {len(mismatches) - max_rows} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(generated_dir, static_dir=None, report_file=None, strict=False):
    if static_dir is None:
        static_dir = str(Path(__file__).resolve().parent.parent / "moabb" / "datasets")

    print(f"Static CSVs:    {static_dir}")
    print(f"Generated CSVs: {generated_dir}")
    print()

    all_divergences = []
    for paradigm in PARADIGMS:
        divs = compare_paradigm(paradigm, static_dir, generated_dir)
        all_divergences.extend(divs)
        n_match = sum(1 for d in divs if d["severity"] == "mismatch")
        n_total = len(divs)
        status = "OK" if n_match == 0 else f"{n_match} mismatches"
        print(f"  [{paradigm}] {n_total} divergences ({status})")

    # Summary
    n_mismatches = print_summary(all_divergences)
    print_details(all_divergences)

    # Write report CSV
    if all_divergences:
        report_path = Path(report_file) if report_file else Path("divergence_report.csv")
        report_df = pd.DataFrame(all_divergences)
        report_df.to_csv(report_path, index=False)
        print(f"\nDivergence report written to: {report_path}")

    # Exit code
    if strict and n_mismatches > 0:
        print(f"\nFAILED: {n_mismatches} mismatches found (strict mode).")
        return 1
    elif n_mismatches > 0:
        print(f"\nWARNING: {n_mismatches} mismatches found.")
    else:
        print("\nAll generated values match static CSVs (excluding format differences).")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate generated summary CSVs against static ground-truth CSVs."
    )
    parser.add_argument(
        "--generated-dir",
        required=True,
        help="Directory containing generated CSVs.",
    )
    parser.add_argument(
        "--static-dir",
        default=None,
        help="Directory containing static CSVs (default: moabb/datasets/).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path for divergence report CSV (default: divergence_report.csv).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any mismatches found.",
    )
    args = parser.parse_args()
    sys.exit(main(args.generated_dir, args.static_dir, args.report, args.strict))
