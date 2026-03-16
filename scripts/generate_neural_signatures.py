#!/usr/bin/env python
"""Generate interactive neural signature HTML figures for all datasets.

Produces one set of HTML files per dataset in the output directory.

Usage (from repo root):
    PYTHONPATH=. python scripts/generate_neural_signatures.py
"""

import os
import sys
import traceback


# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from moabb.analysis.neural_signatures import generate_neural_signature
from moabb.datasets.utils import dataset_list


OUTPUT_DIR = os.path.join("docs", "source", "_static", "neural_signatures")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_ok = 0
    n_fail = 0

    for ds_cls in dataset_list:
        name = ds_cls.__name__
        print(f"  {name} ...", end=" ", flush=True)
        try:
            ds = ds_cls()
        except Exception:
            print("SKIP (cannot instantiate)")
            n_fail += 1
            continue

        try:
            paths = generate_neural_signature(ds, output_dir=OUTPUT_DIR)
            if paths:
                print(", ".join(p.name for p in paths))
                n_ok += 1
            else:
                print("no output")
        except Exception:
            traceback.print_exc()
            n_fail += 1

    print(f"\nDone: {n_ok} datasets with figures, {n_fail} skipped/failed")


if __name__ == "__main__":
    main()
