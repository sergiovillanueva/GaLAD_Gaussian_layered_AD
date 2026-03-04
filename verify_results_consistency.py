"""Verify that GaLAD/results matches the paper artifacts and/or root output/.

This script is meant for reviewers (and for sanity checks before pushing to GitHub).
It compares:
- Raw results: results/{galad_results,baseline_results}.csv
- Aggregates: results/{results_per_class,results_per_benchmark}.csv
- Tables: results/tables/*
- Appendix: results/appendix/*

If a sibling folder "../output" exists (repo root output/), it also compares those
CSV files against GaLAD/results.

Usage:
  uv run verify_results_consistency.py
  python verify_results_consistency.py

Optional:
  --tolerance 1e-9   Numeric tolerance for value comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out = out.drop(columns=["timestamp"])
    for col in ("dataset", "model"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    for col in ("train_limit", "seed"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def compare_files(a: Path, b: Path) -> dict:
    return {
        "exists_a": a.exists(),
        "exists_b": b.exists(),
        "size_a": a.stat().st_size if a.exists() else None,
        "size_b": b.stat().st_size if b.exists() else None,
        "sha_a": sha256(a) if a.exists() else None,
        "sha_b": sha256(b) if b.exists() else None,
    }


def compare_raw_csv(a: Path, b: Path, *, sep: str, tolerance: float) -> tuple[bool, str]:
    if not a.exists() or not b.exists():
        return False, f"missing: {a.exists()} {b.exists()}"

    dfa = pd.read_csv(a, sep=sep)
    dfb = pd.read_csv(b, sep=sep)

    dfa = norm_df(dfa)
    dfb = norm_df(dfb)

    key = ["dataset", "model", "train_limit", "seed"]
    for k in key:
        if k not in dfa.columns or k not in dfb.columns:
            return False, f"missing key column: {k}"

    # Check duplicate keys
    dup_a = dfa.duplicated(subset=key).sum()
    dup_b = dfb.duplicated(subset=key).sum()
    if dup_a or dup_b:
        return False, f"duplicate keys (output={dup_a}, GaLAD={dup_b})"

    dfa = dfa.set_index(key).sort_index()
    dfb = dfb.set_index(key).sort_index()

    idx_a = set(dfa.index)
    idx_b = set(dfb.index)
    only_a = len(idx_a - idx_b)
    only_b = len(idx_b - idx_a)
    if only_a or only_b:
        return False, f"key mismatch: only_in_a={only_a} only_in_b={only_b}"

    # Compare intersection columns
    cols = [c for c in dfa.columns if c in dfb.columns]
    if not cols:
        return False, "no common columns"

    # Numeric compare where possible
    mismatches = 0
    checked = 0
    for c in cols:
        sa = pd.to_numeric(dfa[c], errors="coerce")
        sb = pd.to_numeric(dfb[c], errors="coerce")
        if sa.notna().any() and sb.notna().any():
            diff = (sa - sb).abs()
            bad = diff > tolerance
            mismatches += int(bad.sum())
            checked += int(diff.notna().sum())
        else:
            # Fallback: string equality
            aa = dfa[c].astype(str)
            bb = dfb[c].astype(str)
            bad = aa != bb
            mismatches += int(bad.sum())
            checked += len(bad)

    ok = mismatches == 0
    msg = f"checked={checked} mismatches={mismatches}"
    return ok, msg


def recompute_aggregates(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Import from local scripts to ensure same logic used in paper.
    sys.path.insert(0, str(results_dir.parent))
    from results_summary import load_results, compute_per_class, compute_per_benchmark  # type: ignore

    df_raw = load_results()
    df_class = compute_per_class(df_raw)
    df_bench = compute_per_benchmark(df_class)
    return df_class, df_bench


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    results_dir = here / "results"
    root_output = here.parent / "output"

    if not results_dir.exists():
        print(f"ERROR: results folder not found: {results_dir}")
        return 2

    print(f"Results dir: {results_dir}")
    print(f"Root output: {root_output} ({'found' if root_output.exists() else 'missing'})")

    # 1) Compare folder-to-folder CSVs if output exists
    targets = [
        "galad_results.csv",
        "baseline_results.csv",
        "results_per_class.csv",
        "results_per_benchmark.csv",
        "efficiency_comparison.csv",
    ]

    if root_output.exists():
        print("\n== Comparing main CSVs (output/ vs GaLAD/results/) ==")
        for name in targets:
            a = root_output / name
            b = results_dir / name
            meta = compare_files(a, b)
            same_hash = meta["sha_a"] == meta["sha_b"] and meta["sha_a"] is not None
            print(f"- {name}: sha_equal={same_hash} size_out={meta['size_a']} size_galad={meta['size_b']}")

        print("\n== Comparing raw CSV content (normalized) ==")
        for name in ("galad_results.csv", "baseline_results.csv"):
            ok, msg = compare_raw_csv(root_output / name, results_dir / name, sep=";", tolerance=args.tolerance)
            print(f"- {name}: ok={ok} {msg}")
            if not ok:
                return 1

        # Tables + appendix exact match (these are derived artifacts; should match bytewise)
        print("\n== Comparing derived artifacts (tables + appendix) ==")
        for sub in ("tables", "appendix"):
            out_dir = root_output / sub
            gal_dir = results_dir / sub
            if not out_dir.exists() or not gal_dir.exists():
                print(f"- {sub}: missing folder")
                return 1
            names_out = sorted([p.name for p in out_dir.glob("*.csv")])
            names_gal = sorted([p.name for p in gal_dir.glob("*.csv")])
            if names_out != names_gal:
                print(f"- {sub}: file list mismatch ({len(names_out)} vs {len(names_gal)})")
                return 1
            mism = 0
            mism_files = []
            for fname in names_out:
                pa = out_dir / fname
                pb = gal_dir / fname
                if sha256(pa) != sha256(pb):
                    mism += 1
                    mism_files.append(fname)
            print(f"- {sub}: files={len(names_out)} sha_mismatches={mism}")
            if mism_files:
                for fn in mism_files:
                    print(f"  - {fn}")
            if mism:
                return 1

    # 2) Self-consistency: recompute aggregates from raw and compare
    print("\n== Self-consistency (recompute aggregates from raw) ==")
    try:
        df_class_new, df_bench_new = recompute_aggregates(results_dir)
    except Exception as e:
        print(f"ERROR: failed to recompute aggregates: {type(e).__name__}: {e}")
        return 2

    # Load existing
    df_class_old = pd.read_csv(results_dir / "results_per_class.csv", sep=";")
    df_bench_old = pd.read_csv(results_dir / "results_per_benchmark.csv", sep=";")

    def key_sort(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        out = df.copy()
        for k in keys:
            if k in out.columns:
                out[k] = out[k].astype(str)
        return out.sort_values(keys).reset_index(drop=True)

    # Compare only stable columns that exist in both
    def df_equal(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> tuple[bool, str]:
        a = key_sort(a, keys)
        b = key_sort(b, keys)
        cols = [c for c in a.columns if c in b.columns]
        a = a[cols]
        b = b[cols]
        if len(a) != len(b):
            return False, f"row_mismatch {len(a)} vs {len(b)}"
        # Numeric tolerance
        mism = 0
        checked = 0
        for c in cols:
            sa = pd.to_numeric(a[c], errors="coerce")
            sb = pd.to_numeric(b[c], errors="coerce")
            if sa.notna().any() and sb.notna().any():
                diff = (sa - sb).abs()
                bad = diff > args.tolerance
                mism += int(bad.sum())
                checked += int(diff.notna().sum())
            else:
                bad = a[c].astype(str) != b[c].astype(str)
                mism += int(bad.sum())
                checked += len(bad)
        return (mism == 0), f"checked={checked} mismatches={mism}"

    ok1, msg1 = df_equal(df_class_old, df_class_new, ["dataset", "train_limit", "model"])
    ok2, msg2 = df_equal(df_bench_old, df_bench_new, ["benchmark", "train_limit", "model"])
    print(f"- results_per_class: ok={ok1} {msg1}")
    print(f"- results_per_benchmark: ok={ok2} {msg2}")

    if not (ok1 and ok2):
        return 1

    print("\nOK: results artifacts are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
