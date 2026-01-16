#!/usr/bin/env python3
"""
Detect and optionally remove bad/suspicious values from results CSV.

Bad values include:
1. Markovian > Sum (violates upper bound)
2. Negative values (except near H=0.5 where value ≈ 0)
3. Zero values for Full when Sum/Markovian are non-zero (convergence failure)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def detect_bad_values(results_file: str = "results/all_results.csv",
                      model: str = None,
                      n: int = None,
                      alpha: float = None,
                      verbose: bool = True):
    """
    Detect rows with suspicious values.

    Returns DataFrame of bad rows.
    """
    df = pd.read_csv(results_file)

    # Convert value columns to numeric
    for col in ['value_sum', 'value_markovian', 'value_full']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Apply filters
    mask = pd.Series([True] * len(df))
    if model is not None:
        mask &= df['model'] == model
    if n is not None:
        mask &= df['n'] == n
    if alpha is not None:
        mask &= np.isclose(df['alpha'], alpha)

    df_filtered = df[mask].copy()

    bad_rows = []

    for idx, row in df_filtered.iterrows():
        H = row['H']
        v_sum = row['value_sum']
        v_mark = row['value_markovian']
        v_full = row['value_full']

        issues = []

        # Check 1: Markovian > Sum (violates upper bound)
        if pd.notna(v_mark) and pd.notna(v_sum):
            if v_mark > v_sum * 1.01 + 0.01:  # Small tolerance
                issues.append(f"mark({v_mark:.3f}) > sum({v_sum:.3f})")

        # Check 2: Negative Full when others are positive
        if pd.notna(v_full) and v_full < -0.01:
            if pd.notna(v_sum) and v_sum > 0.1:
                issues.append(f"full negative ({v_full:.3f})")

        # Check 3: Zero Full when others are large (convergence failure)
        if pd.notna(v_full) and abs(v_full) < 0.001:
            if pd.notna(v_sum) and v_sum > 0.5:
                issues.append(f"full≈0 but sum={v_sum:.3f}")

        # Check 4: Sudden jumps (would need neighbors to detect)

        if issues:
            bad_rows.append({
                'index': idx,
                'model': row['model'],
                'n': row['n'],
                'H': H,
                'alpha': row['alpha'],
                'value_sum': v_sum,
                'value_markovian': v_mark,
                'value_full': v_full,
                'issues': '; '.join(issues)
            })

    bad_df = pd.DataFrame(bad_rows)

    if verbose and len(bad_df) > 0:
        print(f"Found {len(bad_df)} rows with issues:")
        print("=" * 100)
        for _, row in bad_df.iterrows():
            print(f"  n={row['n']:4d}, H={row['H']:.2f}, alpha={row['alpha']:.1f}: {row['issues']}")
    elif verbose:
        print("No bad values detected.")

    return bad_df


def remove_bad_values(results_file: str = "results/all_results.csv",
                      bad_df: pd.DataFrame = None,
                      column: str = None,
                      backup: bool = True):
    """
    Remove or clear bad values from results CSV.

    Parameters
    ----------
    results_file : str
        Path to results CSV.
    bad_df : DataFrame
        DataFrame with 'index' column indicating rows to fix.
    column : str, optional
        If specified, only clear this column (e.g., 'value_markovian').
        If None, clear all value columns for bad rows.
    backup : bool
        If True, create backup before modifying.
    """
    if bad_df is None or len(bad_df) == 0:
        print("No bad values to remove.")
        return

    df = pd.read_csv(results_file)

    if backup:
        backup_file = results_file.replace('.csv', '_backup.csv')
        df.to_csv(backup_file, index=False)
        print(f"Backup saved to: {backup_file}")

    bad_indices = bad_df['index'].values

    if column is not None:
        # Clear only specified column
        df.loc[bad_indices, column] = ''
        print(f"Cleared {column} for {len(bad_indices)} rows.")
    else:
        # Clear all value columns
        for col in ['value_sum', 'value_markovian', 'value_full']:
            df.loc[bad_indices, col] = ''
        print(f"Cleared all value columns for {len(bad_indices)} rows.")

    df.to_csv(results_file, index=False)
    print(f"Updated: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and remove bad values from results")
    parser.add_argument("--results", type=str, default="results/all_results.csv")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--remove", action="store_true",
                       help="Remove bad values (clear columns)")
    parser.add_argument("--remove-column", type=str, default=None,
                       choices=['value_sum', 'value_markovian', 'value_full'],
                       help="Only clear this specific column")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup before removing")

    args = parser.parse_args()

    bad_df = detect_bad_values(
        results_file=args.results,
        model=args.model,
        n=args.n,
        alpha=args.alpha,
    )

    if args.remove and len(bad_df) > 0:
        print()
        remove_bad_values(
            results_file=args.results,
            bad_df=bad_df,
            column=args.remove_column,
            backup=not args.no_backup,
        )
        print("\nRerun jobs with --incremental to recompute cleared values.")
