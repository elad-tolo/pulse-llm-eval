"""Generate delta-controlled test subsets from the test matrix.

For each delta in `targets`, find the row whose binarised mean is
closest from below to (top_mean - delta), anchor it as #2, and fill
the remaining `m_subset - 2` rows with consecutive lower-mean arms.
Outputs preserve the source CSV's index and metadata columns.

Usage:
    python build_test_subsets.py --src bbh/test.csv
    python build_test_subsets.py --src MMLU/test.csv

Writes test_delta_<X>.csv next to the input.
"""

import os
import numpy as np
import pandas as pd


def generate_test_delta_subsets(
    targets=(0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10),
    m_subset=1000,
    src="bbh/test.csv",
):
    out_dir = os.path.dirname(src) or "."
    df = pd.read_csv(src, index_col=0)
    df_clean = df.copy()
    for c in ("created_date", "sha"):
        if c in df_clean.columns:
            df_clean = df_clean.drop(columns=[c])
    M = (np.nan_to_num(df_clean.values.astype(np.float64), nan=0.0) > 0).astype(
        np.float64
    )
    means = M.mean(axis=1)
    sorted_idx = np.argsort(means)[::-1]
    sorted_means = means[sorted_idx]

    for delta_target in targets:
        gaps = sorted_means[0] - sorted_means[1:]
        j = int(np.argmin(np.abs(gaps - delta_target))) + 1
        if j + (m_subset - 2) >= len(sorted_idx):
            print(
                f"[skip] delta={delta_target}: not enough lower-mean arms "
                f"(j={j}, need {m_subset - 2} more, have "
                f"{len(sorted_idx) - 1 - j})."
            )
            continue
        positions = [0, j] + list(range(j + 1, j + 1 + (m_subset - 2)))
        chosen = sorted_idx[positions]
        achieved = float(sorted_means[0] - sorted_means[j])

        out_df = df.iloc[chosen]
        out_path = os.path.join(out_dir, f"test_delta_{delta_target}.csv")
        out_df.to_csv(out_path)

        verify = (
            np.nan_to_num(
                out_df.drop(
                    columns=[c for c in ("created_date", "sha") if c in out_df.columns],
                    errors="ignore",
                ).values.astype(np.float64),
                nan=0.0,
            )
            > 0
        ).astype(np.float64)
        v_means = verify.mean(axis=1)
        v_sorted = np.sort(v_means)[::-1]
        v_gap = float(v_sorted[0] - v_sorted[1])
        print(
            f"delta_target={delta_target}: achieved={achieved:.4f}, "
            f"verify_gap={v_gap:.4f}, top5={v_sorted[:5].round(4).tolist()}, "
            f"saved {out_path}"
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--src", default="bbh/test.csv")
    args = p.parse_args()
    generate_test_delta_subsets(src=args.src)
