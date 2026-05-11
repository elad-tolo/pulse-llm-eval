"""CSV → binary observation matrix loaders for the test-with-train-side
pipeline.

Schema of the CSV files: index column = model id, optional metadata
columns `created_date` / `sha` (dropped on load), all remaining
columns = per-example correctness (∈ {0, 1, NaN}). NaN is treated as
0; values > 0 are treated as 1.
"""
import numpy as np
import pandas as pd
import torch


def _csv_to_binary(path):
    df = pd.read_csv(path, index_col=0)
    for c in ("created_date", "sha"):
        if c in df.columns:
            df = df.drop(columns=[c])
    M = (
        np.nan_to_num(df.values.astype(np.float64), nan=0.0) > 0
    ).astype(np.float64)
    return torch.tensor(M, dtype=torch.float32)


def load_train_test_stack(test_path="bbh/test.csv",
                          train_path="bbh/train.csv"):
    """Stack [M_train; M_test] vertically. Returns (M_full, test_pos)
    where test_pos is a LongTensor of row indices into M_full
    corresponding to the test rows (i.e. M_train.shape[0] ..
    M_full.shape[0])."""
    M_train = _csv_to_binary(train_path)
    M_test = _csv_to_binary(test_path)
    M_full = torch.cat([M_train, M_test], dim=0)
    test_pos = torch.arange(M_train.shape[0], M_full.shape[0])
    return M_full, test_pos
