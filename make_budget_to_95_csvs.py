"""Per-(family, dataset) CSV listing the budget each method needs to
first reach 95% top-1 accuracy. One row per method, with bootstrap
mean and stderr (B=200 seed resamples)."""
import csv
import os
import numpy as np

DATASETS = ["test", "0.01", "0.02", "0.03"]
THRESHOLD = 0.95
B_BOOT = 200


def _x_axis(z, mean_key):
    bs = int(z["wor_batch_size"])
    L = len(z[mean_key])
    return np.arange(L) * bs


def _bootstrap_cells_to(threshold, raw_top1, x_cells, B=200, seed=0):
    rng = np.random.default_rng(seed)
    n_seeds = raw_top1.shape[0]
    crossings = []
    for _ in range(B):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        m = raw_top1[idx].mean(axis=0)
        above = np.where(m >= threshold)[0]
        if above.size:
            crossings.append(int(x_cells[min(above[0], len(x_cells) - 1)]))
    if not crossings:
        return None, None
    arr = np.asarray(crossings, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def _path(family, variant, dataset, suffix_tail):
    family_token = "" if family == "bbh" else f"_{family}"
    if dataset == "test":
        if family == "MMLU" and variant == "std_wor":
            ds_piece = "_test"
        else:
            ds_piece = ""
    else:
        ds_piece = f"_delta{dataset}"
    return (
        f"output_si/test_with_train_side__{variant}{family_token}{ds_piece}"
        f"_{suffix_tail}.npz"
    )


METHODS = [
    ("UCB-E",            "std_wor",            "bs64_b400K"),
    ("PULSE",            "cv_wor",
     "bs64_b400K_initretrain_cvopt_warmup0"),
    ("predictions only", "imp_wor_noretrain",
     "bs64_b400K_cvopt_noretrain_warmup0"),
]


def _budget_to_95(family, dataset, variant, suffix_tail):
    """Returns (mean_budget_to_95, stderr) or (None, None) if the
    method never crosses 95% (or the npz is missing)."""
    path = _path(family, variant, dataset, suffix_tail)
    if not os.path.exists(path):
        return None, None
    z = np.load(path)
    raw_key = f"raw_top1_{variant}"
    if raw_key not in z.files:
        return None, None
    raw = z[raw_key]
    x = _x_axis(z, f"mean_hist_{variant}")
    L = min(len(x), raw.shape[1])
    return _bootstrap_cells_to(
        THRESHOLD, raw[:, :L], x[:L], B=B_BOOT,
    )


def main():
    for family in ("bbh", "MMLU"):
        out_dir = f"plots_{family}"
        os.makedirs(out_dir, exist_ok=True)
        for ds in DATASETS:
            tag = "test" if ds == "test" else f"delta{ds}"
            measurements = {
                label: _budget_to_95(family, ds, variant, suffix_tail)
                for label, variant, suffix_tail in METHODS
            }
            ucbe_mean = measurements["UCB-E"][0]
            rows = []
            for label, _v, _s in METHODS:
                mean_c, sd_c = measurements[label]
                if mean_c is None:
                    rows.append([label, "never", "", "", ""])
                    continue
                if ucbe_mean is None or mean_c == 0:
                    ratio_str = ""
                    abs_str = ""
                else:
                    ratio_str = f"{ucbe_mean / mean_c:.2f}"
                    abs_str = str(int(round(ucbe_mean - mean_c)))
                rows.append([
                    label, int(round(mean_c)),
                    int(round(sd_c)) if sd_c is not None else "",
                    ratio_str, abs_str,
                ])
            csv_path = f"{out_dir}/budget_to_95_{tag}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    ["method", "budget_to_95", "stderr",
                     "speedup_vs_ucbe", "speedup_abs_cells"]
                )
                w.writerows(rows)
            print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
