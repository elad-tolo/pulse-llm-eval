"""Bar chart of budget-to-95% across all 8 (suite, dataset) combos.

Reads the per-suite CSVs in plots_bbh/ and plots_MMLU/ written by
make_budget_to_95_csvs.py. One plot with 8 groups along X, each
group has UCB-E (green) and PULSE (blue) bars with bootstrap stderr
error bars (B=200) read from the CSV's `stderr` column.

Saved to both plots_bbh/budget_to_95_bars.pdf and
plots_MMLU/budget_to_95_bars.pdf.
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


SUITES = [
    ("bbh",  "plots_bbh",  "BBH"),
    ("MMLU", "plots_MMLU", "MMLU"),
]
DATASETS = ["test", "0.01", "0.02", "0.03"]


def _ds_label(ds):
    return "test" if ds == "test" else fr"$\Delta_{{\mathrm{{top2}}}}=0.0{ds[-1]}$"


def _read(csv_path, method):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == method:
                return row
    return {}


def _val_pair(row):
    """Return (budget, stderr) as floats, or (nan, 0) if 'never'/missing."""
    b = row.get("budget_to_95", "")
    s = row.get("stderr", "")
    try:
        budget = float(b)
    except (ValueError, TypeError):
        return float("nan"), 0.0
    try:
        stderr = float(s)
    except (ValueError, TypeError):
        stderr = 0.0
    return budget, stderr


def _plot(datasets, basename, figsize=(12, 6),
          annotate_distance=False):
    labels = []
    ucbe_b, ucbe_se = [], []
    pulse_b, pulse_se = [], []
    for suite, plot_dir, suite_label in SUITES:
        for ds in datasets:
            tag = "test" if ds == "test" else f"delta{ds}"
            csv_path = f"{plot_dir}/budget_to_95_{tag}.csv"
            ucbe = _read(csv_path, "UCB-E")
            pulse = _read(csv_path, "PULSE")
            b, s = _val_pair(ucbe)
            ucbe_b.append(b); ucbe_se.append(s)
            b, s = _val_pair(pulse)
            pulse_b.append(b); pulse_se.append(s)
            labels.append(f"{suite_label}\n{_ds_label(ds)}")

    n = len(labels)
    x = np.arange(n)
    bar_w = 0.4

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - bar_w / 2, ucbe_b, bar_w, yerr=ucbe_se, capsize=4,
           color="#9CA89C", label="UCB-E", edgecolor="black",
           linewidth=0.6, alpha=0.7)
    ax.bar(x + bar_w / 2, pulse_b, bar_w, yerr=pulse_se, capsize=4,
           color="#1565C0", label="PULSE", edgecolor="black",
           linewidth=0.6, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("LLM calls to reach 95%\naccuracy", fontsize=16)
    ax.tick_params(axis="y", labelsize=13)

    ymax = ax.get_ylim()[1]
    step = 50_000
    ticks = np.arange(0, ymax + step, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(t // 1000)}k" if t else "0" for t in ticks])

    # Suite separator (between BBH and MMLU halves).
    ax.axvline(x=len(datasets) - 0.5, color="0.6", linewidth=1.0,
               linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.95)

    if annotate_distance:
        for xi, ub, pb, pse in zip(x, ucbe_b, pulse_b, pulse_se):
            if not (np.isfinite(ub) and np.isfinite(pb)):
                continue
            saved = ub - pb
            if saved <= 0:
                continue
            anchor_y = pb + (pse if np.isfinite(pse) else 0.0)
            ax.annotate(
                f"{int(round(saved / 1000))}k\nsaved",
                xy=(xi + bar_w / 2, anchor_y),
                xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=12,
                color="black",
            )

    plt.tight_layout()
    for plot_dir in ("plots_bbh", "plots_MMLU"):
        os.makedirs(plot_dir, exist_ok=True)
        out_path = f"{plot_dir}/{basename}.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def _plot_horizontal(datasets, basename, figsize=(8, 4.2),
                     annotate_distance=False):
    """Rotated variant of `_plot` — bars go left→right instead of
    bottom→top. X axis is the budget value; Y axis is the
    (suite, dataset) labels."""
    labels = []
    ucbe_b, ucbe_se = [], []
    pulse_b, pulse_se = [], []
    for suite, plot_dir, suite_label in SUITES:
        for ds in datasets:
            tag = "test" if ds == "test" else f"delta{ds}"
            csv_path = f"{plot_dir}/budget_to_95_{tag}.csv"
            ucbe = _read(csv_path, "UCB-E")
            pulse = _read(csv_path, "PULSE")
            b, s = _val_pair(ucbe)
            ucbe_b.append(b); ucbe_se.append(s)
            b, s = _val_pair(pulse)
            pulse_b.append(b); pulse_se.append(s)
            labels.append(f"{suite_label}\n({_ds_label(ds)})")

    n = len(labels)
    y = np.arange(n)[::-1]
    bar_h = 0.4

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y + bar_h / 2, ucbe_b, bar_h, xerr=ucbe_se, capsize=4,
            color="#9CA89C", label="UCB-E", edgecolor="black",
            linewidth=0.6, alpha=0.7)
    ax.barh(y - bar_h / 2, pulse_b, bar_h, xerr=pulse_se, capsize=4,
            color="#1565C0", label="PULSE", edgecolor="black",
            linewidth=0.6, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, multialignment="center")
    ax.set_xlabel("LLM calls to reach 95%\naccuracy", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)

    xmax = ax.get_xlim()[1]
    xticks = [t for t in (0, 100_000, 200_000) if t <= xmax + 1e-6]
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [f"{int(t // 1000)}k" if t else "0" for t in xticks]
    )

    ax.axhline(y=n - len(datasets) - 0.5, color="0.6", linewidth=1.0,
               linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, fontsize=12, framealpha=0.95,
    )

    if annotate_distance:
        for yi, ub, pb, pse in zip(y, ucbe_b, pulse_b, pulse_se):
            if not (np.isfinite(ub) and np.isfinite(pb)) or ub <= 0:
                continue
            saved = ub - pb
            if saved <= 0:
                continue
            pct_saved = 100.0 * saved / ub
            anchor_x = pb + (pse if np.isfinite(pse) else 0.0)
            ax.annotate(
                f"{pct_saved:.0f}% saved",
                xy=(anchor_x, yi - bar_h / 2),
                xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=12,
                color="black",
            )

    plt.tight_layout()
    for plot_dir in ("plots_bbh", "plots_MMLU"):
        os.makedirs(plot_dir, exist_ok=True)
        out_path = f"{plot_dir}/{basename}.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def main():
    # Full chart: all 4 datasets per suite (8 groups).
    _plot(DATASETS, "budget_to_95_bars")
    # Subset: just delta=0.02 and delta=0.03 (4 groups).
    _plot(["0.02", "0.03"], "budget_to_95_bars_d02_d03",
          figsize=(8, 4.2), annotate_distance=True)
    # Rotated (horizontal) version of the d02_d03 subset.
    _plot_horizontal(
        ["0.02", "0.03"], "budget_to_95_bars_d02_d03_horizontal",
        figsize=(4.0, 4.3), annotate_distance=True,
    )


if __name__ == "__main__":
    main()
