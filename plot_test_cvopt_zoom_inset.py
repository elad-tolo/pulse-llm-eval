"""Side-by-side accuracy plot with the zoomed view embedded as an
inset in the bottom-right of the main plot. Parametric over
family ∈ {bbh, MMLU} and dataset ∈ {0.01, 0.02, 0.03}.

Usage:
    python plot_test_cvopt_zoom_inset.py                     # bbh / 0.02
    python plot_test_cvopt_zoom_inset.py bbh 0.03
    python plot_test_cvopt_zoom_inset.py MMLU 0.02
    python plot_test_cvopt_zoom_inset.py MMLU 0.03

Saves to plots_<family>/<family>_cvopt_delta<X>_top1_zoom_inset.pdf
(append `--with-retrain` to switch the predictions-only curve from
imp_wor_noretrain to imp_wor and tag the filename).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

WITH_RETRAIN = "--with-retrain" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FAMILY = ARGS[0] if len(ARGS) > 0 else "bbh"
ARG = ARGS[1] if len(ARGS) > 1 else "0.02"
SUFFIX_PIECE = f"_delta{ARG}"
FAMILY_TOKEN = "" if FAMILY == "bbh" else f"_{FAMILY}"
OUT_DIR = f"plots_{FAMILY}"
os.makedirs(OUT_DIR, exist_ok=True)

X_FULL_MAX = 400_000
ZOOM_X = (20_000, 80_000)
N_PLOT_POINTS = 100


def _x_axis(z, mean_key):
    bs = int(z["wor_batch_size"])
    L = len(z[mean_key])
    return np.arange(L) * bs


def _load(path):
    return np.load(path) if os.path.exists(path) else None


std = _load(
    f"output_si/test_with_train_side__std_wor{FAMILY_TOKEN}"
    f"{SUFFIX_PIECE}_bs64_b400K.npz"
)
cv_w0 = _load(
    f"output_si/test_with_train_side__cv_wor{FAMILY_TOKEN}{SUFFIX_PIECE}"
    f"_bs64_b400K_initretrain_cvopt_warmup0.npz"
)
if WITH_RETRAIN:
    imp_choice = _load(
        f"output_si/test_with_train_side__imp_wor{FAMILY_TOKEN}{SUFFIX_PIECE}"
        f"_bs64_b400K_initretrain_cvopt_warmup0.npz"
    )
    imp_var = "imp_wor"
else:
    imp_choice = _load(
        f"output_si/test_with_train_side__imp_wor_noretrain{FAMILY_TOKEN}"
        f"{SUFFIX_PIECE}_bs64_b400K_cvopt_noretrain_warmup0.npz"
    )
    imp_var = "imp_wor_noretrain"
if std is None or cv_w0 is None or imp_choice is None:
    raise SystemExit("Missing one or more required npz inputs.")

UCBE = (
    "UCB-E",
    std["mean_hist_std_wor"], std["stderr_std_wor"],
    _x_axis(std, "mean_hist_std_wor"),
    "#55A868", "-",
)
PULSE = (
    "PULSE",
    cv_w0["mean_hist_cv_wor"], cv_w0["stderr_cv_wor"],
    _x_axis(cv_w0, "mean_hist_cv_wor"),
    "#4C72B0", "-",
)
PRED_ONLY = (
    "predictions only",
    imp_choice[f"mean_hist_{imp_var}"],
    imp_choice[f"stderr_{imp_var}"],
    _x_axis(imp_choice, f"mean_hist_{imp_var}"),
    "#C44E52", "-",
)
FULL_CURVES = [UCBE, PULSE, PRED_ONLY]
ZOOM_CURVES = [UCBE, PULSE]


def _draw_curves(ax, curves, x_lo, x_hi):
    for label, m, s, x_arr, color, ls in curves:
        L = min(len(x_arr), len(m))
        x, m, s = x_arr[:L], m[:L], s[:L]
        mask = (x >= x_lo) & (x <= x_hi)
        x_v, m_v, s_v = x[mask], m[mask], s[mask]
        if x_v.size == 0:
            continue
        step = max(1, len(x_v) // N_PLOT_POINTS)
        ax.plot(x_v[::step], m_v[::step], label=label,
                linewidth=1.6, color=color, linestyle=ls)
        ax.fill_between(
            x_v[::step],
            m_v[::step] - s_v[::step],
            m_v[::step] + s_v[::step],
            color=color, alpha=0.18,
        )


def _value_at(curve, x_target):
    _label, m, _s, x_arr, _color, _ls = curve
    L = min(len(x_arr), len(m))
    x_arr, m = x_arr[:L], m[:L]
    idx = int(np.argmin(np.abs(x_arr - x_target)))
    return float(m[idx])


def main():
    fig, ax_main = plt.subplots(figsize=(12, 7))

    _draw_curves(ax_main, FULL_CURVES, 0, X_FULL_MAX)
    ax_main.set_xlim(0, X_FULL_MAX)

    ax_main.set_xticks(
        [0, 100_000, 200_000, 300_000, 400_000]
    )
    ax_main.set_xticklabels(
        ["0", "100k", "200k", "300k", "400k"], fontsize=30
    )
    ax_main.tick_params(axis="y", labelsize=30)
    ax_main.set_xlabel("Budget", fontsize=38)
    ax_main.set_ylabel("Accuracy", fontsize=38)
    ax_main.grid(True, linestyle="--", alpha=0.5)

    ax_inset = ax_main.inset_axes([0.44, 0.14, 0.50, 0.50])
    _draw_curves(ax_inset, ZOOM_CURVES, *ZOOM_X)
    ax_inset.set_xlim(*ZOOM_X)

    rect_lo = _value_at(UCBE, ZOOM_X[0]) - 0.005
    rect_hi = _value_at(PULSE, ZOOM_X[1]) + 0.005
    ax_inset.set_ylim(rect_lo, rect_hi)

    ax_inset.tick_params(axis="both", labelsize=30)
    ax_inset.grid(True, linestyle="--", alpha=0.5)
    ax_inset.set_facecolor((1.0, 1.0, 1.0, 0.92))

    indicator = ax_main.indicate_inset_zoom(
        ax_inset, edgecolor="black", linewidth=1.4,
        linestyle="--", alpha=1.0,
    )
    connectors = getattr(indicator, "connectors", None)
    if connectors is None and isinstance(indicator, tuple):
        connectors = indicator[1]
    if connectors is not None:
        for c in connectors:
            c.set_visible(False)

    rect_xR = ZOOM_X[1]
    inset_xL = ZOOM_X[0]
    for y in (rect_lo, rect_hi):
        con = ConnectionPatch(
            xyA=(rect_xR, y), coordsA=ax_main.transData,
            xyB=(inset_xL, y), coordsB=ax_inset.transData,
            color="black", linewidth=1.1, linestyle="--",
            zorder=5,
        )
        fig.add_artist(con)

    plt.tight_layout()
    file_suffix = "_with_retrain" if WITH_RETRAIN else ""
    out_path = (
        f"{OUT_DIR}/{FAMILY}_cvopt_delta{ARG}"
        f"_top1_zoom_inset{file_suffix}.pdf"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
