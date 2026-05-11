"""Side-information orchestration for the test-with-train-side
pipeline.

Owns:
  - per-test stage-1 V cache (`_load_or_train_si_side_only`)
  - per-seed prewarm cache (`_si_prewarm_cache_path`,
    `_warm_train_nuclear_mf*`)
  - GPU-friendly MF training kernel (`_train_mf_device_local`)
  - per-seed worker (`run_single_variant_wor_si`) — dispatch limited
    to std_wor / cv_wor / imp_wor / imp_wor_noretrain

The `test_with_train_side.py` entry point owns warm-cache /
populate-prewarm / bandit / merge for this pipeline.
"""

import os
import copy
import hashlib
import multiprocessing
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from bandit import (
    NuclearNormMF,
    train_nuclear_mf,
    run_standard_ucbe_bandit_wo_replacement,
    run_cv_ucbe_bandit_wo_replacement,
    ALL_VARIANTS,
    VARIANT_KEYS,
)


# -------------------------------------------------------------------------
# MF disk caches
# -------------------------------------------------------------------------
MF_CACHE_DIR = "output_si/mf_cache"
PREWARM_CACHE_DIR = os.path.join(MF_CACHE_DIR, "prewarm")


def _si_prewarm_cache_path(seed, rank, n_epochs, prewarm_epochs,
                           prewarm_lr, prewarm_lam,
                           init_pulls_per_arm, batch_size,
                           test_path_id):
    """Per-seed cache filename for the U-only prewarm output."""
    key = (
        f"seed={seed}|rank={rank}|n_ep={n_epochs}|"
        f"pw_ep={prewarm_epochs}|pw_lr={prewarm_lr}|pw_lam={prewarm_lam}|"
        f"ipa={init_pulls_per_arm}|bs={batch_size}|tp={test_path_id}"
    )
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(PREWARM_CACHE_DIR, f"mf_prewarm_{h}.pt")


# -------------------------------------------------------------------------
# Side-only stage 1: train MF where test rows are fully unobserved.
# -------------------------------------------------------------------------
def _side_only_mf_cache_path(rank, n_epochs, test_path):
    key = (
        f"side_only|test={os.path.basename(test_path)}|"
        f"rank={rank}|epochs={n_epochs}"
    )
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(MF_CACHE_DIR, f"mf_side_only_{h}.pt")


def _train_mf_device_local(
    X, mask, rank, lr=0.1, n_epochs=5, lam=0.01,
    batch_size=131072, device="cpu", log_prefix="side-only", verbose=True,
):
    """GPU-friendly MF training. Pre-moves rows/cols/labels to `device`,
    shuffles + chunks on-device per epoch. Returns the trained model on CPU.
    """
    m, n = X.shape
    model = NuclearNormMF(m, n, rank).to(device)
    effective_wd = lam / (m + n)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=effective_wd,
    )
    bce = nn.BCEWithLogitsLoss()
    rows, cols = torch.where(mask)
    labels = X[rows, cols].float()
    rows = rows.to(device)
    cols = cols.to(device)
    labels = labels.to(device)
    n_obs = rows.numel()
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n_obs, device=device)
        total = 0.0
        n_batches = (n_obs + batch_size - 1) // batch_size
        for start in range(0, n_obs, batch_size):
            idx = perm[start : start + batch_size]
            r = rows[idx]
            c = cols[idx]
            y = labels[idx]
            optimizer.zero_grad()
            loss = bce(model(r, c), y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        if verbose:
            print(
                f"      [{log_prefix}] epoch {ep+1}/{n_epochs}, "
                f"loss={total / max(1, n_batches):.4f}"
            )
    return model.to("cpu")


def _load_or_train_si_side_only(
    X_full_full, test_pos_np,
    rank=100, n_epochs=5,
    lr=0.1, lam=0.01, mf_batch_size=4096,
    test_path="bbh/test.csv",
    device="cpu",
):
    """Stage 1 MF trained on side rows only (test rows fully unobserved).
    Returns (model, side_mask). Cache key omits the bandit seed; inside,
    fixed init seed 0 makes the trained model reproducible.
    """
    M_full, N = X_full_full.shape
    n_test = len(test_pos_np)
    cache_path = _side_only_mf_cache_path(rank, n_epochs, test_path)

    if os.path.exists(cache_path):
        try:
            blob = torch.load(cache_path, map_location="cpu")
        except Exception:
            blob = None
        if (
            blob is not None
            and blob.get("M_full") == M_full
            and blob.get("N") == N
            and blob.get("n_test") == n_test
        ):
            model = NuclearNormMF(M_full, N, rank)
            model.load_state_dict(blob["state_dict"])
            return model, blob["side_mask"]

    torch.manual_seed(0)
    np.random.seed(0)
    side_mask = torch.ones((M_full, N), dtype=torch.bool)
    side_mask[torch.tensor(test_pos_np, dtype=torch.long)] = False
    X_obs_full = X_full_full * side_mask

    if device == "cpu":
        with redirect_stdout(open(os.devnull, "w")):
            model = train_nuclear_mf(
                X_obs_full, side_mask,
                rank=rank, lr=lr, n_epochs=n_epochs,
                lam=lam, batch_size=mf_batch_size,
            )
    else:
        model = _train_mf_device_local(
            X_obs_full, side_mask,
            rank=rank, n_epochs=n_epochs,
            lr=lr, lam=lam, batch_size=131072, device=device,
        )

    os.makedirs(MF_CACHE_DIR, exist_ok=True)
    tmp_path = cache_path + f".tmp.{os.getpid()}"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "side_mask": side_mask,
            "M_full": M_full, "N": N, "n_test": n_test,
            "rank": rank, "n_epochs": n_epochs,
        },
        tmp_path,
    )
    os.replace(tmp_path, cache_path)
    return model, side_mask


# -------------------------------------------------------------------------
# Warm-start MF training (in-place fine-tune)
# -------------------------------------------------------------------------
def _warm_train_nuclear_mf(
    model, X, mask, lr=0.1, n_epochs=1, lam=0.01, batch_size=4096,
):
    """Fine-tune an existing NuclearNormMF model in place on (X, mask)."""
    m, n = X.shape
    effective_wd = lam / (m + n)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=effective_wd)

    rows, cols = torch.where(mask)
    labels = X[rows, cols].float()

    dataset = torch.utils.data.TensorDataset(rows, cols, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )

    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(n_epochs):
        for r, c, y in loader:
            optimizer.zero_grad()
            logits = model(r, c)
            loss = bce(logits, y)
            loss.backward()
            optimizer.step()
    return model


def _warm_train_nuclear_mf_device(
    model, X, mask, lr=0.1, n_epochs=1, lam=0.01,
    batch_size=131072, device="cpu",
):
    """GPU-friendly variant of `_warm_train_nuclear_mf`. With
    `model.V.requires_grad=False` only U updates."""
    m, n = X.shape
    model = model.to(device)
    effective_wd = lam / (m + n)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=effective_wd,
    )
    bce = nn.BCEWithLogitsLoss()
    rows, cols = torch.where(mask)
    labels = X[rows, cols].float()
    rows = rows.to(device)
    cols = cols.to(device)
    labels = labels.to(device)
    n_obs = rows.numel()
    model.train()
    for ep in range(n_epochs):
        perm = torch.randperm(n_obs, device=device)
        for start in range(0, n_obs, batch_size):
            idx = perm[start : start + batch_size]
            r = rows[idx]
            c = cols[idx]
            y = labels[idx]
            optimizer.zero_grad()
            loss = bce(model(r, c), y)
            loss.backward()
            optimizer.step()
    return model.to("cpu")


# -------------------------------------------------------------------------
# Per-variant worker (one seed)
# -------------------------------------------------------------------------
def run_single_variant_wor_si(args):
    """Side-information worker for one (variant, seed). Dispatch slimmed
    to the 4 variants the paper exercises:
      - std_wor             : UCB-E baseline (proxy not used)
      - cv_wor              : PULSE — CV-corrected UCB-E with online retrains
      - imp_wor             : Naive-Pooling — imputation with online retrains
      - imp_wor_noretrain   : Naive-Pooling with frozen post-prewarm proxy

    args layout:
        (variant_name, exp_idx, X_full_full, test_pos,
         actual_best_arm, top_k_arms, eps_arms,
         budget, batch_size, a, lam_cv, obs_prob)

    Returns: dict with keys variant, hist_top1, hist_topk, hist_eps,
        cells (or None), predicted_arm.
    """
    (
        variant_name,
        exp_idx,
        X_full_full,
        test_pos,
        actual_best_arm,
        top_k_arms,
        eps_arms,
        budget,
        batch_size,
        a,
        lam_cv,
        obs_prob,
    ) = args

    if variant_name not in ALL_VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant_name}. Choose from {ALL_VARIANTS}."
        )

    si_device = os.environ.get("SI_DEVICE", "cpu")
    torch.set_num_threads(1)

    seed = exp_idx * 10000
    torch.manual_seed(seed)
    np.random.seed(seed)

    M_full, N = X_full_full.shape
    if isinstance(test_pos, torch.Tensor):
        test_pos_np = test_pos.cpu().numpy()
    else:
        test_pos_np = np.asarray(test_pos, dtype=np.int64)
    n_test = len(test_pos_np)

    # Test-row mask. With obs_prob > 0 a Bernoulli mask; with obs_prob == 0
    # an init mask of `INIT_PULLS_PER_ARM * batch_size` cells per arm so
    # the worker can pre-warm U with enough data. The bandit runner
    # treats these mask cells as initial observations (counted in
    # init_T_sum, NOT in the post-mask budget).
    INIT_PULLS_PER_ARM = int(os.environ.get("SI_INIT_PULLS_PER_ARM", "1"))
    if obs_prob == 0:
        test_mask = torch.zeros((n_test, N), dtype=torch.bool)
        n_init_cells_per_arm = min(INIT_PULLS_PER_ARM * batch_size, N)
        for _arm in range(n_test):
            _cols = np.random.choice(
                N, size=n_init_cells_per_arm, replace=False,
            )
            test_mask[_arm, _cols] = True
    else:
        test_mask = torch.rand(n_test, N) < obs_prob

    X_full_test = X_full_full[test_pos_np]
    cells_history = None

    # std_wor doesn't use the proxy, so skip MF training entirely.
    if variant_name == "std_wor":
        with redirect_stdout(open(os.devnull, "w")):
            np.random.seed(seed)
            pred, hist = run_standard_ucbe_bandit_wo_replacement(
                X_full_test, budget=budget, batch_size=batch_size, a=a,
                mask=test_mask,
            )
        return {
            "variant": variant_name,
            "hist_top1": (hist == actual_best_arm).astype(float),
            "hist_topk": np.isin(hist, top_k_arms).astype(float),
            "hist_eps": np.isin(hist, eps_arms).astype(float),
            "cells": None,
            "predicted_arm": int(pred),
        }

    # MF hyperparameters: stage 1 trains on side rows only.
    MF_RANK = int(os.environ.get("SI_MF_RANK", "100"))
    MF_INIT_EPOCHS = int(os.environ.get("SI_MF_INIT_EPOCHS", "5"))
    SI_TEST_PATH_ID = os.environ.get(
        "SI_TEST_PATH_ID", "bbh/test.csv",
    )
    # Each retrain is a proper fine-tune (~10 epochs over the cumulative
    # bandit cells) so U[test rows] doesn't drift.
    MF_RETRAIN_EPOCHS = 10

    model, _ = _load_or_train_si_side_only(
        X_full_full, test_pos_np,
        rank=MF_RANK, n_epochs=MF_INIT_EPOCHS,
        test_path=SI_TEST_PATH_ID,
        device=si_device,
    )

    # Optional: re-init U[test_rows] from the per-dim Normal of U over
    # side rows. Enabled via SI_SMART_U_INIT=1.
    if os.environ.get("SI_SMART_U_INIT", "0") == "1":
        with torch.no_grad():
            U_data = model.U.data
            test_idx = torch.tensor(test_pos_np, dtype=torch.long)
            side_mask_idx = torch.ones(M_full, dtype=torch.bool)
            side_mask_idx[test_idx] = False
            U_side = U_data[side_mask_idx]
            u_mean = U_side.mean(dim=0)
            u_std = U_side.std(dim=0).clamp_min(1e-8)
            sampled = torch.normal(
                u_mean.unsqueeze(0).expand(len(test_pos_np), -1),
                u_std.unsqueeze(0).expand(len(test_pos_np), -1),
            )
            U_data[test_idx] = sampled

    # Freeze V for the rest of the run unless overridden.
    SI_FREEZE_V = os.environ.get("SI_FREEZE_V", "1") == "1"
    model.V.requires_grad_(not SI_FREEZE_V)

    # Stage 1.5: pre-warm U[test rows] on the per-arm init-mask cells
    # with V frozen. Without this, U[test rows] is at near-random init
    # for the first 1000 bandit steps.
    init_mask_full = torch.zeros((M_full, N), dtype=torch.bool)
    init_mask_full[torch.tensor(test_pos_np, dtype=torch.long)] = test_mask
    init_X_obs_full = X_full_full * init_mask_full
    MF_PREWARM_EPOCHS = int(os.environ.get("SI_PREWARM_EPOCHS", "10"))
    MF_PREWARM_LR = float(os.environ.get("SI_PREWARM_LR", "0.1"))
    MF_PREWARM_LAM = float(os.environ.get("SI_PREWARM_LAM", "0.01"))

    # Per-seed prewarm cache: deterministic in (seed, rank, n_epochs,
    # prewarm_*, init_pulls_per_arm, batch_size, test_path_id).
    prewarm_cache = _si_prewarm_cache_path(
        seed=seed, rank=MF_RANK, n_epochs=MF_INIT_EPOCHS,
        prewarm_epochs=MF_PREWARM_EPOCHS,
        prewarm_lr=MF_PREWARM_LR, prewarm_lam=MF_PREWARM_LAM,
        init_pulls_per_arm=INIT_PULLS_PER_ARM, batch_size=batch_size,
        test_path_id=SI_TEST_PATH_ID,
    )
    import sys as _sys  # for stderr prints (stdout is redirected below)
    prewarm_hit = False
    if os.path.exists(prewarm_cache):
        try:
            blob = torch.load(prewarm_cache, map_location="cpu")
        except Exception:
            blob = None
        if (
            blob is not None
            and blob.get("M_full") == M_full
            and blob.get("N") == N
            and blob.get("rank") == MF_RANK
        ):
            model.load_state_dict(blob["state_dict"])
            prewarm_hit = True
            _sys.stderr.write(
                f"[SI worker] prewarm cache HIT seed={seed} "
                f"path={prewarm_cache}\n"
            )

    if not prewarm_hit:
        with redirect_stdout(open(os.devnull, "w")):
            if si_device == "cpu":
                model = _warm_train_nuclear_mf(
                    model, init_X_obs_full, init_mask_full,
                    lr=MF_PREWARM_LR, n_epochs=MF_PREWARM_EPOCHS,
                    lam=MF_PREWARM_LAM, batch_size=4096,
                )
            else:
                model = _warm_train_nuclear_mf_device(
                    model, init_X_obs_full, init_mask_full,
                    lr=MF_PREWARM_LR, n_epochs=MF_PREWARM_EPOCHS,
                    lam=MF_PREWARM_LAM, batch_size=131072,
                    device=si_device,
                )
        os.makedirs(PREWARM_CACHE_DIR, exist_ok=True)
        tmp = prewarm_cache + f".tmp.{os.getpid()}"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "M_full": M_full, "N": N, "rank": MF_RANK,
            },
            tmp,
        )
        os.replace(tmp, prewarm_cache)
        _sys.stderr.write(
            f"[SI worker] prewarm cache MISS seed={seed} "
            f"saved={prewarm_cache}\n"
        )

    with redirect_stdout(open(os.devnull, "w")):
        model.eval()
        with torch.no_grad():
            logits = model.get_full_logits()
            probs_full = torch.sigmoid(logits)

        initial_model_state = copy.deepcopy(model.state_dict())

        X_pred_test = probs_full[test_pos_np]

        def make_retrain_callback(corr_trace_out=None):
            holder = [copy.deepcopy(model)]
            holder[0].load_state_dict(initial_model_state)
            holder[0].V.requires_grad_(not SI_FREEZE_V)
            x_full_test_np = (
                X_full_test.cpu().numpy() if corr_trace_out is not None else None
            )

            def retrain_callback(sampled_cols):
                # Stage 2: train U on cumulative observed test cells:
                # init-mask cells (where U was pre-warmed) + bandit-drawn cells.
                new_mask = init_mask_full.clone()
                for ti, cols in enumerate(sampled_cols):
                    if cols:
                        new_mask[int(test_pos_np[ti]), list(cols)] = True
                new_X_obs = X_full_full * new_mask
                if si_device == "cpu":
                    holder[0] = _warm_train_nuclear_mf(
                        holder[0],
                        new_X_obs, new_mask,
                        lr=0.1, n_epochs=MF_RETRAIN_EPOCHS,
                        lam=0.01, batch_size=4096,
                    )
                else:
                    holder[0] = _warm_train_nuclear_mf_device(
                        holder[0],
                        new_X_obs, new_mask,
                        lr=0.1, n_epochs=MF_RETRAIN_EPOCHS,
                        lam=0.01, batch_size=131072, device=si_device,
                    )
                holder[0].eval()
                with torch.no_grad():
                    logits_new = holder[0].get_full_logits()
                    probs_new_full = torch.sigmoid(logits_new)
                X_pred_new_test = probs_new_full[test_pos_np]
                mean_X_pred_new_test = (
                    X_pred_new_test.float().mean(dim=1).cpu().numpy()
                )

                if corr_trace_out is not None:
                    x_pred_test_np = X_pred_new_test.cpu().numpy()
                    rs = np.full(n_test, np.nan, dtype=np.float32)
                    for ti in range(n_test):
                        obs = sampled_cols[ti]
                        obs_mask = np.zeros(N, dtype=bool)
                        if obs:
                            obs_mask[list(obs)] = True
                        unobs = ~obs_mask
                        if unobs.sum() < 2:
                            continue
                        y = x_full_test_np[ti, unobs]
                        p = x_pred_test_np[ti, unobs]
                        if y.std() < 1e-12 or p.std() < 1e-12:
                            continue
                        rs[ti] = float(np.corrcoef(y, p)[0, 1])
                    corr_trace_out.append(rs)

                return X_pred_new_test, mean_X_pred_new_test, X_pred_new_test

            return retrain_callback

        SI_TRAIN_EVERY = int(os.environ.get("SI_TRAIN_EVERY", "1000"))
        import sys as _sys
        _sys.stderr.write(
            f"[SI worker] variant={variant_name} train_every={SI_TRAIN_EVERY} "
            f"(seed={seed})\n"
        )
        np.random.seed(seed)

        if variant_name == "cv_wor":
            # Trace logging for the per-seed CV diagnostics. Saved per
            # seed under SI_CV_WOR_TRACE_DIR (default output_si/cv_wor_traces).
            lam_trace, mu_trace, corr_trace = [], [], []
            cb = make_retrain_callback(corr_trace_out=corr_trace)
            pred, hist = run_cv_ucbe_bandit_wo_replacement(
                X_full_test, X_pred_test,
                budget=budget, batch_size=batch_size, a=a, lam_cv=lam_cv,
                mask=test_mask, scale_exploration=False,
                online_training=True, train_every=SI_TRAIN_EVERY,
                retrain_callback=cb,
                lam_cv_method="eb_optimal",
                lam_trace_out=lam_trace,
                mu_trace_out=mu_trace,
            )
            trace_dir = os.environ.get(
                "SI_CV_WOR_TRACE_DIR", "output_si/cv_wor_traces"
            )
            os.makedirs(trace_dir, exist_ok=True)
            np.savez(
                os.path.join(trace_dir, f"seed_{exp_idx}.npz"),
                lam_trace=np.asarray(lam_trace, dtype=np.float32),
                mu_trace=np.asarray(mu_trace, dtype=np.float32),
                corr_trace=np.asarray(corr_trace, dtype=np.float32),
                seed=seed,
            )

        elif variant_name == "imp_wor":
            cb = make_retrain_callback()
            pred, hist = run_cv_ucbe_bandit_wo_replacement(
                X_full_test, X_pred_test,
                budget=budget, batch_size=batch_size, a=a, lam_cv=lam_cv,
                mask=test_mask, scale_exploration=False,
                online_training=True, train_every=SI_TRAIN_EVERY,
                retrain_callback=cb, cv_enabled=False,
            )

        elif variant_name == "imp_wor_noretrain":
            # Imputation only, with the proxy frozen at its post-prewarm
            # state — no retrains during the bandit loop. Removes the
            # sawtooth that retrains induce in the imputation row estimator.
            pred, hist = run_cv_ucbe_bandit_wo_replacement(
                X_full_test, X_pred_test,
                budget=budget, batch_size=batch_size, a=a, lam_cv=lam_cv,
                mask=test_mask, scale_exploration=False,
                online_training=False, retrain_callback=None,
                cv_enabled=False,
            )

        else:
            raise RuntimeError(f"Variant dispatch missing: {variant_name}")

    return {
        "variant": variant_name,
        "hist_top1": (hist == actual_best_arm).astype(float),
        "hist_topk": np.isin(hist, top_k_arms).astype(float),
        "hist_eps": np.isin(hist, eps_arms).astype(float),
        "cells": cells_history,
        "predicted_arm": int(pred),
    }
