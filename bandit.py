"""Bandit runners for the test-with-train-side pipeline.

  - NuclearNormMF + train_nuclear_mf           (low-rank MF proxy)
  - compute_lam_cv_eb                          (EB-optimal CV coefficient)
  - run_standard_ucbe_bandit_wo_replacement    (UCB-E baseline)
  - run_cv_ucbe_bandit_wo_replacement          (PULSE / Naive-Pooling)
  - ALL_VARIANTS, VARIANT_KEYS                 (variant registry)
"""
import os
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# -------------------------------------------------------------------------
# Variant registry (slimmed to the 4 variants the paper uses)
# -------------------------------------------------------------------------
ALL_VARIANTS = [
    "std_wor",
    "cv_wor",
    "imp_wor",
    "imp_wor_noretrain",
]

VARIANT_KEYS = {
    "std_wor": {
        "mean": "mean_hist_std_wor",
        "stderr": "stderr_std_wor",
        "mean_topk": "mean_hist_std_wor_topk",
        "stderr_topk": "stderr_std_wor_topk",
        "mean_eps": "mean_hist_std_wor_eps",
        "stderr_eps": "stderr_std_wor_eps",
        "cells_key": None,
    },
    "cv_wor": {
        "mean": "mean_hist_cv_wor",
        "stderr": "stderr_cv_wor",
        "mean_topk": "mean_hist_cv_wor_topk",
        "stderr_topk": "stderr_cv_wor_topk",
        "mean_eps": "mean_hist_cv_wor_eps",
        "stderr_eps": "stderr_cv_wor_eps",
        "cells_key": None,
    },
    "imp_wor": {
        "mean": "mean_hist_imp_wor",
        "stderr": "stderr_imp_wor",
        "mean_topk": "mean_hist_imp_wor_topk",
        "stderr_topk": "stderr_imp_wor_topk",
        "mean_eps": "mean_hist_imp_wor_eps",
        "stderr_eps": "stderr_imp_wor_eps",
        "cells_key": None,
    },
    "imp_wor_noretrain": {
        "mean": "mean_hist_imp_wor_noretrain",
        "stderr": "stderr_imp_wor_noretrain",
        "mean_topk": "mean_hist_imp_wor_noretrain_topk",
        "stderr_topk": "stderr_imp_wor_noretrain_topk",
        "mean_eps": "mean_hist_imp_wor_noretrain_eps",
        "stderr_eps": "stderr_imp_wor_noretrain_eps",
        "cells_key": None,
    },
}


# -------------------------------------------------------------------------
# Low-rank matrix factorization proxy
# -------------------------------------------------------------------------
class NuclearNormMF(nn.Module):
    def __init__(self, m, n, rank=100):
        super().__init__()
        self.U = nn.Parameter(torch.randn(m, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(n, rank) * 0.01)

    def forward(self, row_idx, col_idx):
        u = self.U[row_idx]
        v = self.V[col_idx]
        logits = (u * v).sum(dim=1)
        return logits

    def get_full_logits(self):
        return self.U @ self.V.T


def train_nuclear_mf(X, mask, rank=100, lr=0.1, n_epochs=50, lam=1e-3, batch_size=2048):
    """Factorizes a binary matrix X = sigmoid(U @ V.T) using
    nuclear-norm regularization implemented as Adam weight_decay."""
    m, n = X.shape
    model = NuclearNormMF(m, n, rank)

    # Loss = BCE_loss + (lam / (m+n)) * 0.5 * (||U||^2 + ||V||^2)
    effective_wd = lam / (m + n)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=effective_wd)

    rows, cols = torch.where(mask)
    labels = X[rows, cols].float()

    dataset = torch.utils.data.TensorDataset(rows, cols, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    bce = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for r, c, y in loader:
            optimizer.zero_grad()
            logits = model(r, c)
            loss = bce(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(r)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss / len(rows):.4f}")

    return model


# -------------------------------------------------------------------------
# Empirical-Bernstein-optimal CV coefficient
# -------------------------------------------------------------------------
def compute_lam_cv_eb(F_t, S, Z_bar_prev):
    """Empirical-Bernstein optimal λ_cv. Closed-form minimizer:
        λ* = clip(1 − F_t · Z̄_{t-1} / (S + F_t²), 0, 1)
    with the trivial regime λ* = 1 whenever Z̄_{t-1} ≤ 0.
    """
    if Z_bar_prev <= 0.0:
        return 1.0
    denom = S + F_t * F_t
    if denom <= 0.0:
        return 1.0
    u_star = (F_t * Z_bar_prev) / denom
    return float(np.clip(1.0 - u_star, 0.0, 1.0))


# -------------------------------------------------------------------------
# Bandit runners
# -------------------------------------------------------------------------
def run_standard_ucbe_bandit_wo_replacement(
    X_full, budget=10000, batch_size=10, a=1.0, mask=None
):
    """Vanilla UCB-E (without replacement). Powers the `std_wor`
    variant — used as the UCB-E baseline in the paper."""
    print(
        f"\n--- Running Standard UCB-E Bandit Without Replacement (budget={budget}, batch_size={batch_size}, a={a}) ---"
    )
    m, n = X_full.shape
    T = np.zeros(m)
    sum_Y = np.zeros(m)

    sampled_cols = [set() for _ in range(m)]

    if mask is not None:
        mask_np = mask.cpu().numpy()
        for i in range(m):
            cols = np.where(mask_np[i])[0]
            sampled_cols[i].update(cols)
            if len(cols) > 0:
                Y = X_full[i, cols].float().numpy()
                sum_Y[i] += np.sum(Y)
                T[i] += len(cols)
            else:
                available_cols = list(set(range(n)) - sampled_cols[i])
                cols = np.random.choice(
                    available_cols,
                    size=min(batch_size, len(available_cols)),
                    replace=False,
                )
                sampled_cols[i].update(cols)
                Y = X_full[i, cols].float().numpy()
                sum_Y[i] += np.sum(Y)
                T[i] += len(cols)
    else:
        for i in range(m):
            available_cols = list(set(range(n)) - sampled_cols[i])
            cols = np.random.choice(
                available_cols, size=min(batch_size, len(available_cols)), replace=False
            )
            sampled_cols[i].update(cols)
            Y = X_full[i, cols].float().numpy()
            sum_Y[i] += np.sum(Y)
            T[i] += len(cols)

    row_estimates = sum_Y / T
    history = [np.argmax(row_estimates)]

    # Post-mask budget bookkeeping: count cells drawn during the bandit
    # loop, exclusive of the init mask. Override via SI_INIT_COUNT_BUDGET=1
    # to count init cells too.
    if os.environ.get("SI_INIT_COUNT_BUDGET", "0") == "1":
        init_T_sum = 0
    else:
        init_T_sum = int(T.sum())
    steps = max(0, budget // batch_size)

    for _ in range(steps):
        fully_known = np.array([len(s) == n for s in sampled_cols])
        for i in np.where(fully_known)[0]:
            row_estimates[i] = X_full[i].float().mean().item()

        if np.all(fully_known):
            history.append(np.argmax(row_estimates))
            continue

        ucbe = row_estimates + np.sqrt(a / T)
        ucbe[fully_known] = -np.inf

        best_arm = np.argmax(ucbe)

        available_cols = list(set(range(n)) - sampled_cols[best_arm])
        cols = np.random.choice(
            available_cols, size=min(batch_size, len(available_cols)), replace=False
        )
        sampled_cols[best_arm].update(cols)
        Y = X_full[best_arm, cols].float().numpy()

        sum_Y[best_arm] += np.sum(Y)
        T[best_arm] += len(cols)

        row_estimates[best_arm] = sum_Y[best_arm] / T[best_arm]
        history.append(np.argmax(row_estimates))

        if int(T.sum()) - init_T_sum >= budget:
            break

    predicted_best_arm = np.argmax(row_estimates)
    true_means = X_full.float().mean(dim=1).numpy()
    actual_best_arm = np.argmax(true_means)

    print(
        f"Predicted Best Arm: {predicted_best_arm} (Estimate: {row_estimates[predicted_best_arm]:.4f}, Pulls: {T[predicted_best_arm]})"
    )
    print(f"True Mean of Predicted Arm: {true_means[predicted_best_arm]:.4f}")
    return predicted_best_arm, np.array(history)


def run_cv_ucbe_bandit_wo_replacement(
    X_full,
    X_pred,
    budget=10000,
    batch_size=10,
    a=1.0,
    lam_cv=1.0,
    mask=None,
    scale_exploration=False,
    online_training=False,
    train_every=4000,
    retrain_callback=None,
    # If False: ablation that uses pure imputation
    #   mu_hat_i = (sum_{j in S} Y[i,j] + sum_{j not in S} X_pred[i,j]) / n
    # Sampling is uniform WoR. Powers the `imp_wor` / `imp_wor_noretrain`
    # variants ("Naive-Pooling" in the paper).
    cv_enabled=True,
    # "cov_var" (classical) or "eb_optimal" (closed-form EB minimizer).
    # "fixed" pins λ at lam_cv across all steps + retrains.
    lam_cv_method="cov_var",
    lam_trace_out=None,
    mu_trace_out=None,
):
    """CV-corrected UCB-E without replacement. Powers PULSE (`cv_wor`,
    cv_enabled=True) and Naive-Pooling (`imp_wor*`, cv_enabled=False)."""
    print(
        f"\n--- Running CV UCB-E Bandit W/O Replacement (budget={budget}, batch_size={batch_size}, a={a}, online={online_training}) ---"
    )
    m, n = X_full.shape
    T = np.zeros(m)

    sum_Y_obs = np.zeros(m)
    sum_Y_pred_obs = np.zeros(m)

    sum_est_Y = np.zeros(m)
    phase_sum_est_Y_comp = np.zeros(m)
    phase_sum_mu = np.zeros(m)
    # Predictable per-step CV correction: at each pull we add
    # lam_cv_arr[i] * (mean_X_pred[i] - est_Y_comp_step) using the
    # F_{t-1}-measurable lam, so the cumulative correction is a martingale-
    # difference sum (replaces post-hoc lam_cv_arr * phase_sum_*).
    phase_sum_lam_cv_term = np.zeros(m)
    total_cv_adj = np.zeros(m)
    pulls_count = np.zeros(m)

    if lam_cv_method == "fixed":
        lam_cv_arr = np.full(m, float(lam_cv))
    else:
        lam_cv_arr = np.zeros(m)

    cv_N = np.zeros(m)
    cv_sum_y = np.zeros(m)
    cv_sum_yc = np.zeros(m)
    cv_sum_y_yc = np.zeros(m)
    cv_sum_yc2 = np.zeros(m)
    cv_sum_y2 = np.zeros(m)

    Z_bar = np.zeros(m)
    Z_count = np.zeros(m, dtype=np.int64)

    sampled_cols = [set() for _ in range(m)]

    mean_X_pred = X_pred.float().mean(dim=1).cpu().numpy()

    def update_arm(i, cols):
        B = len(cols)
        N_unobs = n - len(sampled_cols[i])
        p = B / N_unobs if N_unobs > 0 else 1.0

        Y = X_full[i, cols].float().numpy()
        Y_comp = X_pred[i, cols].cpu().numpy()

        est_Y_step = (sum_Y_obs[i] + np.sum(Y) / p) / n
        est_Y_comp_step = (sum_Y_pred_obs[i] + np.sum(Y_comp) / p) / n

        sum_est_Y[i] += est_Y_step
        phase_sum_est_Y_comp[i] += est_Y_comp_step
        phase_sum_mu[i] += mean_X_pred[i]
        # Predictable lam: lam_cv_arr[i] is fit from samples 1..t-1 because
        # the refit at the end of the step runs after update_arm.
        phase_sum_lam_cv_term[i] += lam_cv_arr[i] * (mean_X_pred[i] - est_Y_comp_step)
        pulls_count[i] += 1

        sum_Y_obs[i] += np.sum(Y)
        sum_Y_pred_obs[i] += np.sum(Y_comp)

        T[i] += B

        for c, y, yc in zip(cols, Y, Y_comp):
            if mask_np is None or not mask_np[i, c]:
                cv_N[i] += 1
                cv_sum_y[i] += y
                cv_sum_y2[i] += y * y
                cv_sum_yc[i] += yc
                cv_sum_y_yc[i] += y * yc
                cv_sum_yc2[i] += yc * yc

    if mask is not None:
        mask_np = mask.cpu().numpy()
        for i in range(m):
            cols = np.where(mask_np[i])[0]
            sampled_cols[i].update(cols)
            if len(cols) > 0:
                B = len(cols)
                N_unobs = n
                p = B / N_unobs if N_unobs > 0 else 1.0

                Y = X_full[i, cols].float().numpy()
                Y_comp = X_pred[i, cols].cpu().numpy()

                est_Y_step = (np.sum(Y) / p) / n
                est_Y_comp_step = (np.sum(Y_comp) / p) / n

                sum_est_Y[i] += est_Y_step
                phase_sum_est_Y_comp[i] += est_Y_comp_step
                phase_sum_mu[i] += mean_X_pred[i]
                pulls_count[i] += 1

                sum_Y_obs[i] += np.sum(Y)
                sum_Y_pred_obs[i] += np.sum(Y_comp)
                T[i] += B
            else:
                available_cols = list(set(range(n)) - sampled_cols[i])
                cols = np.random.choice(
                    available_cols,
                    size=min(batch_size, len(available_cols)),
                    replace=False,
                )
                update_arm(i, cols)
                sampled_cols[i].update(cols)
    else:
        mask_np = np.zeros((m, n), dtype=bool)
        for i in range(m):
            available_cols = list(set(range(n)) - sampled_cols[i])
            cols = np.random.choice(
                available_cols, size=min(batch_size, len(available_cols)), replace=False
            )
            update_arm(i, cols)
            sampled_cols[i].update(cols)

    steps = max(0, budget // batch_size)
    # 10% warm-up by default; override via SI_WARMUP_STEPS env var.
    _warmup_env = os.environ.get("SI_WARMUP_STEPS")
    warmup_steps = int(_warmup_env) if _warmup_env is not None else int(steps * 0.1)

    a_cv = np.full(m, float(a))

    lam_cv_list = [np.mean(lam_cv_arr)]

    unobs_mask = ~mask_np
    exact_var_pred = np.zeros(m)
    for i in range(m):
        row_unobs = X_pred[i, unobs_mask[i]].cpu().numpy()
        if len(row_unobs) > 1:
            exact_var_pred[i] = np.var(row_unobs)

    # Per-arm proxy snapshots for the imputation row estimator
    # (cv_enabled=False). Refreshed on-pull only so a retrain doesn't
    # shake all arms' row estimates simultaneously.
    mean_X_pred_snapshot = mean_X_pred.copy()
    sum_Y_pred_obs_snapshot = sum_Y_pred_obs.copy()

    def compute_row_estimates(during_warmup):
        if during_warmup:
            return sum_Y_obs / np.maximum(T, 1)
        if cv_enabled:
            # Predictable form: phase_sum_lam_cv_term has each λ_s baked in
            # at step s (F_{s-1}-measurable), so the residual sequence is a
            # martingale-difference sum.
            return (
                sum_est_Y
                + total_cv_adj
                + phase_sum_lam_cv_term
            ) / np.maximum(pulls_count, 1)
        # Imputation estimator: (sum_obs Y + sum_unobs X_pred) / n.
        sum_unobs_pred = (
            n * mean_X_pred_snapshot - sum_Y_pred_obs_snapshot
        )
        return (sum_Y_obs + sum_unobs_pred) / n

    row_estimates = compute_row_estimates(during_warmup=True)
    history = [np.argmax(row_estimates)]
    if lam_trace_out is not None:
        lam_trace_out.append(lam_cv_arr.copy())
    if mu_trace_out is not None:
        mu_trace_out.append(row_estimates.copy())

    if os.environ.get("SI_INIT_COUNT_BUDGET", "0") == "1":
        init_T_sum = 0
    else:
        init_T_sum = int(T.sum())

    # Optional one-shot pre-loop retrain so X_pred reflects init-pull data
    # before the bandit's CV correction kicks in.
    if (
        os.environ.get("SI_RETRAIN_AFTER_INIT", "0") == "1"
        and online_training
        and retrain_callback is not None
    ):
        X_pred_new, mean_X_pred_new, _ = retrain_callback(sampled_cols)
        X_pred = (
            torch.tensor(X_pred_new, dtype=torch.float32)
            if not isinstance(X_pred_new, torch.Tensor)
            else X_pred_new
        )
        mean_X_pred = mean_X_pred_new
        sum_Y_pred_obs = np.zeros(m)
        for i in range(m):
            if len(sampled_cols[i]) > 0:
                sum_Y_pred_obs[i] = np.sum(
                    X_pred[i, list(sampled_cols[i])].numpy()
                )
        mean_X_pred_snapshot = mean_X_pred.copy()
        sum_Y_pred_obs_snapshot = sum_Y_pred_obs.copy()

    worker_id = (
        multiprocessing.current_process()._identity[0]
        if multiprocessing.current_process()._identity
        else 1
    )
    for step in tqdm(
        range(steps),
        position=worker_id,
        desc=f"Worker {worker_id}",
        leave=False,
        mininterval=1.0,
    ):
        fully_known = np.array([len(s) == n for s in sampled_cols])
        for i in np.where(fully_known)[0]:
            row_estimates[i] = X_full[i].float().mean().item()

        if np.all(fully_known):
            history.append(np.argmax(row_estimates))
            lam_cv_list.append(np.mean(lam_cv_arr))
            if lam_trace_out is not None:
                lam_trace_out.append(lam_cv_arr.copy())
            if mu_trace_out is not None:
                mu_trace_out.append(row_estimates.copy())
            continue

        ucbe = row_estimates + np.sqrt(a_cv / T)
        ucbe[fully_known] = -np.inf

        best_arm = np.argmax(ucbe)

        available_cols = list(set(range(n)) - sampled_cols[best_arm])
        N_avail_uniform = len(available_cols)
        cols = np.random.choice(
            available_cols, size=min(batch_size, N_avail_uniform), replace=False
        )

        # eb_optimal: compute λ_{cv,t} from pool sums F_t, S and Z̄_{t-1}.
        # Uniform-WoR has constant inclusion π = B/|U|, so
        # S = ((1-π)/π) · Σ f².
        if (
            cv_enabled
            and lam_cv_method == "eb_optimal"
            and step >= warmup_steps
            and N_avail_uniform > 0
        ):
            available_cols_arr = np.asarray(available_cols)
            f_avail = X_pred[best_arm, available_cols_arr].cpu().numpy().astype(np.float64)
            B_eff = float(len(cols))
            pi_uniform = B_eff / float(N_avail_uniform)
            pi_uniform = max(min(pi_uniform, 1.0), 1e-12)
            F_t = float(np.sum(f_avail))
            S_eb = float(((1.0 - pi_uniform) / pi_uniform) * np.sum(f_avail * f_avail))
            lam_cv_arr[best_arm] = compute_lam_cv_eb(F_t, S_eb, float(Z_bar[best_arm]))

        update_arm(best_arm, cols)
        sampled_cols[best_arm].update(cols)

        # Refresh just-pulled arm's imputation snapshot.
        mean_X_pred_snapshot[best_arm] = mean_X_pred[best_arm]
        sum_Y_pred_obs_snapshot[best_arm] = float(np.sum(
            X_pred[best_arm, list(sampled_cols[best_arm])].cpu().numpy()
        ))

        # eb_optimal: update Z_bar with this step's HT residual.
        if (
            cv_enabled
            and lam_cv_method == "eb_optimal"
            and step >= warmup_steps
            and N_avail_uniform > 0
        ):
            B_eff = float(len(cols))
            pi_uniform = max(min(B_eff / float(N_avail_uniform), 1.0), 1e-12)
            Y_raw_batch = X_full[best_arm, cols].float().numpy().astype(np.float64)
            f_batch = X_pred[best_arm, cols].cpu().numpy().astype(np.float64)
            Z_t = float(
                np.sum(Y_raw_batch - lam_cv_arr[best_arm] * f_batch) / pi_uniform
            )
            n_prev = int(Z_count[best_arm])
            Z_bar[best_arm] = (n_prev * float(Z_bar[best_arm]) + Z_t) / (n_prev + 1)
            Z_count[best_arm] = n_prev + 1

        if cv_enabled and lam_cv_method == "cov_var" and step == warmup_steps:
            for i in range(m):
                if cv_N[i] > 10:
                    v_pred = exact_var_pred[i]
                    if v_pred > 1e-8:
                        cov_xy = (
                            cv_sum_y_yc[i] - (cv_sum_y[i] * cv_sum_yc[i]) / cv_N[i]
                        ) / (cv_N[i] - 1)
                        lam_cv_arr[i] = np.clip(cov_xy / v_pred, 0.0, 1.0)

                        var_y = (cv_sum_y2[i] - (cv_sum_y[i] ** 2) / cv_N[i]) / (
                            cv_N[i] - 1
                        )
                        if var_y > 1e-8:
                            rho_sq = (cov_xy**2) / (var_y * v_pred)
                            if scale_exploration:
                                a_cv[i] = a * max(0.5, 1.0 - rho_sq)
        elif cv_enabled and lam_cv_method == "cov_var" and step > warmup_steps:
            if cv_N[best_arm] > 10:
                v_pred = exact_var_pred[best_arm]
                if v_pred > 1e-8:
                    cov_xy = (
                        cv_sum_y_yc[best_arm]
                        - (cv_sum_y[best_arm] * cv_sum_yc[best_arm]) / cv_N[best_arm]
                    ) / (cv_N[best_arm] - 1)
                    lam_cv_arr[best_arm] = np.clip(cov_xy / v_pred, 0.0, 1.0)

                    var_y = (
                        cv_sum_y2[best_arm] - (cv_sum_y[best_arm] ** 2) / cv_N[best_arm]
                    ) / (cv_N[best_arm] - 1)
                    if var_y > 1e-8:
                        rho_sq = (cov_xy**2) / (var_y * v_pred)
                        if scale_exploration:
                            a_cv[best_arm] = a * max(0.5, 1.0 - rho_sq)

        lam_cv_list.append(float(np.mean(lam_cv_arr)))
        if lam_trace_out is not None:
            lam_trace_out.append(lam_cv_arr.copy())

        row_estimates = compute_row_estimates(during_warmup=(step < warmup_steps))

        fully_known_current = np.array([len(s) == n for s in sampled_cols])
        for i in np.where(fully_known_current)[0]:
            row_estimates[i] = X_full[i].float().mean().item()

        history.append(np.argmax(row_estimates))
        if mu_trace_out is not None:
            mu_trace_out.append(row_estimates.copy())

        # Retrain at the END of the loop body, after row estimate /
        # history / traces for step t have been finalized. Makes it
        # textually obvious that the new MF can only enter at step t+1.
        if (
            online_training
            and (step > 0)
            and (step % train_every == 0)
            and retrain_callback is not None
        ):
            # Commit the predictable per-step CV corrections accumulated
            # this phase.
            total_cv_adj += phase_sum_lam_cv_term

            phase_sum_est_Y_comp = np.zeros(m)
            phase_sum_mu = np.zeros(m)
            phase_sum_lam_cv_term = np.zeros(m)
            cv_N = np.zeros(m)
            cv_sum_y = np.zeros(m)
            cv_sum_yc = np.zeros(m)
            cv_sum_y_yc = np.zeros(m)
            cv_sum_yc2 = np.zeros(m)
            cv_sum_y2 = np.zeros(m)
            if lam_cv_method == "fixed":
                lam_cv_arr = np.full(m, float(lam_cv))
            else:
                lam_cv_arr = np.zeros(m)
            Z_bar = np.zeros(m)
            Z_count = np.zeros(m, dtype=np.int64)
            if scale_exploration:
                a_cv = np.full(m, float(a))

            X_pred_new, mean_X_pred_new, probs = retrain_callback(sampled_cols)
            X_pred = (
                torch.tensor(X_pred_new, dtype=torch.float32)
                if not isinstance(X_pred_new, torch.Tensor)
                else X_pred_new
            )
            mean_X_pred = mean_X_pred_new

            for i in range(m):
                row_unobs = X_pred[i, unobs_mask[i]].cpu().numpy()
                if len(row_unobs) > 1:
                    exact_var_pred[i] = np.var(row_unobs)

            # Imputation snapshots are deliberately NOT touched here.
            # cv_wor's row estimator never reads sum_Y_pred_obs, so
            # leaving it stale is harmless.

        if int(T.sum()) - init_T_sum >= budget:
            break

    predicted_best_arm = np.argmax(row_estimates)
    true_means = X_full.float().mean(dim=1).numpy()
    actual_best_arm = np.argmax(true_means)

    print(
        f"Predicted Best Arm: {predicted_best_arm} (Estimate: {row_estimates[predicted_best_arm]:.4f}, Pulls: {T[predicted_best_arm]})"
    )
    print(
        f"Actual Best Arm:    {actual_best_arm} (True Mean: {true_means[actual_best_arm]:.4f}, Pulls: {T[actual_best_arm]})"
    )
    print(f"True Mean of Predicted Arm: {true_means[predicted_best_arm]:.4f}")
    return predicted_best_arm, np.array(history)


