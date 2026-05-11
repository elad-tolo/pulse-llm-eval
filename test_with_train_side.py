"""Test-pool bandit with a train matrix as side information.

Stacks [M_train; M_test] into a (n_train + n_test) x N matrix, trains
stage-1 V on M_train only, then runs one of {std_wor, imp_wor, cv_wor,
imp_wor_noretrain} on the M_test rows with V frozen across all retrains.

Modes:
  --warm-cache           : train and cache stage-1 V once, then exit.
  --variant <name>       : run the bandit (per-seed serial GPU) for one variant.
  --merge --variant <n>  : merge per-chunk npz files into the canonical npz.
"""
import argparse
import glob
import os

import numpy as np
import torch

from data_loaders import _csv_to_binary, load_train_test_stack
from bandit_si import (
    ALL_VARIANTS, VARIANT_KEYS,
    run_single_variant_wor_si,
    _load_or_train_si_side_only,
    _side_only_mf_cache_path,
    _train_mf_device_local,
    _warm_train_nuclear_mf_device,
    _si_prewarm_cache_path,
    PREWARM_CACHE_DIR,
)
from bandit import NuclearNormMF, train_nuclear_mf

# Mirror of the SI worker's per-arm init-pull count. Both read the
# same env var so they stay in sync.
SI_INIT_PULLS_PER_ARM = int(os.environ.get("SI_INIT_PULLS_PER_ARM", "1"))


DEFAULT_TEST_PATH = "bbh/test.csv"
DEFAULT_TRAIN_PATH = "bbh/train.csv"
DEFAULT_MF_RANK = 10
DEFAULT_MF_INIT_EPOCHS = 5
DEFAULT_STAGE1_LR = 0.1
DEFAULT_STAGE1_LAM = 0.01
DEFAULT_PREWARM_EPOCHS = 10
DEFAULT_PREWARM_LR = 0.1
DEFAULT_PREWARM_LAM = 0.01
DEFAULT_BUDGET = 1_000_000
DEFAULT_BATCH_SIZE = 1
DEFAULT_TRAIN_EVERY = 10_000
DEFAULT_OBS_PROB = 0.0
OUT_PREFIX = "output_si/test_with_train_side__"


def _si_test_path_id(test_path,
                     rank=DEFAULT_MF_RANK,
                     lr=DEFAULT_STAGE1_LR,
                     lam=DEFAULT_STAGE1_LAM):
    """Cache-key salt for the SI per-test-set cache. The legacy salt
    (rank=10, lr=0.1, lam=0.01) keeps existing cache slots; any other
    combo gets a distinct salt so we never reuse a stale cache."""
    if (rank, lr, lam) == (DEFAULT_MF_RANK, DEFAULT_STAGE1_LR,
                           DEFAULT_STAGE1_LAM):
        return f"train+{test_path}"
    return f"train+{test_path}+rank{rank}+lr{lr}+lam{lam}"


def _set_si_env(device, mf_rank, mf_init_epochs, test_path,
                stage1_lr=DEFAULT_STAGE1_LR,
                stage1_lam=DEFAULT_STAGE1_LAM,
                prewarm_epochs=DEFAULT_PREWARM_EPOCHS,
                prewarm_lr=DEFAULT_PREWARM_LR,
                prewarm_lam=DEFAULT_PREWARM_LAM):
    os.environ["SI_DEVICE"] = device
    os.environ["SI_MF_RANK"] = str(mf_rank)
    os.environ["SI_MF_INIT_EPOCHS"] = str(mf_init_epochs)
    os.environ["SI_TEST_PATH_ID"] = _si_test_path_id(
        test_path, rank=mf_rank, lr=stage1_lr, lam=stage1_lam,
    )
    os.environ["SI_PREWARM_EPOCHS"] = str(prewarm_epochs)
    os.environ["SI_PREWARM_LR"] = str(prewarm_lr)
    os.environ["SI_PREWARM_LAM"] = str(prewarm_lam)
    os.environ["SI_INIT_PULLS_PER_ARM"] = str(SI_INIT_PULLS_PER_ARM)


def _master_train_only_cache_path(rank, n_epochs,
                                  train_path=DEFAULT_TRAIN_PATH,
                                  lr=DEFAULT_STAGE1_LR,
                                  lam=DEFAULT_STAGE1_LAM):
    """Filename includes a sanitised train basename so different train
    sources get distinct cache slots; lr/lam appear only when
    non-default, preserving backward compat for existing cached files."""
    if train_path == DEFAULT_TRAIN_PATH:
        train_part = f"mf_train_only_rank{rank}_ep{n_epochs}"
    else:
        salt = train_path.replace("/", "_").replace(".", "_")
        train_part = f"mf_train_only_{salt}_rank{rank}_ep{n_epochs}"
    if (lr, lam) == (DEFAULT_STAGE1_LR, DEFAULT_STAGE1_LAM):
        return f"output_si/mf_cache/{train_part}.pt"
    return f"output_si/mf_cache/{train_part}_lr{lr}_lam{lam}.pt"


def _load_or_fit_train_master(rank, n_epochs, device,
                              train_path=DEFAULT_TRAIN_PATH,
                              lr=DEFAULT_STAGE1_LR,
                              lam=DEFAULT_STAGE1_LAM):
    """Train (and cache) a NuclearNormMF on the train matrix alone.
    Loss is over every train cell; no test rows enter the matrix at
    all. Cached at a key salted by train_path + (lr, lam) when
    non-default."""
    path = _master_train_only_cache_path(rank, n_epochs, train_path, lr, lam)
    if os.path.exists(path):
        try:
            blob = torch.load(path, map_location="cpu")
        except Exception:
            blob = None
        if blob is not None and blob.get("rank") == rank:
            return blob
    M_train = _csv_to_binary(train_path)
    n_train = M_train.shape[0]
    side_mask = torch.ones_like(M_train, dtype=torch.bool)
    print(
        f"[ts master] training train-only rank={rank} n_epochs={n_epochs} "
        f"lr={lr} lam={lam} device={device} "
        f"(train shape {tuple(M_train.shape)})"
    )
    torch.manual_seed(0)
    np.random.seed(0)
    if device == "cpu":
        model = train_nuclear_mf(
            M_train, side_mask, rank=rank, n_epochs=n_epochs,
            lr=lr, lam=lam, batch_size=4096,
        )
    else:
        model = _train_mf_device_local(
            M_train, side_mask, rank=rank, n_epochs=n_epochs,
            lr=lr, lam=lam, batch_size=131072, device=device,
            log_prefix="train-master", verbose=True,
        )
    blob = {
        "state_dict": model.state_dict(),
        "n_train": int(n_train),
        "rank": int(rank),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(blob, path)
    print(f"[ts master] Saved {path}")
    return blob


def _materialise_si_cache_for_test(test_path, rank, n_epochs, device,
                                   train_path=DEFAULT_TRAIN_PATH,
                                   stage1_lr=DEFAULT_STAGE1_LR,
                                   stage1_lam=DEFAULT_STAGE1_LAM):
    """If the SI cache slot for `test_path` is missing, build it by
    cloning V and U[0:n_train] from the train-only master into a fresh
    NuclearNormMF(M_full_size, N, rank) and saving in the format that
    `_load_or_train_si_side_only` expects. Idempotent."""
    si_test_path_id = _si_test_path_id(
        test_path, rank=rank, lr=stage1_lr, lam=stage1_lam,
    )
    target = _side_only_mf_cache_path(rank, n_epochs, si_test_path_id)
    if os.path.exists(target):
        return target
    master = _load_or_fit_train_master(
        rank=rank, n_epochs=n_epochs, device=device, train_path=train_path,
        lr=stage1_lr, lam=stage1_lam,
    )
    M_full, test_pos = load_train_test_stack(
        test_path=test_path, train_path=train_path,
    )
    M_full_, N = M_full.shape
    n_train = master["n_train"]
    new_model = NuclearNormMF(M_full_, N, rank)
    msd = master["state_dict"]
    with torch.no_grad():
        new_model.V.data.copy_(msd["V"])
        new_model.U.data[:n_train].copy_(msd["U"])
        # U[n_train:] keeps NuclearNormMF's randn*0.01 init.
    side_mask = torch.ones((M_full_, N), dtype=torch.bool)
    side_mask[test_pos] = False
    blob = {
        "M_full": M_full_,
        "N": N,
        "n_test": int(len(test_pos)),
        "state_dict": new_model.state_dict(),
        "side_mask": side_mask,
    }
    os.makedirs(os.path.dirname(target), exist_ok=True)
    torch.save(blob, target)
    print(
        f"[ts materialise] Saved per-test cache {target} "
        f"for {test_path}"
    )
    return target


def warm_cache(device="cuda", mf_rank=DEFAULT_MF_RANK,
               mf_init_epochs=DEFAULT_MF_INIT_EPOCHS,
               test_path=DEFAULT_TEST_PATH,
               train_path=DEFAULT_TRAIN_PATH,
               stage1_lr=DEFAULT_STAGE1_LR,
               stage1_lam=DEFAULT_STAGE1_LAM,
               prewarm_epochs=DEFAULT_PREWARM_EPOCHS,
               prewarm_lr=DEFAULT_PREWARM_LR,
               prewarm_lam=DEFAULT_PREWARM_LAM):
    _set_si_env(
        device, mf_rank, mf_init_epochs, test_path,
        stage1_lr=stage1_lr, stage1_lam=stage1_lam,
        prewarm_epochs=prewarm_epochs,
        prewarm_lr=prewarm_lr, prewarm_lam=prewarm_lam,
    )
    print(
        f"[ts warm-cache] train_path={train_path} test_path={test_path} "
        f"rank={mf_rank} stage1_lr={stage1_lr} stage1_lam={stage1_lam}"
    )
    _materialise_si_cache_for_test(
        test_path=test_path, rank=mf_rank, n_epochs=mf_init_epochs,
        device=device, train_path=train_path,
        stage1_lr=stage1_lr, stage1_lam=stage1_lam,
    )
    print("[ts warm-cache] Done.")


def populate_prewarm_cache(n_seeds, seed_start=0,
                           batch_size=DEFAULT_BATCH_SIZE,
                           mf_rank=DEFAULT_MF_RANK,
                           mf_init_epochs=DEFAULT_MF_INIT_EPOCHS,
                           device="cuda",
                           test_path=DEFAULT_TEST_PATH,
                           train_path=DEFAULT_TRAIN_PATH,
                           stage1_lr=DEFAULT_STAGE1_LR,
                           stage1_lam=DEFAULT_STAGE1_LAM,
                           prewarm_epochs=DEFAULT_PREWARM_EPOCHS,
                           prewarm_lr=DEFAULT_PREWARM_LR,
                           prewarm_lam=DEFAULT_PREWARM_LAM):
    """Pre-populate the per-seed prewarm cache on GPU. Runs the
    deterministic prewarm once per seed (V frozen, U-only fine-tune on
    the per-seed init mask) and writes the cache file the bandit
    workers will load from. Skips seeds whose cache file already
    exists."""
    _set_si_env(
        device, mf_rank, mf_init_epochs, test_path,
        stage1_lr=stage1_lr, stage1_lam=stage1_lam,
        prewarm_epochs=prewarm_epochs,
        prewarm_lr=prewarm_lr, prewarm_lam=prewarm_lam,
    )
    _materialise_si_cache_for_test(
        test_path=test_path, rank=mf_rank,
        n_epochs=mf_init_epochs, device=device,
        train_path=train_path,
        stage1_lr=stage1_lr, stage1_lam=stage1_lam,
    )
    si_test_path_id = os.environ["SI_TEST_PATH_ID"]
    print(
        f"[ts populate-prewarm] n_seeds={n_seeds} seed_start={seed_start} "
        f"bs={batch_size} mf_rank={mf_rank} stage1=(lr={stage1_lr},"
        f"lam={stage1_lam}) prewarm=(ep={prewarm_epochs},lr={prewarm_lr},"
        f"lam={prewarm_lam}) test_path={test_path} device={device}"
    )

    M_full, test_pos = load_train_test_stack(
        test_path=test_path, train_path=train_path,
    )
    if isinstance(test_pos, torch.Tensor):
        test_pos_np = test_pos.cpu().numpy()
    else:
        test_pos_np = np.asarray(test_pos, dtype=np.int64)
    M_full_size, N = M_full.shape
    n_test = len(test_pos_np)
    n_init_cells_per_arm = min(SI_INIT_PULLS_PER_ARM * batch_size, N)

    # Pin torch CPU thread count so the on-CPU side of the prewarm
    # (data prep, optimizer state) stays cheap.
    torch.set_num_threads(1)

    n_hits = 0
    n_misses = 0
    for offset in range(n_seeds):
        exp_idx = seed_start + offset
        seed = exp_idx * 10000

        cache_path = _si_prewarm_cache_path(
            seed=seed, rank=mf_rank, n_epochs=mf_init_epochs,
            prewarm_epochs=prewarm_epochs,
            prewarm_lr=prewarm_lr, prewarm_lam=prewarm_lam,
            init_pulls_per_arm=SI_INIT_PULLS_PER_ARM,
            batch_size=batch_size,
            test_path_id=si_test_path_id,
        )
        if os.path.exists(cache_path):
            n_hits += 1
            print(
                f"[ts populate-prewarm] seed={seed} HIT {cache_path}"
            )
            continue

        # Mirror the SI worker's RNG state and init-mask draw exactly so
        # the cached prewarm matches what the bandit worker would do.
        torch.manual_seed(seed)
        np.random.seed(seed)
        test_mask = torch.zeros((n_test, N), dtype=torch.bool)
        for _arm in range(n_test):
            _cols = np.random.choice(
                N, size=n_init_cells_per_arm, replace=False,
            )
            test_mask[_arm, _cols] = True
        init_mask_full = torch.zeros((M_full_size, N), dtype=torch.bool)
        init_mask_full[
            torch.tensor(test_pos_np, dtype=torch.long)
        ] = test_mask
        init_X_obs_full = M_full * init_mask_full

        # Load the per-test SI cache (V trained on train rows only,
        # U[side] cloned from master). Cache is shared across seeds.
        model, _ = _load_or_train_si_side_only(
            M_full, test_pos_np,
            rank=mf_rank, n_epochs=mf_init_epochs,
            test_path=si_test_path_id,
            device=device,
        )
        model.V.requires_grad_(False)

        print(
            f"[ts populate-prewarm] seed={seed} MISS — running "
            f"prewarm ({prewarm_epochs} epochs) on {device}"
        )
        model = _warm_train_nuclear_mf_device(
            model, init_X_obs_full, init_mask_full,
            lr=prewarm_lr, n_epochs=prewarm_epochs,
            lam=prewarm_lam, batch_size=131072,
            device=device,
        )

        os.makedirs(PREWARM_CACHE_DIR, exist_ok=True)
        tmp = cache_path + f".tmp.{os.getpid()}"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "M_full": M_full_size, "N": N, "rank": mf_rank,
            },
            tmp,
        )
        os.replace(tmp, cache_path)
        n_misses += 1
        print(
            f"[ts populate-prewarm] seed={seed} saved {cache_path}"
        )

    print(
        f"[ts populate-prewarm] Done. hits={n_hits} misses={n_misses} "
        f"of {n_seeds} seeds."
    )


def _stack_truncate(lst):
    if not lst:
        return np.zeros(0)
    L = min(len(a) for a in lst)
    return np.stack([np.asarray(a)[:L] for a in lst])


def _mean_stderr_pct(arr):
    if arr.size == 0:
        return np.zeros(0), np.zeros(0)
    mean = arr.mean(axis=0) * 100
    stderr = (
        (arr.std(axis=0, ddof=1) * 100) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1 else np.zeros_like(mean)
    )
    return mean, stderr


def run_variant(variant, n_seeds=10, seed_start=0, out_suffix="",
                save_raw=False,
                budget=DEFAULT_BUDGET, batch_size=DEFAULT_BATCH_SIZE,
                train_every=DEFAULT_TRAIN_EVERY, obs_prob=DEFAULT_OBS_PROB,
                a=1.0, lam_cv=1.0,
                mf_rank=DEFAULT_MF_RANK, mf_init_epochs=DEFAULT_MF_INIT_EPOCHS,
                device="cuda", processes=1,
                test_path=DEFAULT_TEST_PATH,
                train_path=DEFAULT_TRAIN_PATH,
                stage1_lr=DEFAULT_STAGE1_LR,
                stage1_lam=DEFAULT_STAGE1_LAM,
                prewarm_epochs=DEFAULT_PREWARM_EPOCHS,
                prewarm_lr=DEFAULT_PREWARM_LR,
                prewarm_lam=DEFAULT_PREWARM_LAM):
    if variant not in ALL_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; pick from {ALL_VARIANTS}")
    _set_si_env(
        device, mf_rank, mf_init_epochs, test_path,
        stage1_lr=stage1_lr, stage1_lam=stage1_lam,
        prewarm_epochs=prewarm_epochs,
        prewarm_lr=prewarm_lr, prewarm_lam=prewarm_lam,
    )
    _materialise_si_cache_for_test(
        test_path=test_path, rank=mf_rank,
        n_epochs=mf_init_epochs, device=device,
        train_path=train_path,
        stage1_lr=stage1_lr, stage1_lam=stage1_lam,
    )
    print(
        f"[ts {variant}] n_seeds={n_seeds} seed_start={seed_start} "
        f"budget={budget} bs={batch_size} obs_prob={obs_prob} "
        f"mf_rank={mf_rank} stage1_lr={stage1_lr} stage1_lam={stage1_lam} "
        f"prewarm=({prewarm_epochs}ep,lr={prewarm_lr},lam={prewarm_lam}) "
        f"train_path={train_path} test_path={test_path} device={device}"
    )

    M_full, test_pos = load_train_test_stack(
        test_path=test_path, train_path=train_path,
    )
    n_test = len(test_pos)
    X_full_test = M_full[test_pos.cpu().numpy()]
    true_means = X_full_test.float().mean(dim=1).numpy()
    actual_best_arm = int(np.argmax(true_means))
    top_k_arms = np.argsort(true_means)[-5:]
    eps_arms = np.where(
        true_means >= true_means[actual_best_arm] - 0.05
    )[0]
    print(
        f"[ts {variant}] test best arm idx_in_test={actual_best_arm} "
        f"(mean={true_means[actual_best_arm]:.4f}); "
        f"|eps_arms|={len(eps_arms)}"
    )

    tasks = []
    for offset in range(n_seeds):
        exp_idx = seed_start + offset
        tasks.append((
            variant, exp_idx, M_full, test_pos,
            actual_best_arm, top_k_arms, eps_arms,
            budget, batch_size, a, lam_cv, obs_prob,
        ))

    hist_top1_list, hist_topk_list, hist_eps_list, cells_list = [], [], [], []
    if processes <= 1:
        for t in tasks:
            print(f"[ts {variant}] seed_idx={t[1]} "
                  f"(seed_value={t[1]*10000})")
            res = run_single_variant_wor_si(t)
            hist_top1_list.append(res["hist_top1"])
            hist_topk_list.append(res["hist_topk"])
            hist_eps_list.append(res["hist_eps"])
            if res["cells"] is not None:
                cells_list.append(res["cells"])
    else:
        import multiprocessing
        from tqdm import tqdm
        print(f"[ts {variant}] Pool(processes={processes}) over "
              f"{len(tasks)} seeds")
        with multiprocessing.Pool(processes=processes) as pool:
            for res in tqdm(
                pool.imap_unordered(run_single_variant_wor_si, tasks),
                total=len(tasks), desc=f"ts {variant}",
            ):
                hist_top1_list.append(res["hist_top1"])
                hist_topk_list.append(res["hist_topk"])
                hist_eps_list.append(res["hist_eps"])
                if res["cells"] is not None:
                    cells_list.append(res["cells"])

    top1_arr = _stack_truncate(hist_top1_list)
    topk_arr = _stack_truncate(hist_topk_list)
    eps_arr = _stack_truncate(hist_eps_list)
    mean_top1, stderr_top1 = _mean_stderr_pct(top1_arr)
    mean_topk, stderr_topk = _mean_stderr_pct(topk_arr)
    mean_eps, stderr_eps = _mean_stderr_pct(eps_arr)

    keys = VARIANT_KEYS[variant]
    save_dict = {
        keys["mean"]: mean_top1,
        keys["stderr"]: stderr_top1,
        keys["mean_topk"]: mean_topk,
        keys["stderr_topk"]: stderr_topk,
        keys["mean_eps"]: mean_eps,
        keys["stderr_eps"]: stderr_eps,
        "wor_batch_size": int(batch_size),
        "wor_m": int(n_test),
        "actual_best_arm": int(actual_best_arm),
        "n_eps_arms": int(len(eps_arms)),
        "budget": int(budget),
    }
    if keys.get("cells_key") is not None and cells_list:
        cells_arr = _stack_truncate(cells_list)
        save_dict[keys["cells_key"]] = cells_arr.mean(axis=0)
        if save_raw:
            save_dict[f"raw_cells_{variant}"] = cells_arr

    if save_raw:
        save_dict[f"raw_top1_{variant}"] = top1_arr
        save_dict[f"raw_topk_{variant}"] = topk_arr
        save_dict[f"raw_eps_{variant}"] = eps_arr
        save_dict["seed_indices"] = np.array(
            [seed_start + i for i in range(n_seeds)], dtype=np.int64
        )

    os.makedirs("output_si", exist_ok=True)
    out_path = f"{OUT_PREFIX}{variant}{out_suffix}.npz"
    np.savez(out_path, **save_dict)
    print(f"[ts {variant}] Saved {out_path}")
    print(f"  final top-1: {mean_top1[-1]:.2f}% +/- {stderr_top1[-1]:.2f}%")
    print(f"  final top-5: {mean_topk[-1]:.2f}% +/- {stderr_topk[-1]:.2f}%")
    return out_path


def merge_chunks(variant, keep_chunks=False, chunk_tag_filter=None,
                 final_suffix=""):
    """Concatenate raw_* arrays from per-chunk npz files into the
    canonical aggregate npz.

    `chunk_tag_filter` (optional): only merge chunks whose filename
    contains this substring (so a bs=64 run can be merged separately
    from a bs=1 run via a tag like "_bs64_").
    `final_suffix`: appended to the canonical output filename, e.g.
    "_bs64" gives `test_with_train_side__<variant>_bs64.npz`.
    """
    if variant not in ALL_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    if chunk_tag_filter is not None:
        chunk_glob = f"{OUT_PREFIX}{variant}_gpu_chunk_*{chunk_tag_filter}*.npz"
    else:
        chunk_glob = f"{OUT_PREFIX}{variant}_gpu_chunk_*.npz"
    files = sorted(glob.glob(chunk_glob))
    if not files:
        raise SystemExit(f"No chunks matching {chunk_glob}")
    print(f"[merge {variant}] {len(files)} chunks:")
    for f in files:
        print(f"  - {f}")

    keys = VARIANT_KEYS[variant]
    raw_top1, raw_topk, raw_eps, raw_cells = [], [], [], []
    scalars = {}
    for f in files:
        z = np.load(f, allow_pickle=False)
        raw_top1.append(z[f"raw_top1_{variant}"])
        raw_topk.append(z[f"raw_topk_{variant}"])
        raw_eps.append(z[f"raw_eps_{variant}"])
        if f"raw_cells_{variant}" in z.files:
            raw_cells.append(z[f"raw_cells_{variant}"])
        for k in (
            "wor_batch_size", "wor_m", "actual_best_arm",
            "n_eps_arms", "budget",
        ):
            if k in z.files:
                scalars[k] = z[k]

    def _trunc_concat(lst):
        L = min(a.shape[1] for a in lst)
        return np.concatenate([a[:, :L] for a in lst], axis=0)

    top1 = _trunc_concat(raw_top1)
    topk = _trunc_concat(raw_topk)
    eps = _trunc_concat(raw_eps)
    cells = _trunc_concat(raw_cells) if raw_cells else None

    n_seeds = top1.shape[0]
    mean_top1, stderr_top1 = _mean_stderr_pct(top1)
    mean_topk, stderr_topk = _mean_stderr_pct(topk)
    mean_eps, stderr_eps = _mean_stderr_pct(eps)

    save_dict = {
        keys["mean"]: mean_top1,
        keys["stderr"]: stderr_top1,
        keys["mean_topk"]: mean_topk,
        keys["stderr_topk"]: stderr_topk,
        keys["mean_eps"]: mean_eps,
        keys["stderr_eps"]: stderr_eps,
        "n_seeds": int(n_seeds),
        f"raw_top1_{variant}": top1,
        f"raw_topk_{variant}": topk,
        f"raw_eps_{variant}": eps,
    }
    save_dict.update(scalars)
    if cells is not None and keys.get("cells_key") is not None:
        save_dict[keys["cells_key"]] = cells.mean(axis=0)
        save_dict[f"raw_cells_{variant}"] = cells

    out_path = f"{OUT_PREFIX}{variant}{final_suffix}.npz"
    np.savez(out_path, **save_dict)
    print(f"[merge {variant}] Saved {out_path} (n_seeds={n_seeds})")
    print(f"  final top-1: {mean_top1[-1]:.2f}% +/- {stderr_top1[-1]:.2f}%")
    print(f"  final top-5: {mean_topk[-1]:.2f}% +/- {stderr_topk[-1]:.2f}%")
    if not keep_chunks:
        for f in files:
            os.remove(f)
        print(f"[merge {variant}] Removed {len(files)} chunk files")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=None,
                   choices=ALL_VARIANTS + [None],
                   help="Bandit variant to run.")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out-suffix", default="")
    p.add_argument("--save-raw", action="store_true")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--train-every", type=int, default=DEFAULT_TRAIN_EVERY)
    p.add_argument("--obs-prob", type=float, default=DEFAULT_OBS_PROB)
    p.add_argument("--mf-rank", type=int, default=DEFAULT_MF_RANK)
    p.add_argument(
        "--mf-init-epochs", type=int, default=DEFAULT_MF_INIT_EPOCHS,
    )
    p.add_argument("--test-path", default=DEFAULT_TEST_PATH,
                   help="CSV providing test rows. Defaults to bbh/test.csv.")
    p.add_argument("--train-path", default=DEFAULT_TRAIN_PATH,
                   help="CSV providing train (side-info) rows. "
                        "Defaults to bbh/train.csv.")
    p.add_argument("--stage1-lr", type=float, default=DEFAULT_STAGE1_LR)
    p.add_argument("--stage1-lam", type=float, default=DEFAULT_STAGE1_LAM)
    p.add_argument("--prewarm-epochs", type=int,
                   default=DEFAULT_PREWARM_EPOCHS)
    p.add_argument("--prewarm-lr", type=float, default=DEFAULT_PREWARM_LR)
    p.add_argument("--prewarm-lam", type=float, default=DEFAULT_PREWARM_LAM)
    p.add_argument("--processes", type=int, default=1,
                   help="If >1, run seeds in a multiprocess Pool "
                        "(use for std_wor on CPU).")
    p.add_argument("--warm-cache", action="store_true",
                   help="Train+cache stage-1 V once and exit.")
    p.add_argument("--populate-prewarm-cache", action="store_true",
                   help="GPU-serial prewarm-cache populator: loop seeds, "
                        "run prewarm, save per-seed cache. Skips seeds "
                        "whose cache already exists.")
    p.add_argument("--merge", action="store_true",
                   help="Merge per-chunk npz files for --variant.")
    p.add_argument("--keep-chunks", action="store_true")
    p.add_argument("--chunk-tag-filter", default=None,
                   help="(merge mode) only merge chunks whose filename "
                        "contains this substring.")
    p.add_argument("--final-suffix", default="",
                   help="(merge mode) appended to the output npz path "
                        "after the variant name (e.g. '_bs64').")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.warm_cache:
        warm_cache(
            device=args.device,
            mf_rank=args.mf_rank, mf_init_epochs=args.mf_init_epochs,
            test_path=args.test_path, train_path=args.train_path,
            stage1_lr=args.stage1_lr, stage1_lam=args.stage1_lam,
            prewarm_epochs=args.prewarm_epochs,
            prewarm_lr=args.prewarm_lr, prewarm_lam=args.prewarm_lam,
        )
        return
    if args.populate_prewarm_cache:
        populate_prewarm_cache(
            n_seeds=args.n_seeds, seed_start=args.seed_start,
            batch_size=args.batch_size,
            mf_rank=args.mf_rank, mf_init_epochs=args.mf_init_epochs,
            device=args.device,
            test_path=args.test_path, train_path=args.train_path,
            stage1_lr=args.stage1_lr, stage1_lam=args.stage1_lam,
            prewarm_epochs=args.prewarm_epochs,
            prewarm_lr=args.prewarm_lr, prewarm_lam=args.prewarm_lam,
        )
        return
    if args.merge:
        if args.variant is None:
            raise SystemExit("--merge requires --variant")
        merge_chunks(
            args.variant, keep_chunks=args.keep_chunks,
            chunk_tag_filter=args.chunk_tag_filter,
            final_suffix=args.final_suffix,
        )
        return
    if args.variant is None:
        raise SystemExit(
            "Need one of --warm-cache / --variant <name> / "
            "--merge --variant <name>."
        )
    run_variant(
        args.variant,
        n_seeds=args.n_seeds, seed_start=args.seed_start,
        out_suffix=args.out_suffix, save_raw=args.save_raw,
        budget=args.budget, batch_size=args.batch_size,
        train_every=args.train_every, obs_prob=args.obs_prob,
        mf_rank=args.mf_rank, mf_init_epochs=args.mf_init_epochs,
        device=args.device, processes=args.processes,
        test_path=args.test_path, train_path=args.train_path,
        stage1_lr=args.stage1_lr, stage1_lam=args.stage1_lam,
        prewarm_epochs=args.prewarm_epochs,
        prewarm_lr=args.prewarm_lr, prewarm_lam=args.prewarm_lam,
    )


if __name__ == "__main__":
    main()
