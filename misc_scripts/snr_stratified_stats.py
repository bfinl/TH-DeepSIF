"""
run_snr_stratified.py — SNR-stratified simulated-data evaluation (R3.3)

Reuses existing predictions for the 1000-trial subset Fig 3 used:
    Single-source: source/test_998_snr15_extE4S1.mat (selected_region (N,1,70))
    Dual-source:   source/test_998_snr15_extenE4.mat (selected_region (N,2,70))
Indices: (0:999)*17+1 (1-indexed in MATLAB, so 0-indexed = np.arange(1000)*17).

Each trial has a per-trial SNR in current_snr (~5-20 dB).
We bin into 4 levels and report TH-DeepSIF (model 44) vs sLORETA per bin
on LE / Sensitivity / Precision / Linear-Correlation. Wilcoxon signed-rank
(paired within bin) with Holm correction across the 4 metrics × 4 bins.

Outputs:
    misc_scripts/snr_stratified_metrics.csv
    misc_scripts/snr_stratified_stats.csv
"""
import os
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_ROOT = os.path.join(ROOT, "model_weights")
STATS_DIR = os.path.join(ROOT, "misc_scripts")

# 1000-trial subset used in the paper: indices (0:999)*17+1 (MATLAB 1-based)
INDICES = np.arange(1000) * 17  # 0-based equivalent

DATASETS = {
    "single": {
        "source_mat": os.path.join(ROOT, "source/test_998_snr15_extE4S1.mat"),
        "deepsif_mat": os.path.join(RESULT_ROOT, "44_the_model/model_best.pth.tar_preds_test_998_snr15_extE4S1.mat"),
        "sloreta_mat": os.path.join(ROOT, "patient_hfo/TraditionalMethod/s_sloreta_eval_results_test_998_snr15_extE4S1.mat"),
    },
    "dual": {
        "source_mat": os.path.join(ROOT, "source/test_998_snr15_extenE4.mat"),
        "deepsif_mat": os.path.join(RESULT_ROOT, "44_the_model/model_best.pth.tar_preds_test_998_snr15_extenE4.mat"),
        "sloreta_mat": os.path.join(ROOT, "patient_hfo/TraditionalMethod/s_sloreta_eval_results_test_998_snr15_extenE4.mat"),
    },
}

# Training SNRs are discrete: 5, 10, 15, 20 dB (250 trials each in the 1000-subset)
SNR_LEVELS = [5, 10, 15, 20]


def load_per_trial_arrays(pred_mat_path, indices):
    """Predictions saved as v7.3 h5 with cell arrays for le/precision/recall."""
    with h5py.File(pred_mat_path, "r") as f:
        le_refs = f["le"][:].squeeze()         # (N,)
        pr_refs = f["precision"][:].squeeze()
        re_refs = f["recall"][:].squeeze()
        # Each entry is a reference to a 1-element double array
        def deref(refs):
            vals = np.empty(len(refs), dtype=float)
            for i, r in enumerate(refs):
                d = f[r][()]
                vals[i] = np.asarray(d).squeeze().item()
            return vals
        le = deref(le_refs)
        pr = deref(pr_refs)
        re = deref(re_refs)
    return le[indices], pr[indices], re[indices]


def load_sloreta_per_trial(sl_mat_path):
    """sLORETA results saved as MATLAB v7.3 (h5py).
    Cell arrays of scalars: shape (1000, 1), already the figure3 subset.
    """
    with h5py.File(sl_mat_path, "r") as f:
        def deref(refs):
            refs = refs.squeeze()
            vals = np.empty(len(refs), dtype=float)
            for i, r in enumerate(refs):
                vals[i] = float(np.asarray(f[r][()]).squeeze().item())
            return vals
        le = deref(f["le"][:])
        pr = deref(f["precision"][:])
        re = deref(f["recall"][:])
    return le, pr, re


def load_snr(source_mat_path, indices):
    d = loadmat(source_mat_path, squeeze_me=False)
    csnr = np.asarray(d["current_snr"]).squeeze()
    return csnr[indices]


def compute_lc(pred_mat_path, indices, T=450):
    """Fixed linear-correlation definition (manuscript revision):
        lc = mean over sources of max_r corr(gt_nmm[:, source], out[r, :])
      where the max is over ALL active reconstructed-region rows of `out`
      (n_regions, T). The original MATLAB `compute_linear_correlation.m`
      used only the LAST row due to a single-iteration MATLAB for-loop bug;
      see fig_lc_definition_comparison.png for the side-by-side. We adopt
      the max-over-rows definition in this revision because it (a) matches
      the docstring intent of the original, (b) treats sLORETA and TH-DeepSIF
      symmetrically (sLORETA's row ordering is otherwise arbitrary), and
      (c) better reflects the waveform-recovery capability of the model.
    """
    n = len(indices)
    lcs = np.full(n, np.nan)
    with h5py.File(pred_mat_path, "r") as f:
        out_refs = f["all_out"][:].squeeze()
        nmm_refs = f["all_nmm"][:].squeeze()
        for k, idx in enumerate(indices):
            out = np.asarray(f[out_refs[idx]][()])
            nmm = np.asarray(f[nmm_refs[idx]][()])
            if out.size == 0 or nmm.size == 0:
                continue
            if nmm.shape[0] != T:
                nmm = nmm.T  # (T, n_sources)
            if out.shape[1] != T:
                out = out.T  # (n_regions, T)
            corrs_max = []
            for s in range(nmm.shape[1]):
                truth = nmm[:, s]
                if np.std(truth) == 0:
                    continue
                # Vectorized correlation: truth (T,) vs each row of out (n_regions, T)
                t_c = truth - truth.mean()
                t_norm = np.linalg.norm(t_c)
                if t_norm == 0:
                    continue
                out_c = out - out.mean(axis=1, keepdims=True)
                out_norms = np.linalg.norm(out_c, axis=1)
                valid = out_norms > 0
                if not valid.any():
                    continue
                r_all = (out_c[valid] @ t_c) / (out_norms[valid] * t_norm)
                best = float(np.nanmax(r_all))
                if np.isfinite(best):
                    corrs_max.append(best)
            if corrs_max:
                lcs[k] = float(np.mean(corrs_max))
    return lcs


def holm_correct(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        v = pvals[idx] * (n - rank)
        running = max(running, v)
        adj[idx] = min(running, 1.0)
    return adj


def main():
    rows = []
    stats_rows = []
    trial_rows = []
    for kind, paths in DATASETS.items():
        print(f"\n=== {kind}-source ===")
        snr = load_snr(paths["source_mat"], INDICES)
        le_d, pr_d, re_d = load_per_trial_arrays(paths["deepsif_mat"], INDICES)
        try:
            sl_full = load_sloreta_per_trial(paths["sloreta_mat"])
            # sLORETA may have stored only the 1000 indices already, or full 23856.
            if sl_full[0].shape[0] == 23856:
                le_s = sl_full[0][INDICES]
                pr_s = sl_full[1][INDICES]
                re_s = sl_full[2][INDICES]
            else:
                le_s, pr_s, re_s = sl_full
        except Exception as e:
            print(f"  sLORETA load failed: {e}")
            le_s = pr_s = re_s = np.full_like(le_d, np.nan)

        lc_d = compute_lc(paths["deepsif_mat"], INDICES)
        print(f"  DeepSIF LC: n_valid={np.isfinite(lc_d).sum()}/{len(lc_d)}, "
              f"mean={np.nanmean(lc_d):.3f}")
        # sLORETA LC: file is already at the 1000-trial subset, so pass arange(1000)
        try:
            lc_s_all = compute_lc(paths["sloreta_mat"], np.arange(1000))
            print(f"  sLORETA LC: n_valid={np.isfinite(lc_s_all).sum()}/{len(lc_s_all)}, "
                  f"mean={np.nanmean(lc_s_all):.3f}")
        except Exception as e:
            print(f"  sLORETA LC failed: {e}")
            lc_s_all = np.full_like(lc_d, np.nan)

        for snr_val in SNR_LEVELS:
            mask = (snr == snr_val)
            n = mask.sum()
            if n == 0:
                continue
            lo, hi = snr_val, snr_val
            for metric_name, d_vals, s_vals in [
                ("le", le_d, le_s),
                ("sensitivity", re_d, re_s),
                ("precision", pr_d, pr_s),
                ("lc", lc_d, lc_s_all),
            ]:
                d_sub = d_vals[mask]
                s_sub = s_vals[mask] if not np.all(np.isnan(s_vals)) else None
                rows.append({"kind": kind, "snr_bin": f"{snr_val}dB",
                             "n": int(n),
                             "metric": metric_name,
                             "deepsif_mean": float(np.nanmean(d_sub)),
                             "deepsif_sem": float(np.nanstd(d_sub) / np.sqrt(n)),
                             "sloreta_mean": (float(np.nanmean(s_sub)) if s_sub is not None else np.nan),
                             "sloreta_sem": (float(np.nanstd(s_sub) / np.sqrt(n)) if s_sub is not None else np.nan)})
                # ---- per-trial dump (so legend stats never need a 38 GB re-read) ----
                for _i, (_dv, _sv) in enumerate(zip(
                        d_sub, (s_sub if s_sub is not None
                                else np.full(len(d_sub), np.nan)))):
                    trial_rows.append({"kind": kind, "snr_bin": f"{snr_val}dB",
                                       "metric": metric_name, "trial": _i,
                                       "deepsif": float(_dv), "sloreta": float(_sv)})
                W = np.nan; zstat = np.nan; rbc = np.nan; n_pairs = 0
                if s_sub is not None and not np.all(np.isnan(s_sub)):
                    try:
                        # paired Wilcoxon (same trial idx for both methods)
                        valid = ~(np.isnan(d_sub) | np.isnan(s_sub))
                        n_pairs = int(valid.sum())
                        if n_pairs >= 3:
                            W_stat, p = wilcoxon(d_sub[valid], s_sub[valid])
                            W = float(W_stat); p = float(p)
                            # signed-rank sums -> effect size + tie-corrected z
                            diff = d_sub[valid] - s_sub[valid]
                            nz = diff[diff != 0]
                            nnz = int(nz.size)
                            if nnz:
                                ranks = pd.Series(np.abs(nz)).rank().values
                                r_pos = float(ranks[nz > 0].sum())
                                r_neg = float(ranks[nz < 0].sum())
                                rbc = (r_pos - r_neg) / (r_pos + r_neg)
                                mu = nnz * (nnz + 1) / 4.0
                                _u, tcnt = np.unique(ranks, return_counts=True)
                                tie_corr = float(np.sum(tcnt.astype(float) ** 3 - tcnt))
                                var = (nnz * (nnz + 1) * (2 * nnz + 1)) / 24.0 - tie_corr / 48.0
                                if var > 0:
                                    zstat = (min(r_pos, r_neg) - mu) / np.sqrt(var)
                                    if r_pos > r_neg:
                                        zstat = -zstat
                        else:
                            p = np.nan
                    except ValueError:
                        p = np.nan
                else:
                    p = np.nan
                stats_rows.append({"kind": kind, "snr_bin": f"{snr_val}dB",
                                   "metric": metric_name, "W": W, "z": zstat,
                                   "rank_biserial": rbc, "p": p,
                                   "n": int(n), "n_pairs": n_pairs})

    pd.DataFrame(rows).to_csv(os.path.join(STATS_DIR, "snr_stratified_metrics.csv"), index=False)
    sdf = pd.DataFrame(stats_rows)
    # Holm correction across all comparisons
    if not sdf.empty:
        valid = ~sdf["p"].isna()
        p_holm = np.full(len(sdf), np.nan)
        if valid.any():
            p_holm[valid.values] = holm_correct(sdf.loc[valid, "p"].values)
        sdf["p_holm"] = p_holm
    sdf.to_csv(os.path.join(STATS_DIR, "snr_stratified_stats.csv"), index=False)
    pd.DataFrame(trial_rows).to_csv(os.path.join(STATS_DIR, "snr_stratified_per_trial.csv"), index=False)
    pd.set_option("display.width", 200)
    print(sdf.to_string(index=False))
    print("Wrote snr_stratified_metrics.csv, _stats.csv, _per_trial.csv")


if __name__ == "__main__":
    main()
