"""Statistical tests underlying the reported figures.

Every inferential statistic in the paper comes from one of the functions below.
Plotting code is not included; these are the tests, with the figure each one
produced.

    Figure 3             wilcoxon_paired_holm      TH-DeepSIF vs sLORETA, simulated trials
    Figure 4             lme_two_groups            TH-DeepSIF vs sLORETA, clinical pHFOs
    Figure 5             lme_two_groups            pHFO vs aHFO (all detected HFOs)
    Supplementary Fig 2  snr_stratified_stats.py   per-SNR Wilcoxon, Holm across 32 cells
    Supplementary Fig 4  supp_lme_stats.py         seizure-free vs non-seizure-free
    Supplementary Fig 5  lme_two_groups            pHFO vs iHFO (isolated HFOs)
    Supplementary Fig 6  supp_lme_stats.py         SOZ localization error, pHFO vs aHFO

Conventions shared by all clinical comparisons (Figures 4-5, Supplementary 4-6):

  * Unit of inference is the individual HFO event; a patient-specific random
    intercept absorbs within-patient repeated measures:
        metric ~ group + (1 | patient)
    fitted by maximum likelihood (statsmodels MixedLM, lbfgs, reml=False).
    We report the Wald z-statistic and p-value of the group coefficient.

  * Holm-Bonferroni is applied within two families per figure, via
    holm_split_primary_derived():
        primary  = {localization error, specificity, precision, sensitivity}
        derived  = {geometric mean, F1}
    F1 and GM are arithmetic combinations of the primary metrics, so they do
    not draw on the primary family's error budget.

  * Effect sizes: Cohen's d (paired for within-event method comparisons,
    pooled for between-group comparisons).

Simulation comparisons (Figure 3, Supplementary Figure 2) use a two-sided
paired Wilcoxon signed-rank test, since trials are independent and the two
methods are evaluated on identical trials.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRIMARY_METRICS = ["le", "sp", "pr", "re"]
DERIVED_METRICS = ["gm", "f1"]


# --------------------------------------------------------------------------
# multiple-comparison correction
# --------------------------------------------------------------------------
def holm(pvals):
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, p[idx] * (n - rank))
        adj[idx] = min(running, 1.0)
    return adj


def holm_split_primary_derived(raw_p, primary=PRIMARY_METRICS, derived=DERIVED_METRICS):
    """Holm applied separately to the primary and derived metric families.

    raw_p : dict  metric -> raw p-value
    Returns a dict metric -> Holm-adjusted p-value (metrics absent from
    either family are returned unadjusted).
    """
    out = dict(raw_p)
    for family in (primary, derived):
        keys = [k for k in family if k in raw_p]
        if keys:
            out.update(dict(zip(keys, holm([raw_p[k] for k in keys]))))
    return out


# --------------------------------------------------------------------------
# effect sizes and derived metrics
# --------------------------------------------------------------------------
def cohens_d_paired(x, y):
    """Cohen's d for paired samples: mean(x - y) / sd(x - y)."""
    diff = np.asarray(x, float) - np.asarray(y, float)
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else np.nan


def cohens_d_pooled(x, y):
    """Cohen's d with pooled standard deviation for independent groups."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / sp) if sp > 0 else np.nan


def derived_f1(precision, recall):
    """F1 = 2*Pr*Re / (Pr+Re), with 0/0 -> 0."""
    pr, re = np.asarray(precision, float), np.asarray(recall, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((pr + re) > 0, 2 * pr * re / (pr + re), 0.0)
    f1[~np.isfinite(f1)] = 0.0
    return f1


def derived_gm(specificity, recall):
    """Geometric mean of specificity and sensitivity."""
    return np.sqrt(np.asarray(specificity, float) * np.asarray(recall, float))


# --------------------------------------------------------------------------
# Figure 3 — simulated trials, TH-DeepSIF vs sLORETA
# --------------------------------------------------------------------------
def wilcoxon_paired_holm(deepsif, sloreta, metrics=("re", "pr", "le", "lc")):
    """Paired Wilcoxon signed-rank per metric, Holm across the listed metrics.

    deepsif, sloreta : dict  metric -> per-trial values (same trial order)
    Returns a DataFrame with one row per metric: n, W, p_raw, p_holm, cohens_d.
    Used for each source condition (single, two) of Figure 3.
    """
    rows = []
    for m in metrics:
        a, b = np.asarray(deepsif[m], float), np.asarray(sloreta[m], float)
        mask = ~(np.isnan(a) | np.isnan(b))
        try:
            w, p = wilcoxon(a[mask], b[mask])
        except ValueError:
            w, p = np.nan, np.nan
        rows.append(dict(metric=m, n=int(mask.sum()), W=float(w), p_raw=float(p),
                         cohens_d=cohens_d_paired(a[mask], b[mask])))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_raw"].values)
    return df


# --------------------------------------------------------------------------
# Figures 4, 5 and Supplementary 4-6 — clinical events, linear mixed effects
# --------------------------------------------------------------------------
def lme_two_groups(values_a, values_b, patients_a, patients_b,
                   label_a, label_b, term="group"):
    """metric ~ group + (1 | patient), maximum likelihood.

    values_*   : per-event metric values for the two groups
    patients_* : matching per-event patient identifiers
    Returns dict(beta, se, z, p, n, n_patients) for the group coefficient
    (contrast is label_b relative to label_a, alphabetically per statsmodels).

    Figure 4:            a = TH-DeepSIF, b = sLORETA   (same events, both methods)
    Figure 5:            a = pHFO,       b = aHFO
    Supplementary Fig 5: a = pHFO,       b = iHFO
    Supplementary Fig 6: a = pHFO,       b = aHFO      (localization error vs SOZ)
    Supplementary Fig 4: a = SF cohort,  b = non-SF cohort
    """
    import statsmodels.formula.api as smf
    long = pd.DataFrame({
        "metric": np.concatenate([values_a, values_b]).astype(float),
        term: [label_a] * len(values_a) + [label_b] * len(values_b),
        "patient_id": np.concatenate([patients_a, patients_b]),
    }).dropna()
    md = smf.mixedlm("metric ~ %s" % term, long, groups=long["patient_id"])
    mdf = md.fit(method="lbfgs", reml=False)
    cn = [c for c in mdf.params.index if c != "Intercept" and not c.startswith("Group Var")][0]
    return dict(beta=float(mdf.params[cn]), se=float(mdf.bse[cn]),
                z=float(mdf.tvalues[cn]), p=float(mdf.pvalues[cn]),
                n=int(len(long)), n_patients=int(long["patient_id"].nunique()),
                contrast=cn)


def compare_groups_all_metrics(df_a, df_b, label_a, label_b,
                               metrics=("le", "sp", "pr", "re", "gm", "f1"),
                               paired=False):
    """Run lme_two_groups for every metric and apply the split Holm correction.

    df_a, df_b : DataFrames with a 'patient_id' column and one column per metric
    paired     : True for Figure 4 (same events under two methods -> paired d),
                 False for between-group comparisons (pooled d).
    Returns a DataFrame: metric, beta, se, z, p_raw, p_holm, cohens_d, n, n_patients.
    """
    rows, raw = [], {}
    for m in metrics:
        r = lme_two_groups(df_a[m].values, df_b[m].values,
                           df_a["patient_id"].values, df_b["patient_id"].values,
                           label_a, label_b)
        a, b = df_a[m].dropna().values, df_b[m].dropna().values
        r["cohens_d"] = cohens_d_paired(a, b) if paired else cohens_d_pooled(a, b)
        r["metric"] = m
        raw[m] = r["p"]
        rows.append(r)
    adj = holm_split_primary_derived(raw)
    out = pd.DataFrame(rows).rename(columns={"p": "p_raw"})
    out["p_holm"] = out["metric"].map(adj)
    cols = ["metric", "beta", "se", "z", "p_raw", "p_holm", "cohens_d", "n", "n_patients"]
    return out[cols]
