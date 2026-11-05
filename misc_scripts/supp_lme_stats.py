"""Dump LME test statistics (beta, SE, z) for Supplementary Figures 4 and 6.

Fits the linear mixed-effects models behind two supplementary figures:
  - suppfig_sf_nsf()  ->  Supp Fig 4 : metric ~ cohort  + (1|patient_id), per event
  - suppfig3()        ->  Supp Fig 6 : le     ~ event_type + (1|patient_id), per event
The originals printed/saved only p-values and betas; the Scientific Editor asks for
the test statistic values in the legends, so we re-fit and keep SE / z as well.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_ROOT = os.path.join(ROOT, "model_weights")
R1_STATS = os.path.join(ROOT, "misc_scripts")
OUT_DIR = os.path.join(ROOT, "misc_scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statistical_tests import holm_split_primary_derived  # noqa: E402

METRICS = [("le", "Localization Error (mm)"), ("sp", "Specificity"),
           ("pr", "Precision"), ("re", "Sensitivity"),
           ("gm", "Geometric Mean"), ("f1", "F1 Score")]
PRIMARY, DERIVED = ["le", "sp", "pr", "re"], ["gm", "f1"]


def fit(long, formula, group_col="patient_id"):
    md = smf.mixedlm(formula, long, groups=long[group_col])
    mdf = md.fit(method="lbfgs", reml=False)
    cn = [n for n in mdf.params.index
          if n != "Intercept" and not n.startswith("Group Var")][0]
    return cn, float(mdf.params[cn]), float(mdf.bse[cn]), \
        float(mdf.tvalues[cn]), float(mdf.pvalues[cn])


def supp4():
    """Supp Fig 4 - seizure-free vs non-seizure-free cohort, per pHFO event."""
    sf = pd.read_csv(os.path.join(R1_STATS, "per_event_pHFO_metrics.csv"))
    sf = sf[sf["method"] == "DeepSIF"].copy(); sf["cohort"] = "SF"
    nsf = pd.read_csv(os.path.join(R1_STATS, "per_event_nonSF_metrics.csv"))
    nsf["cohort"] = "nonSF"
    df = pd.concat([sf, nsf], ignore_index=True)
    df["gm"] = np.sqrt(df["sp"] * df["re"])

    rows, raw_p = [], {}
    for m, _ in METRICS:
        long = df[["patient_id", "cohort", m]].dropna().rename(columns={m: "y"})
        cn, beta, se, z, p = fit(long, "y ~ cohort")
        raw_p[m] = p
        rows.append({"figure": "SuppFig4", "metric": m, "contrast": cn,
                     "n_events": int(len(long)),
                     "n_events_SF": int((long["cohort"] == "SF").sum()),
                     "n_events_nonSF": int((long["cohort"] == "nonSF").sum()),
                     "n_patients": int(long["patient_id"].nunique()),
                     "beta": beta, "se": se, "z": z, "p_raw": p})
    adj = holm_split_primary_derived(raw_p, PRIMARY, DERIVED)
    holm = [adj[m] for m, _ in METRICS]
    for r, ph in zip(rows, holm):
        r["p_holm"] = float(ph)
    return rows


def supp6():
    """Supp Fig 6 - SOZ localization error, pHFO vs aHFO (aHFO = pHFO u iHFO)."""
    # Internal identifiers of the patients with intracranial EEG are not
    # distributed. Set this to the identifiers used in your own copy of the data.
    SOZ_PATIENTS = []
    if not SOZ_PATIENTS:
        raise SystemExit("SOZ_PATIENTS must be set - see the comment above")
    pmat = loadmat(os.path.join(R1_STATS, "metrics_soz_pHFO_SF.mat"), squeeze_me=False)
    imat = loadmat(os.path.join(R1_STATS, "metrics_soz_aHFO_SF.mat"), squeeze_me=False)
    p_vals = np.asarray(pmat["metrics"][0, 0][0, 0]).flatten().astype(float)
    i_vals = np.asarray(imat["metrics"][0, 0][0, 0]).flatten().astype(float)

    p_csv = pd.read_csv(os.path.join(R1_STATS, "per_event_pHFO_metrics.csv"))
    p_csv = p_csv[p_csv["method"] == "DeepSIF"]
    i_csv = pd.read_csv(os.path.join(R1_STATS, "per_event_aHFO_metrics.csv"))
    p_counts = [int((p_csv["patient_id"] == p).sum()) for p in SOZ_PATIENTS]
    i_counts = [int((i_csv["patient_id"] == p).sum()) for p in SOZ_PATIENTS]
    assert sum(p_counts) == len(p_vals) and sum(i_counts) == len(i_vals)
    p_pids = np.concatenate([np.full(c, p, dtype=int) for c, p in zip(p_counts, SOZ_PATIENTS)])
    i_pids = np.concatenate([np.full(c, p, dtype=int) for c, p in zip(i_counts, SOZ_PATIENTS)])

    a_vals = np.concatenate([p_vals, i_vals])
    a_pids = np.concatenate([p_pids, i_pids])
    long = pd.DataFrame({
        "le": np.concatenate([p_vals, a_vals]),
        "event_type": ["pHFO"] * len(p_vals) + ["aHFO"] * len(a_vals),
        "patient_id": np.concatenate([p_pids, a_pids]),
    })
    cn, beta, se, z, p = fit(long, "le ~ event_type")

    def med_iqr(x):
        q1, m, q3 = np.percentile(x, [25, 50, 75])
        return m, q1, q3
    mp, q1p, q3p = med_iqr(p_vals)
    ma, q1a, q3a = med_iqr(a_vals)
    return [{"figure": "SuppFig6", "metric": "le", "contrast": cn,
             "n_events": int(len(long)), "n_pHFO": int(len(p_vals)),
             "n_aHFO": int(len(a_vals)), "n_iHFO": int(len(i_vals)),
             "n_patients": int(long["patient_id"].nunique()),
             "beta": beta, "se": se, "z": z, "p_raw": p, "p_holm": p,
             "median_pHFO": mp, "iqr_pHFO_lo": q1p, "iqr_pHFO_hi": q3p,
             "median_aHFO": ma, "iqr_aHFO_lo": q1a, "iqr_aHFO_hi": q3a}]


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    r4, r6 = supp4(), supp6()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    df4 = pd.DataFrame(r4); df6 = pd.DataFrame(r6)
    print("=== Supplementary Figure 4 (SF vs non-SF, per-event LME) ===")
    print(df4.to_string(index=False))
    print("\n=== Supplementary Figure 6 (SOZ: pHFO vs aHFO, per-event LME) ===")
    print(df6.to_string(index=False))
    out = os.path.join(OUT_DIR, "supp4_supp6_lme_teststats.csv")
    pd.concat([df4, df6], ignore_index=True, sort=False).to_csv(out, index=False)
    print(f"\nWrote {out}")
