"""Localization metrics used throughout the paper (Figures 4-5, Supplementary 4-6).

All metrics are computed at the level of the 994 cortical regions. A region is
"active" if its activation exceeds the trial-specific Otsu threshold; the
reference set is the surgical resection (or the intracranial-EEG seizure onset
zone for Supplementary Figure 6).

    localization error  symmetric mean nearest-neighbour distance between the
                        active set and the reference set (mm)
    precision           |active ∩ reference| / |active|
    sensitivity         |active ∩ reference| / |reference|
    specificity         balanced estimate, see specificity_balanced()
    F1                  harmonic mean of precision and sensitivity
    geometric mean      sqrt(specificity * sensitivity)

Specificity deserves a note. The reference set is small (about 20 of 994
regions), so the naive true-negative rate over the whole cortex is ~0.99 for
any estimate and carries no information. The paper therefore samples two
matched negative sets, each the size of the reference:

    near   : regions within 20 mm of the reference, sampled at random
    far    : the most strongly activated regions more than 40 mm from it

computes specificity on each, and reports their mean. The near set tests
whether the estimate bleeds into adjacent cortex; the far set tests whether it
produces spurious distant activity. The near sample is random; pass a seeded
`rng` for reproducibility.
"""
import numpy as np
from skimage.filters import threshold_otsu


def otsu_regions(activation):
    """Indices of regions whose activation exceeds the Otsu threshold."""
    a = np.asarray(activation, float)
    if np.ptp(a) == 0:
        return np.array([], dtype=int)
    return np.where(a > threshold_otsu(a))[0]


def localization_error(active, reference, dis):
    """Symmetric mean nearest-neighbour distance, in the units of `dis`."""
    D = dis[np.ix_(active, reference)]
    return 0.5 * (D.min(axis=0).mean() + D.min(axis=1).mean())


def spatial_dispersion(activation, active, reference, dis):
    """Activation-weighted RMS distance from active regions to the reference."""
    d_min = dis[np.ix_(active, reference)].min(axis=1)
    w = np.asarray(activation, float)[active] ** 2
    return float(np.sqrt((d_min ** 2 * w).sum() / w.sum()))


def specificity_balanced(activation, active, reference, dis,
                         near_mm=20.0, far_mm=40.0, rng=None):
    """Mean of specificity on a near-negative and a far-negative sample."""
    rng = np.random.default_rng(0) if rng is None else rng
    activation = np.asarray(activation, float)
    n_pos = len(reference)
    all_idx = np.arange(len(activation))
    d_to_ref = dis[reference, :].min(axis=0)             # distance of every region to the reference

    near = np.setdiff1d(np.where(d_to_ref <= near_mm)[0], reference)
    near_s = rng.choice(near, n_pos, replace=len(near) < n_pos)

    far_ok = (d_to_ref > far_mm) & ~np.isin(all_idx, reference)
    far_ranked = all_idx[np.argsort(-activation)]
    far = far_ranked[far_ok[far_ranked]]
    far_s = far[:n_pos] if len(far) >= n_pos else rng.choice(far, n_pos, replace=True)

    def spec(neg):
        fp = np.isin(neg, active).sum()
        tn = n_pos - fp
        return tn / (tn + fp)

    return 0.5 * (spec(near_s) + spec(far_s))


def all_metrics(activation, reference, dis, rng=None):
    """Every metric for one event. Returns a dict, or None if nothing is active."""
    active = otsu_regions(activation)
    if active.size == 0:
        return None
    overlap = np.isin(active, reference).sum()
    pr = overlap / len(active)
    re = overlap / len(reference)
    sp = specificity_balanced(activation, active, reference, dis, rng=rng)
    f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0.0
    return dict(le=localization_error(active, reference, dis),
                sd=spatial_dispersion(activation, active, reference, dis),
                pr=pr, re=re, sp=sp, f1=f1, gm=float(np.sqrt(sp * re)),
                n_active=int(active.size))
