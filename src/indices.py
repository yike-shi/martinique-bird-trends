"""Standardised indices and their uncertainty.

Annual indices here are *ratio estimators*: a total over all sampling units of
a year divided by the effort of that year, e.g.

    abundance index_t = (individuals counted in year t) / (point counts in t).

Under a Poisson model for the count of a point, this ratio is exactly the MLE
(and method-of-moments estimate) of the Poisson mean. Absences are
true zeros: the denominator contains every surveyed point, including points
where nothing (or not the species of interest) was recorded.

Uncertainty: points and visits of one transect are not independent (the
independence check fails, see notebook 02), so we use a *cluster*
bootstrap that resamples whole transects with replacement, keeping each
transect's full time series together (dependent data require a block
bootstrap). Resampling transects is implemented with multinomial
weights: drawing transect k w_k times is equivalent to weighting its totals by
w_k, which makes B = 2000+ replicates instantaneous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln


# --------------------------------------------------------------------------- #
# Cluster bootstrap machinery
# --------------------------------------------------------------------------- #
def cluster_weights(n_clusters, B, seed):
    """(B, K) matrix: how many times each of K clusters is drawn in replicate b."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=B)


def transect_year_table(df, value_cols, cluster="transect", year="year"):
    """Sum value columns into dense (K clusters x T years) arrays."""
    g = df.groupby([cluster, year])[list(value_cols)].sum()
    clusters = sorted(df[cluster].unique())
    years = sorted(df[year].unique())
    full = g.reindex(pd.MultiIndex.from_product([clusters, years])).fillna(0)
    arrays = {c: full[c].to_numpy().reshape(len(clusters), len(years)) for c in value_cols}
    return arrays, clusters, years


def boot_ratio(num, den, W):
    """Bootstrap replicates of sum(num)/sum(den) per year: (B, T)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ num) / (W @ den)


def percentile_ci(reps, level=0.95):
    a = (1 - level) / 2
    return np.nanquantile(reps, a, axis=0), np.nanquantile(reps, 1 - a, axis=0)


def basic_ci(estimate, reps, level=0.95):
    """Basic (reflection) bootstrap interval: [2 est - q_hi, 2 est - q_lo]."""
    lo, hi = percentile_ci(reps, level)
    return 2 * estimate - hi, 2 * estimate - lo


def annual_ratio_index(df, num_col, den_col, B=2000, seed=0, base_year=2014, level=0.95):
    """Annual ratio index with cluster-bootstrap CIs, also as base-100 index.

    The base-100 index (index_t / index_base * 100) is bootstrapped jointly, so
    its CI accounts for the uncertainty of the base year as well.
    """
    arr, clusters, years = transect_year_table(df, [num_col, den_col])
    num, den = arr[num_col], arr[den_col]
    est = num.sum(0) / den.sum(0)
    W = cluster_weights(len(clusters), B, seed)
    reps = boot_ratio(num, den, W)
    b = years.index(base_year)
    rel = reps / reps[:, [b]] * 100
    lo, hi = percentile_ci(reps, level)
    rlo, rhi = percentile_ci(rel, level)
    out = pd.DataFrame({"estimate": est, "ci_low": lo, "ci_high": hi,
                        "se_boot": np.nanstd(reps, axis=0, ddof=1),
                        "index100": est / est[b] * 100, "index100_low": rlo, "index100_high": rhi,
                        "effort": den.sum(0)}, index=pd.Index(years, name="year"))
    return out, reps


# --------------------------------------------------------------------------- #
# Rarefied (equal-effort) species richness
# --------------------------------------------------------------------------- #
def _log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rarefied_richness(incidence, n_units, m):
    """Expected number of species in m sampling units drawn without replacement.

    incidence : array (..., S) number of units in which each species occurs
    n_units   : array (...)     number of units N
    Formula (incidence-based rarefaction):
        E[S_m] = sum_s [ 1 - C(N - f_s, m) / C(N, m) ],
    with the ratio = 0 when N - f_s < m. Returns NaN where N < m.
    """
    f = np.asarray(incidence, float)
    N = np.asarray(n_units, float)[..., None]
    with np.errstate(invalid="ignore"):
        ratio = np.where(N - f >= m, np.exp(_log_comb(N - f, m) - _log_comb(N, m)), 0.0)
    s = np.where(f > 0, 1 - ratio, 0.0).sum(-1)
    return np.where(N[..., 0] >= m, s, np.nan)
