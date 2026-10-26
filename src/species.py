"""Species-level indices and rates of change.

* Index: individuals of the species / point counts surveyed (zero-filled by
  construction: every surveyed point is in the denominator).
* Occurrence: share of point counts in which the species was detected
  (Bernoulli MLE).
* Global rate of change, two estimators:
    A. log-linear slope of the annual index (12 values), CI by cluster
       bootstrap of the whole annual series;
    B. visit-level linear regression of the per-point count with transect
       fixed effects, survey conditions and observer fixed effects; the
       year coefficient is expressed as % of the species' mean count.
       B adjusts for effort composition and observers; A does not.
* Trend classes follow the logic of the Pan-European Common Bird Monitoring
  Scheme: the position of the 95 % CI of the annual rate relative to 0 and
  to +/-5 % per year (no hypothesis test involved).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indices as I, models as M


def point_species_table(records, point_visits, species):
    """Wide table: one row per point count, one column per species (counts, zeros filled)."""
    cnt = (records[records["species"].isin(species)]
           .groupby(["visit_id", "point", "species"])["count"].sum().unstack(fill_value=0))
    out = point_visits[["visit_id", "point", "transect", "year"]].merge(
        cnt.reset_index(), on=["visit_id", "point"], how="left")
    out[list(species)] = out[list(species)].fillna(0)
    assert len(out) == len(point_visits)
    return out


def annual_index_series(P, sp, W, tr, yrs):
    """Annual index and bootstrap replicates for one species (shared weights W)."""
    num = (P.groupby(["transect", "year"])[sp].sum().unstack()
           .reindex(index=tr, columns=yrs).fillna(0).to_numpy())
    den = (P.groupby(["transect", "year"]).size().unstack()
           .reindex(index=tr, columns=yrs).fillna(0).to_numpy())
    est = num.sum(0) / den.sum(0)
    reps = (W @ num) / (W @ den)
    return est, reps


def occurrence(P, sp, W, tr, yrs):
    """Annual detection rate (share of point counts with the species) + bootstrap replicates."""
    det = P.assign(_d=(P[sp] > 0).astype(float))
    return annual_index_series(det, "_d", W, tr, yrs)


def wilson(p, n, z=1.959964):
    """Wilson score interval for a binomial proportion (assumes independent trials)."""
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z / (1 + z ** 2 / n) * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return centre - half, centre + half


def rate_annual_loglinear(est, reps, yrs):
    """Estimator A: % per year from the slope of log(annual index); percentile CI."""
    t = np.asarray(yrs) - yrs[0]
    ok = (reps > 0).all(1)
    b = np.polyfit(t, np.log(est), 1)[0]
    sl = np.polyfit(t, np.log(reps[ok]).T, 1)[0]
    lo, hi = np.quantile(sl, [0.025, 0.975])
    return M.pct_per_year(b), M.pct_per_year(lo), M.pct_per_year(hi), 1 - ok.mean()


def rate_visit_model(vv, sp, numeric, B, seed):
    """Estimator B: visit-level OLS with transect FE (within), conditions and observer FE.

    Response = individuals of the species per point count in the visit. The
    year coefficient (individuals per point per year) is divided by the
    species' mean to give % of the mean per year.
    """
    def other(d):
        X, n = M.design(d, numeric=numeric, fixed_effects=["observer"])
        return X[:, 1:], n[1:]
    est, reps = M.cluster_bootstrap_within(vv, other, sp, ["year"], B=B, seed=seed)
    mu = vv[sp].mean()
    lo, hi = np.quantile(reps[:, 0], [0.025, 0.975])
    return 100 * est[0] / mu, 100 * lo / mu, 100 * hi / mu, 100 * reps[:, 0].std(ddof=1) / mu


def trend_class(lo, hi, threshold=5.0):
    """Classify from the 95 % CI of the annual rate (% per year)."""
    if lo > threshold:
        return "strong increase"
    if lo > 0:
        return "moderate increase"
    if hi < -threshold:
        return "steep decline"
    if hi < 0:
        return "moderate decline"
    if lo >= -threshold and hi <= threshold:
        return "stable"
    return "uncertain"


def shrink(est, se):
    """Empirical-Bayes shrinkage of several rates towards their precision-weighted mean.

    theta_s* = mu + w_s (theta_s - mu), w_s = tau^2 / (tau^2 + se_s^2), with the
    between-species variance tau^2 estimated by the method of moments
    (variance of the estimates minus their average sampling variance).
    This is a shrinkage estimator with a data-driven weight.
    """
    est, se = np.asarray(est, float), np.asarray(se, float)
    tau2 = max(est.var(ddof=1) - np.mean(se ** 2), 0.0)
    w = tau2 / (tau2 + se ** 2)
    mu = np.sum(est / se ** 2) / np.sum(1 / se ** 2)
    return mu + w * (est - mu), w, tau2, mu
