"""Linear regression estimated by ordinary least squares, written out explicitly.

Why hand-written: the estimators are short, and writing them out keeps every
quantity reported in the notebooks traceable to an explicit formula.

Likelihood view: for y = X beta + eps with eps ~ N(0, sigma^2 I)
i.i.d., the log-likelihood is
    l(beta, sigma^2) = -n/2 log(2 pi sigma^2) - ||y - X beta||^2 / (2 sigma^2),
so the MLE of beta is the least-squares solution and the MLE of sigma^2 is
RSS / n (biased; RSS / (n - p) is the unbiased version).
The "classical" standard errors sqrt(diag(sigma^2 (X'X)^-1)) come from the
inverse Fisher information and assume independent observations; the cluster
bootstrap (resampling whole transects) relaxes that assumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Design matrix
# --------------------------------------------------------------------------- #
def design(df, numeric=(), fixed_effects=(), year_col="year", year_center=2014,
           year_as_factor=False, year_poly=1):
    """Build X with an intercept, year term(s), numeric covariates and dummies.

    * ``year`` enters as (year - 2014), so the intercept is the 2014 level and
      the coefficient is the change per year on the response scale; with
      ``year_poly=2`` a quadratic term is added; with ``year_as_factor`` one
      dummy per year (2014 = reference) gives annual adjusted indices.
    * ``fixed_effects``: categorical columns coded as dummies with the first
      (most frequent) level as reference.
    Returns (X, column names).
    """
    cols, names = [np.ones(len(df))], ["intercept"]
    t = (df[year_col] - year_center).to_numpy(float)
    if year_as_factor:
        for y in sorted(df[year_col].unique())[1:]:
            cols.append((df[year_col] == y).to_numpy(float)); names.append(f"year[{y}]")
    else:
        for k in range(1, year_poly + 1):
            cols.append(t ** k); names.append("year" if k == 1 else f"year^{k}")
    for c in numeric:
        cols.append(df[c].to_numpy(float)); names.append(c)
    for fe in fixed_effects:
        levels = df[fe].value_counts().index  # reference = most frequent level
        for lv in levels[1:]:
            cols.append((df[fe] == lv).to_numpy(float)); names.append(f"{fe}[{lv}]")
    return np.column_stack(cols), names


# --------------------------------------------------------------------------- #
# OLS fit
# --------------------------------------------------------------------------- #
class OLSFit:
    """Least-squares fit with the usual summaries."""

    def __init__(self, X, y, names=None):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.n, self.p = X.shape
        self.beta, _, self.rank, _ = np.linalg.lstsq(X, y, rcond=None)
        self.fitted = X @ self.beta
        self.resid = y - self.fitted
        self.rss = float(self.resid @ self.resid)
        self.sigma2_mle = self.rss / self.n                    # biased MLE
        self.sigma2 = self.rss / (self.n - self.rank)          # unbiased
        tss = float(((y - y.mean()) ** 2).sum())
        self.r2 = 1 - self.rss / tss
        self.adj_r2 = 1 - (1 - self.r2) * (self.n - 1) / (self.n - self.rank)
        self.loglik = -0.5 * self.n * (np.log(2 * np.pi * self.sigma2_mle) + 1)
        self.aic = 2 * (self.rank + 1) - 2 * self.loglik       # +1 for sigma^2
        self.bic = np.log(self.n) * (self.rank + 1) - 2 * self.loglik
        # classical SE from sigma^2 (X'X)^-1 (pseudo-inverse guards against
        # rank deficiency, e.g. an observer seen on a single transect)
        self.cov = self.sigma2 * np.linalg.pinv(X.T @ X)
        self.se = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        self.names = list(names) if names is not None else [f"x{i}" for i in range(self.p)]
        self.X = X

    def coef(self, name):
        return self.beta[self.names.index(name)]

    def se_of(self, name):
        return self.se[self.names.index(name)]

    def summary_row(self):
        return dict(n=self.n, parameters=self.rank + 1, R2=self.r2, adj_R2=self.adj_r2,
                    sigma_resid=np.sqrt(self.sigma2), loglik=self.loglik, AIC=self.aic, BIC=self.bic)


def pct_per_year(beta):
    """Coefficient of (year) on a log response -> % change per year."""
    return 100 * (np.exp(beta) - 1)


# --------------------------------------------------------------------------- #
# Cluster bootstrap of regression coefficients
# --------------------------------------------------------------------------- #
def cluster_bootstrap_coef(df, build, ycol, coef_names, cluster="transect", B=1000, seed=0):
    """Pairs cluster bootstrap: resample whole clusters with replacement, refit.

    ``build(df) -> (X, names)``. A cluster drawn twice is relabelled as two
    distinct clusters so that cluster fixed effects remain identifiable.
    Returns an array (B, len(coef_names)).
    """
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    out = np.full((B, len(coef_names)), np.nan)
    for b in range(B):
        draw = rng.choice(keys, size=len(keys), replace=True)
        parts = []
        for j, k in enumerate(draw):
            g = groups[k].copy()
            g[cluster] = f"{k}#{j}"
            parts.append(g)
        bd = pd.concat(parts, ignore_index=True)
        X, names = build(bd)
        beta = np.linalg.lstsq(X, bd[ycol].to_numpy(float), rcond=None)[0]
        out[b] = [beta[names.index(c)] if c in names else np.nan for c in coef_names]
    return out


# --------------------------------------------------------------------------- #
# Cross-validated prediction error (bias-variance trade-off)
# --------------------------------------------------------------------------- #
def cv_mse(df, build, ycol, k=10, seed=0, stratify=None):
    """K-fold cross-validated mean squared prediction error.

    The design is built once on the full data (so every dummy column exists);
    a level absent from a training fold gets a zero coefficient, i.e. it is
    predicted at the reference level. With ``stratify`` (e.g. 'transect') the
    folds are assigned within each stratum so every transect is seen in
    training. Returns (mean MSE over folds, standard error across folds).
    """
    rng = np.random.default_rng(seed)
    X, _ = build(df)
    y = df[ycol].to_numpy(float)
    fold = np.empty(len(df), int)
    if stratify is None:
        fold[:] = rng.permutation(np.arange(len(df)) % k)
    else:
        for _, idx in df.groupby(stratify).indices.items():
            fold[idx] = rng.permutation(np.arange(len(idx)) + rng.integers(k)) % k
    mses = []
    for f in range(k):
        tr, te = fold != f, fold == f
        beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        mses.append(np.mean((y[te] - X[te] @ beta) ** 2))
    mses = np.array(mses)
    return mses.mean(), mses.std(ddof=1) / np.sqrt(k)


# --------------------------------------------------------------------------- #
# Fast cluster bootstrap for models with cluster (transect) fixed effects
# --------------------------------------------------------------------------- #
def within_transform(df, cols, cluster="transect"):
    """Subtract cluster means (Frisch-Waugh-Lovell).

    Regressing within-transformed y on within-transformed X gives exactly the
    same slope coefficients as including one dummy per cluster.
    """
    g = df.groupby(cluster)[list(cols)]
    return (df[list(cols)] - g.transform("mean")).to_numpy(float)


def cluster_bootstrap_within(df, build_other, ycol, coef_names, cluster="transect", B=1000, seed=0):
    """Cluster bootstrap of slope coefficients in a cluster-fixed-effects model.

    ``build_other(df) -> (X, names)`` builds every regressor EXCEPT the
    intercept and the cluster dummies. Rows are within-transformed once; a
    resampled cluster keeps its own demeaned rows, which is exactly the
    fixed-effects estimator on the resampled data (each copy of a cluster is
    its own cluster). Returns (point estimate, replicates (B, k)).
    """
    X, names = build_other(df)
    Z = pd.DataFrame(np.column_stack([X, df[ycol].to_numpy(float)]))
    Z[cluster] = df[cluster].to_numpy()
    Zw = within_transform(Z, list(range(X.shape[1] + 1)), cluster)
    Xw, yw = Zw[:, :-1], Zw[:, -1]
    idx_cols = [names.index(c) for c in coef_names]
    est = np.linalg.lstsq(Xw, yw, rcond=None)[0][idx_cols]
    rows = df.groupby(cluster).indices
    keys = list(rows)
    rng = np.random.default_rng(seed)
    reps = np.empty((B, len(coef_names)))
    for b in range(B):
        sel = np.concatenate([rows[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        reps[b] = np.linalg.lstsq(Xw[sel], yw[sel], rcond=None)[0][idx_cols]
    return est, reps
