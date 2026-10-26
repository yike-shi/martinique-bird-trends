"""Sampling-effort tables and zero-filling.

The observation sheet only lists birds that *were* detected. To compute
effort, occurrence rates or per-visit means we first reconstruct the sampling
frame:

* ``point_visits``: one row per (visit, point) actually surveyed, i.e. one
  5-minute point count. A point visit exists if at least one row (including an
  empty 'RAS' row) was recorded for it.
* ``visits``: one row per transect visit (transect x year x passage), with the
  number of points completed, timing, weather, observer and summary counts.

``zero_fill`` then crosses the sampling frame with a species list so that a
species not detected during a point (or visit) gets an explicit 0.
Caveat: a point at which *no bird at all* was detected and nothing was written
down cannot be distinguished from a point that was skipped; with ~10 birds per
point on average this is expected to be very rare.
"""
import numpy as np
import pandas as pd


def _mode(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else pd.NA


def build_point_visits(df):
    """One row per 5-minute point count."""
    rec = df[df["is_record"]]
    pv = (df.groupby(["visit_id", "point"], as_index=False)
            .agg(transect=("transect", "first"), year=("year", "first"),
                 passage=("passage", "first"), date=("date", "first"),
                 observer=("observer", _mode), start_min=("start_min", "min")))
    agg = (rec.groupby(["visit_id", "point"])
              .agg(n_ind=("count", "sum"), n_ind_lt50=("count_lt50", "sum"),
                   n_aud=("count_aud", "sum"), n_vis=("count_vis", "sum"),
                   n_flight=("flight", "sum")))
    agg["richness"] = (rec[rec["is_species"]].groupby(["visit_id", "point"])["species"]
                         .nunique())
    pv = pv.merge(agg, on=["visit_id", "point"], how="left")
    pv[agg.columns] = pv[agg.columns].fillna(0)
    return pv


def build_visits(df, sites):
    """One row per transect visit, with effort and context covariates."""
    rec = df[df["is_record"]]
    v = (df.groupby("visit_id", as_index=False)
           .agg(transect=("transect", "first"), year=("year", "first"),
                passage=("passage", "first"), date=("date", "first"),
                doy=("doy", "first"),
                observer=("observer", _mode), n_observer_labels=("observer", "nunique"),
                n_points=("point", "nunique"),
                start_min=("start_min", "min"), end_min=("start_min", "max"),
                clouds=("clouds", _mode), rain=("rain", _mode),
                wind=("wind", _mode), visibility=("visibility", _mode),
                mentored=("mentored", "any"), done_by_other=("done_by_other", "any")))
    s = (rec.groupby("visit_id")
            .agg(n_records=("species", "size"), n_ind=("count", "sum"),
                 n_ind_lt50=("count_lt50", "sum"), n_aud=("count_aud", "sum"),
                 n_vis=("count_vis", "sum"), n_flight=("flight", "sum")))
    rich = (rec[rec["is_species"]].groupby("visit_id")["species"].nunique()
              .rename("richness"))
    v = v.merge(s, on="visit_id", how="left").merge(rich, on="visit_id", how="left")
    v[list(s.columns) + ["richness"]] = v[list(s.columns) + ["richness"]].fillna(0)
    hab = sites.groupby("transect")["habitat"].agg(_mode).rename("habitat")
    v = v.merge(hab, left_on="transect", right_index=True, how="left")
    v["duration_min"] = v["end_min"] - v["start_min"]
    v["complete"] = v["n_points"] == 10
    # Both passages of a year recorded on the same date: impossible by
    # protocol (visits are weeks apart), so one date is probably a copy of the
    # other. Kept, but flagged: day-of-year is unreliable for these visits.
    v["same_day_passages"] = v.duplicated(["transect", "year", "date"], keep=False)
    return v.sort_values(["year", "transect", "passage"]).reset_index(drop=True)


def zero_fill(records, frame, species, keys=("visit_id", "point"), value="count"):
    """Cross a sampling frame with a species list, filling non-detections with 0.

    Parameters
    ----------
    records : clean records (rows with ``is_record``)
    frame   : DataFrame with one row per sampling unit (columns ``keys``)
    species : iterable of species names to include
    keys    : sampling-unit identifiers ('visit_id', 'point') or ('visit_id',)
    value   : count column to aggregate
    """
    keys = list(keys)
    species = pd.Index(sorted(set(species)), name="species")
    agg = (records[records["species"].isin(species)]
             .groupby(keys + ["species"])[value].sum())
    full = pd.MultiIndex.from_product(
        [frame[keys].drop_duplicates().itertuples(index=False, name=None), species],
        names=["_unit", "species"])
    out = pd.DataFrame(index=full).reset_index()
    out[keys] = pd.DataFrame(out["_unit"].tolist(), index=out.index)
    out = out.drop(columns="_unit")
    out = out.merge(agg.rename(value).reset_index(), on=keys + ["species"], how="left")
    out[value] = out[value].fillna(0)
    out["detected"] = (out[value] > 0).astype(int)
    # sanity: totals must be preserved
    assert np.isclose(out[value].sum(), agg.sum()), "zero-fill changed the total count"
    return out


def add_observer_experience(visits):
    """Observer experience at the time of each visit.

    ``obs_prev_seasons``  number of earlier years in which the observer made at
                          least one visit in this programme (0 = first year);
    ``obs_first_season``  True in the observer's first year.
    Caveat: experience is left-censored in 2014 (programme start): everyone is
    'new' that year, even observers who were experienced elsewhere.
    """
    v = visits.copy()
    seasons = v.groupby("observer")["year"].apply(lambda s: np.sort(s.unique()))
    v["obs_prev_seasons"] = [int((seasons[o] < y).sum()) for o, y in zip(v["observer"], v["year"])]
    v["obs_first_season"] = v["obs_prev_seasons"].eq(0)
    return v
