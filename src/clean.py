"""Cleaning of the observation log.

Every rule below was motivated by a concrete problem found while auditing the
raw workbook (see notebook 01). Each step appends a line to a quality log
(``QualityLog``) stating what was changed and how many rows were affected, so
the whole pipeline is auditable and the report can quote exact numbers.

Design principles
-----------------
* Never silently drop data: rows are removed only for clear data-entry
  artefacts (whole visits typed twice, conflicting fragments), and the count is
  logged.
* Prefer the most primitive record: individual counts are re-computed from the
  eight distance x detection cells, because the workbook's total columns are
  broken formulas in a few hundred rows.
* Keep raw columns next to cleaned ones (``*_raw``) for traceability.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Quality log
# --------------------------------------------------------------------------- #
class QualityLog:
    """Collects (step, issue, action, rows affected) records."""

    def __init__(self):
        self.records = []

    def add(self, step, issue, action, n_rows):
        self.records.append(dict(step=step, issue=issue, action=action,
                                 rows_affected=int(n_rows)))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def _squash(s: pd.Series) -> pd.Series:
    """Strip and collapse internal whitespace (keeps case and accents)."""
    return s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def _key(s: pd.Series) -> pd.Series:
    """Case- and whitespace-insensitive matching key."""
    return _squash(s).str.casefold()


def _ascii_key(x: str) -> str:
    """Accent-insensitive key used for fuzzy observer matching."""
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", x.lower())


# --------------------------------------------------------------------------- #
# Individual steps
# --------------------------------------------------------------------------- #
def harmonise_transects(df, sites, log):
    """Map transect names to the canonical spelling of the GPS-MILIEU sheet.

    Raw names differ only by case / trailing spaces (e.g. 'BORELIE' vs
    'Borelie') plus one singular/plural variant ('Desmarinière').
    """
    canon = {k: v for k, v in zip(_key(sites["transect"]), sites["transect"])}
    manual = {"desmarinière": "Desmarinières"}
    k = _key(df["transect_raw"])
    out = k.map(canon)
    out = out.fillna(k.map(manual))
    changed = (out != _squash(df["transect_raw"])) & out.notna()
    log.add("transect names", "case/whitespace/spelling variants of the same transect",
            "mapped to GPS-MILIEU spelling", changed.sum())
    if out.isna().any():
        raise ValueError(f"Unmapped transects: {sorted(k[out.isna()].unique())}")
    df["transect"] = out.astype("string")
    return df


def harmonise_observers(df, log):
    """Unify observer spellings ('MARTIN Anne', 'Anne MARTIN', 'Martin Anne ').

    Names are matched on an accent/case/order-insensitive key (sorted tokens).
    Multi-person teams (e.g. 'MARTIN Anne - DUPONT Paul') are kept as a
    single 'observer unit' because that is how the visit was recorded.
    """
    sq = _squash(df["observer_raw"])
    tok_key = sq.map(lambda x: " ".join(sorted(_ascii_key(t) for t in re.split(r"[\s/\-]+", x) if t))
                     if isinstance(x, str) else x)
    # Canonical label = most frequent raw spelling within each key
    canon = (pd.DataFrame({"k": tok_key, "s": sq}).dropna()
             .groupby("k")["s"].agg(lambda s: s.value_counts().index[0]))
    df["observer"] = tok_key.map(canon).astype("string")
    n_changed = (df["observer"] != df["observer_raw"]).sum()
    log.add("observer names", "whitespace / word-order / case variants of the same person",
            "unified on a token key", n_changed)
    return df


def parse_dates_and_codes(df, log):
    """Parse dates, passage, point, weather; null out impossible codes."""
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Unparseable dates in raw data")
    df["year"] = df["date"].dt.year.astype(int)
    df["passage"] = pd.to_numeric(df["passage_raw"]).astype(int)
    df["point"] = pd.to_numeric(df["point"]).astype(int)

    # Department: the survey is in Martinique (972). Other values (961-974)
    # form a +1 sequence typical of an Excel drag-fill; the column is unused.
    bad_dept = (df["dept"] != 972).sum()
    log.add("department code", "values 961-974 (Excel drag-fill artefact)",
            "ignored (column not used; all transects are in Martinique)", bad_dept)

    # Weather: valid codes are 1-3. Values such as -5..0 (drag-fill) or 4 are
    # set to missing rather than guessed.
    for col, valid in C.WEATHER_VALID.items():
        x = pd.to_numeric(_squash(df[col].astype(str)), errors="coerce")
        invalid = x.notna() & ~x.isin(valid)
        missing_before = x.isna().sum()
        x[invalid] = np.nan
        df[col] = x.astype("Int64")
        log.add(f"weather: {col}", f"out-of-range codes ({invalid.sum()}) or blank ({missing_before})",
                "set to missing", invalid.sum() + missing_before)

    # Start time (per point): keep as minutes after midnight.
    # Stored in several forms: time objects ('06:20:00'), text ('7:42',
    # '5h48') or full datetimes ('1900-01-07 06:00:00'). We take the trailing
    # hh:mm in all cases.
    txt = _squash(df["start_time"].astype(str))
    hm = txt.str.extract(r"(\d{1,2})[:hH](\d{2})(?::\d{2})?$").astype(float)
    df["start_min"] = hm[0] * 60 + hm[1]
    df.loc[(df["start_min"] < 0) | (df["start_min"] >= 24 * 60), "start_min"] = np.nan
    log.add("start time", "missing or unparseable start time", "left missing",
            df["start_min"].isna().sum())
    return df


def drop_double_entered_visits(df, log, threshold=0.4):
    """Remove visits that were typed into the workbook twice.

    Exact duplicate rows are concentrated in a few visits where ~50 % of rows
    are duplicates, i.e. the whole visit was entered twice (mostly 2016). Those
    duplicates are removed. Isolated exact duplicates elsewhere (a few rows per
    visit) can be genuine separate contacts with identical profiles, so they
    are kept and flagged (``isolated_exact_dup``) for sensitivity analysis.
    """
    cols = [c for c in C.OBS_COLUMNS]
    is_dup = df.duplicated(subset=cols, keep="first")
    block = ["transect", "date", "passage"]
    frac = is_dup.groupby([df[b] for b in block]).transform("mean")
    whole = is_dup & (frac >= threshold)
    n_visits = df.loc[whole, block].drop_duplicates().shape[0]
    log.add("duplicates", f"{n_visits} visits entered twice (>= {threshold:.0%} of rows duplicated)",
            "second copy removed", whole.sum())
    df["isolated_exact_dup"] = is_dup & ~whole
    log.add("duplicates", "isolated exact duplicate rows inside otherwise normal visits",
            "kept, flagged for sensitivity analysis", df["isolated_exact_dup"].sum())
    return df.loc[~whole].copy()


def reconcile_visits(df, log, min_main_points=5):
    """Repair date / passage inconsistencies so that 1 visit = 1 date.

    By protocol a transect is surveyed twice per breeding season (passage 1 and
    2), each time covering points 1-10 in one morning. The raw data contain
    four kinds of inconsistencies, handled in this order:

    1. Drag-filled dates: a (transect, year, passage) block with more distinct
       dates than points (impossible). Dates were incremented row by row, so
       the first date is kept.
    2. Swapped passage labels: two complete visits in one year labelled with
       the same passage while the other passage is absent -> relabelled
       chronologically.
    3. Mislabelled rows: a small 'fragment' (< ``min_main_points`` points)
       whose date equals the date of a main visit -> takes that visit's passage.
    4. Fragments on another date:
       * points complementary to the main visit of the same passage -> the
         visit was split over two days (or the date mistyped): merged.
       * points overlapping that main visit -> ambiguous re-survey of points
         already counted: removed to keep effort at 1 count per point.
       * no main visit for that passage -> kept as an incomplete visit.
    """
    df = df.copy()
    df["date_orig"] = df["date"]
    df["passage_orig"] = df["passage"]

    # 1. Drag-filled dates
    nd = df.groupby(["transect", "year", "passage"])["date"].transform("nunique")
    drag = nd > 10
    first = df.groupby(["transect", "year", "passage"])["date"].transform("min")
    df.loc[drag, "date"] = first[drag]
    log.add("visit dates", "dates incremented row by row within one visit (Excel drag-fill)",
            "set to first date of the visit", drag.sum())

    blocks = (df.groupby(["transect", "year", "passage", "date"])["point"]
                .agg(lambda s: frozenset(s)).rename("pts").reset_index())
    blocks["main"] = blocks["pts"].map(len) >= min_main_points

    relabel, merged_rows, dropped_idx = {}, 0, []
    for (tr, yr), b in blocks.groupby(["transect", "year"]):
        mains = b[b["main"]].sort_values("date")
        # 2. Same passage label on two complete visits, other passage missing
        if len(mains) == 2 and mains["passage"].nunique() == 1:
            for new_p, (_, r) in enumerate(mains.iterrows(), start=1):
                relabel[(tr, yr, r["passage"], r["date"])] = (new_p, r["date"])
            mains = mains.assign(passage=[1, 2])
        main_by_date = dict(zip(mains["date"], mains["passage"]))
        main_by_pass = {p: (d, pts) for p, d, pts in zip(mains["passage"], mains["date"], mains["pts"])}

        for _, f in b[~b["main"]].iterrows():
            k = (tr, yr, f["passage"], f["date"])
            if f["date"] in main_by_date:                                   # 3.
                relabel[k] = (main_by_date[f["date"]], f["date"])
            elif f["passage"] in main_by_pass:                              # 4.
                d_main, pts_main = main_by_pass[f["passage"]]
                if f["pts"].isdisjoint(pts_main):
                    relabel[k] = (f["passage"], d_main)
                else:
                    relabel[k] = ("drop", None)

    key = list(zip(df["transect"], df["year"], df["passage"], df["date"]))
    new = [relabel.get(k) for k in key]
    n_swapped = n_moved = n_merged = n_drop = 0
    for i, (k, v) in enumerate(zip(key, new)):
        if v is None:
            continue
        if v[0] == "drop":
            dropped_idx.append(df.index[i]); n_drop += 1
            continue
        p_new, d_new = v
        if d_new != k[3]:
            n_merged += 1
        elif p_new != k[2]:
            n_moved += 1
        df.iat[i, df.columns.get_loc("passage")] = p_new
        df.iat[i, df.columns.get_loc("date")] = d_new

    changed_pass = (df["passage"] != df["passage_orig"]) & (df["date"] == df["date_orig"])
    log.add("visit passage", "passage label inconsistent with visit date (swapped or mislabelled)",
            "relabelled from the visit date", changed_pass.sum())
    log.add("visit dates", "fragment of a visit recorded on another day, complementary points",
            "merged into the visit of the same passage", n_merged)
    log.add("visit dates", "fragment on another day re-surveying points already counted",
            "removed (keeps 1 count per point per visit)", n_drop)
    df = df.drop(index=dropped_idx)

    # 5. A visit still spread over two dates with disjoint points (two halves
    #    of >= 5 points each, e.g. points 1-5 / 6-10): one visit over two days.
    k3 = ["transect", "year", "passage"]
    nd = df.groupby(k3)["date"].transform("nunique")
    first = df.groupby(k3)["date"].transform("min")
    split = (nd > 1) & (df["date"] != first)
    df.loc[split, "date"] = first[split]
    log.add("visit dates", "visit split in two halves over two days",
            "dated to its first day", split.sum())
    df["doy"] = df["date"].dt.dayofyear
    df["visit_id"] = (df["transect"] + " | " + df["year"].astype(str)
                      + " | P" + df["passage"].astype(str))
    return df


def recompute_counts(df, log):
    """Individual counts from the 8 distance x detection cells.

    ``count``        = all individuals heard or seen (excluding fly-overs);
    ``count_aud``    = heard only cells; ``count_vis`` = seen only cells;
    ``count_lt50``   = individuals within 50 m (less sensitive to observer
                       hearing ability -> used in robustness checks);
    ``flight``       = fly-overs (excluded from abundance: not using the site).
    The workbook total columns disagree with the cells in a few hundred rows
    (broken formulas, fractional values), hence the recomputation.
    """
    det = df[C.DETAIL_COLS].apply(lambda s: pd.to_numeric(_squash(s.astype(str)).str.replace("#", ""),
                                                          errors="coerce"))
    raw_txt = df[C.DETAIL_COLS].astype("string").apply(lambda s: s.str.strip())
    nonnum = (raw_txt.notna() & raw_txt.ne("")
              & raw_txt.apply(lambda s: pd.to_numeric(s, errors="coerce")).isna()).sum().sum()
    log.add("counts", "non-numeric entries in count cells (e.g. '1#', ' ')",
            "cleaned ('#' stripped) or treated as 0", nonnum)
    df[C.DETAIL_COLS] = det.fillna(0)
    df["flight"] = pd.to_numeric(df["flight"], errors="coerce").fillna(0)
    df["count_aud"] = df[C.AUD_COLS].sum(axis=1)
    df["count_vis"] = df[C.VIS_COLS].sum(axis=1)
    df["count"] = df["count_aud"] + df["count_vis"]
    df["count_lt50"] = df[["aud_lt25", "vis_lt25", "aud_25_50", "vis_25_50"]].sum(axis=1)

    xls_tot = pd.to_numeric(df["xls_total_av_flight"], errors="coerce")
    no_detail = (det.notna().sum(axis=1) == 0) & (df["flight"] == 0)
    # 4 rows have only totals filled in: fall back to the workbook total,
    # attributing it to the 'count' (non-flight) column.
    df.loc[no_detail, "count"] = xls_tot[no_detail].fillna(0)
    mism = (~no_detail) & (df["count"] + df["flight"] != xls_tot)
    log.add("counts", "workbook total columns inconsistent with distance cells",
            "counts recomputed from the cells", mism.sum())
    log.add("counts", "rows with only the total column filled in",
            "workbook total used", no_detail.sum())
    frac = (df["count"] % 1 != 0).sum()
    log.add("counts", "non-integer counts", "kept as recorded", frac)
    return df


def harmonise_species(df, species, log):
    """Attach scientific name and status; flag records not identified to species.

    * Names are matched case/whitespace-insensitively to the ESPECES sheet.
    * One spelling error in ESPECES ('Pigeon à coup rouge') is corrected.
    * Two species absent from ESPECES are added from external knowledge
      (Scaly-naped Pigeon, Semipalmated Sandpiper); one missing status is
      filled (Estrilda melpoda is an introduced waxbill). These are flagged.
    * Records identified only to genus/family (e.g. 'Trochilidae',
      'Moqueur sp.') are kept for total abundance but excluded from species
      richness and species-level analyses (``is_species = False``).
    """
    sp = species.drop_duplicates("name_fr").copy()
    sp.loc[sp["name_fr"] == "Pigeon à coup rouge", "name_fr"] = "Pigeon à cou rouge"
    sp.loc[sp["name_fr"] == "Astrild à joues orange", "status_fr"] = "Exogène introduit par l'homme"
    extra = pd.DataFrame({
        "name_fr": ["Bécasseau semipalmé"],
        "name_sci": ["Calidris pusilla"],
        "status_fr": ["Migrateur"]})
    sp = pd.concat([sp, extra], ignore_index=True)
    # Scientific names: collapse spaces; where two names are given ('A ou B'),
    # keep the currently accepted one.
    sci = sp["name_sci"].str.replace(r"\s+", " ", regex=True).str.strip()
    sci = sci.replace({"Melanospiza bicolor ou Tiaris bicolor": "Melanospiza bicolor",
                       "Dendroica ou Setophaga petechia ruficapilla": "Setophaga petechia ruficapilla"})
    still = sci.str.contains(r"\bou\b", na=False)
    sci[still] = sci[still].str.split(r"\s+ou\s+").str[-1]
    sp["name_sci"] = sci
    sp["status"] = sp["status_fr"].map(C.STATUS_EN)
    sp["status_group"] = sp["status"].map(C.STATUS_GROUP)

    k = _key(df["species_raw"])
    sp_key = dict(zip(_key(sp["name_fr"]), sp["name_fr"]))
    df["species"] = k.map(sp_key).astype("string")
    df["is_species"] = df["species"].notna()

    # Unidentified taxa keep a readable label
    lumped = {"trochilidae": "Trochilidae (unid.)", "fauconidae": "Falconidae (unid.)",
              "falconidé": "Falconidae (unid.)"}
    lbl = k.map(lumped).fillna(_squash(df["species_raw"]) + " (unid.)")
    df.loc[~df["is_species"], "species"] = lbl[~df["is_species"]]

    nas = k.isna() | k.isin(["ras"])
    df["is_record"] = ~nas
    df.loc[nas, "species"] = pd.NA
    log.add("species", "records not identified to species (genus/family level)",
            "kept for abundance, excluded from richness/species analyses",
            (~df["is_species"] & ~nas).sum())
    log.add("species", "'RAS' (nothing to report) or blank species rows",
            "kept as empty point visits for effort, not as records", nas.sum())
    df = df.merge(sp[["name_fr", "name_sci", "status", "status_group"]],
                  left_on="species", right_on="name_fr", how="left").drop(columns="name_fr")
    return df, sp


def flag_observer_context(df, log):
    """Parse the free-text remark column.

    'Compagnonnage par X' / 'avec X' / 'en formation' -> a trainee observer
    was accompanied (mentored) by an experienced one; 'Fait par X' -> the count
    was actually performed by X. Both matter for observer-effect analyses.
    """
    n = _key(df["note"]).fillna("")
    df["mentored"] = n.str.contains(r"compagn|copagn|accompagn|acompagn|avec |formation")
    df["done_by_other"] = n.str.contains(r"fait par")
    log.add("remarks", "remarks indicating mentoring / training", "flag 'mentored'",
            df["mentored"].sum())
    log.add("remarks", "remarks indicating the count was done by someone else",
            "flag 'done_by_other'", df["done_by_other"].sum())
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def clean_observations(obs_raw, species_raw, sites_raw):
    """Run all cleaning steps; return (records, species_table, quality_log)."""
    log = QualityLog()
    df = obs_raw.copy()
    log.add("raw", "rows in observation sheet", "-", len(df))
    df = harmonise_transects(df, sites_raw, log)
    df = harmonise_observers(df, log)
    df = parse_dates_and_codes(df, log)
    df = drop_double_entered_visits(df, log)
    df = reconcile_visits(df, log)
    df = recompute_counts(df, log)
    df = flag_observer_context(df, log)
    df, sp = harmonise_species(df, species_raw, log)
    log.add("final", "clean rows (incl. empty point visits)", "-", len(df))
    log.add("final", "clean bird records", "-", df["is_record"].sum())
    return df.reset_index(drop=True), sp, log.to_frame()


def prepare_sites(sites_raw):
    """Point catalogue with point number 1-10 and English habitat labels.

    Points are listed in order along each transect; the point code (e.g.
    'S1P2') is parsed where possible and checked against that order (one code
    is mistyped 'SIPI' for 'S1P1').
    """
    st = sites_raw.copy()
    st["point"] = st.groupby("transect").cumcount() + 1
    parsed = pd.to_numeric(st["point_code"].astype("string").str.extract(r"P(\d+)$")[0],
                           errors="coerce")
    agree = (parsed == st["point"]) | parsed.isna()
    if not agree.all():
        raise ValueError("Point order does not match point codes")
    st["habitat"] = st["habitat_fr"].map(C.HABITAT_EN)
    return st[["transect", "point", "x_utm", "y_utm", "habitat", "point_code"]]
