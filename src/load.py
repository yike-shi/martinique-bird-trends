"""Read the three sheets of the raw workbook into tidy, English-named tables.

Nothing is cleaned here: this module only fixes the *layout* problems of the
workbook (multi-row headers, header-less sheets, blank leading columns) so that
every downstream step can be audited against the raw file. The workbook is
opened read-only and never written to.
"""
import pandas as pd

from . import config as C


def load_observations(path=C.RAW_XLSX) -> pd.DataFrame:
    """Observation log (sheet 'NOM FRANÇAIS'), one row per species x point visit.

    The sheet has a 3-row header (variable / distance band / detection mode),
    so we skip those rows and assign explicit names from ``config.OBS_COLUMNS``.
    ``excel_row`` stores the 1-based row number in Excel for traceability.
    """
    raw = pd.read_excel(path, sheet_name=C.SHEET_OBS, header=None, skiprows=3)
    if raw.shape[1] != len(C.OBS_COLUMNS):
        raise ValueError(f"Expected {len(C.OBS_COLUMNS)} columns, got {raw.shape[1]}")
    raw.columns = list(C.OBS_COLUMNS)
    raw.insert(0, "excel_row", raw.index + 4)  # 3 header rows + 1-based index
    return raw


def load_species(path=C.RAW_XLSX) -> pd.DataFrame:
    """Species list (sheet 'ESPECES'): French name, scientific name, status.

    The sheet has no header row and two empty leading columns.
    """
    raw = pd.read_excel(path, sheet_name=C.SHEET_SPECIES, header=None)
    sp = raw.iloc[:, 2:5].copy()
    sp.columns = ["name_fr", "name_sci", "status_fr"]
    sp = sp.dropna(subset=["name_fr"]).reset_index(drop=True)
    for c in sp.columns:
        sp[c] = sp[c].astype("string").str.strip()
    return sp


def load_sites(path=C.RAW_XLSX) -> pd.DataFrame:
    """Point catalogue (sheet 'GPS-MILIEU'): transect, UTM coordinates, habitat.

    Row 1 holds the CRS label, row 2 the partial header; data start on row 3.
    Coordinates are WGS84 / UTM zone 20N, in metres.
    """
    raw = pd.read_excel(path, sheet_name=C.SHEET_SITES, header=None, skiprows=2)
    st = raw.iloc[:, 2:8].copy()
    st.columns = ["transect", "x_utm", "y_utm", "habitat_fr", "site_code", "point_code"]
    st = st.dropna(subset=["transect"]).reset_index(drop=True)
    st["transect"] = st["transect"].astype("string").str.strip()
    st["x_utm"] = pd.to_numeric(st["x_utm"])
    st["y_utm"] = pd.to_numeric(st["y_utm"])
    return st
