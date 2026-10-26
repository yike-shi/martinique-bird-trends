"""Project-wide configuration: paths, constants and column dictionaries.

All paths are resolved relative to the project folder (the parent of ``src/``),
so the code runs unchanged from a clean checkout of
``projects/birds-biodiversity/`` or from any copy of that folder.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = PROJECT_ROOT / "data" / "raw" / "Observations 2012-2025.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"            # everything the notebooks export
FIG_DIR = OUTPUT_DIR / "figures"
RES_DIR = OUTPUT_DIR / "results"

for _d in (PROCESSED_DIR, FIG_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 20261017

# --------------------------------------------------------------------------- #
# Study window required by the assignment brief
# --------------------------------------------------------------------------- #
TREND_YEARS = list(range(2014, 2026))

# --------------------------------------------------------------------------- #
# Sheet names (pd.read_excel only reads the first sheet unless told otherwise)
# --------------------------------------------------------------------------- #
SHEET_SPECIES = "ESPECES"
SHEET_SITES = "GPS-MILIEU"
SHEET_OBS = "NOM FRANÇAIS"

# The observation sheet has a 3-row header (variable / distance band /
# detection mode). We skip it and assign explicit English names, in sheet
# order. The French original is kept in the dictionary for the report.
OBS_COLUMNS = {
    "observer_raw": "Nom observateur",
    "dept": "code département",
    "transect_raw": "Nom transect",
    "date_raw": "date",
    "passage_raw": "1er, 2e ou 3e passage",
    "clouds": "nuages",
    "rain": "pluie",
    "wind": "vent",
    "visibility": "visibilité",
    "point": "N° point",
    "start_time": "heure début",
    "species_raw": "ESPECE",
    "aud_lt25": "distances de contact / < 25m / Auditif",
    "vis_lt25": "distances de contact / < 25m / Visuel",
    "aud_25_50": "distances de contact / <50m / Auditif",
    "vis_25_50": "distances de contact / <50m / Visuel",
    "aud_50_100": "distances de contact / <100m / Auditif",
    "vis_50_100": "distances de contact / <100m / Visuel",
    "aud_gt100": "distances de contact / > 100m / Auditif",
    "vis_gt100": "distances de contact / > 100m / Visuel",
    "flight": "vol (fly-overs)",
    "xls_total_aud": "totaux / Auditif",
    "xls_total_vis": 'totaux / Visuel (sans "Vol")',
    "xls_total_av": 'totaux / A+V (sans "Vol")',
    "xls_total_av_flight": 'totaux / A+V (avec "Vol")',
    "note": "(unnamed free-text remarks)",
}

AUD_COLS = ["aud_lt25", "aud_25_50", "aud_50_100", "aud_gt100"]
VIS_COLS = ["vis_lt25", "vis_25_50", "vis_50_100", "vis_gt100"]
DETAIL_COLS = AUD_COLS + VIS_COLS
DIST_BANDS = {"<25 m": ["aud_lt25", "vis_lt25"],
              "25-50 m": ["aud_25_50", "vis_25_50"],
              "50-100 m": ["aud_50_100", "vis_50_100"],
              ">100 m": ["aud_gt100", "vis_gt100"]}

# Weather codes are ordinal 1-3 (higher = worse). No codebook ships with the
# data; the labels below follow the French STOC-EPS protocol on which this
# survey design is modelled, and are an ASSUMPTION stated in the report.
WEATHER_VALID = {"clouds": {1, 2, 3}, "rain": {1, 2, 3},
                 "wind": {1, 2, 3}, "visibility": {1, 2, 3}}
WEATHER_LABELS = {
    "clouds": {1: "0-33 %", 2: "33-66 %", 3: "66-100 %"},
    "rain": {1: "none", 2: "drizzle", 3: "showers"},
    "wind": {1: "none", 2: "light", 3: "moderate/strong"},
    "visibility": {1: "good", 2: "moderate", 3: "poor"},
}

# Species status (ESPECES sheet) -> English labels and coarse groups.
STATUS_EN = {
    "Endémique de la Martinique": "Endemic (Martinique)",
    "Endémique des Petites Antilles": "Endemic (Lesser Antilles)",
    "Autochtone": "Native",
    "Exogène colonisateur naturel": "Exotic (natural coloniser)",
    "Exogène introduit par l'homme": "Exotic (human-introduced)",
    "Migrateur": "Migrant",
    "Migrateur rare": "Migrant",
    "Marin": "Seabird",
}
STATUS_GROUP = {
    "Endemic (Martinique)": "Endemic",
    "Endemic (Lesser Antilles)": "Endemic",
    "Native": "Native",
    "Exotic (natural coloniser)": "Exotic",
    "Exotic (human-introduced)": "Exotic",
    "Migrant": "Migrant",
    "Seabird": "Seabird",
}

# Habitat (GPS-MILIEU sheet) -> English
HABITAT_EN = {
    "Forêt sèche": "Dry forest",
    "Forêt humide": "Wet forest",
    "Agricole": "Agricultural",
    "Périurbain": "Peri-urban",
    "Urbain": "Urban",
    "Mangrove": "Mangrove",
    "Plage": "Beach",
}
