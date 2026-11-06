"""Settings shared by every module: folders, standard column names, labels."""
from pathlib import Path

APP_VERSION = "3.6"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"        # <- your experimental data files go here
MANIFEST = RAW_DIR / "sources.csv"     # one line per file: file name, citation
OUT_DIR = ROOT / "outputs"
MODEL_DIR = OUT_DIR / "models"
for _folder in (RAW_DIR, OUT_DIR, MODEL_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# Standard column names. Every data file is renamed to these automatically.
COMPOSITION_COLS = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Co", "W", "Al", "Ti", "Nb", "N",
                    "Cu", "Be", "Zn", "Mg", "Fe", "Zr", "Sn"]   # wt %
SELECTION_PROPS = ["yield_strength", "tensile_strength", "youngs_modulus", "density"]
HT_TARGETS = ["yield_strength", "tensile_strength", "hardness", "elongation", "fatigue_strength"]
PROPERTY_COLS = ["yield_strength", "tensile_strength", "youngs_modulus", "density",
                 "hardness", "elongation", "fatigue_strength"]
TREATMENT_NUM = ["austenitize_temp_C", "austenitize_time_min", "temper_temp_C", "temper_time_min"]

PROP_LABEL = {"yield_strength": "Yield strength", "tensile_strength": "Tensile strength", "hardness": "Hardness",
              "elongation": "Elongation", "fatigue_strength": "Fatigue strength",
              "youngs_modulus": "Young's modulus", "density": "Density"}
PROP_UNIT = {"yield_strength": "MPa", "tensile_strength": "MPa", "hardness": "HRC", "elongation": "%",
             "fatigue_strength": "MPa", "youngs_modulus": "GPa", "density": "g/cm³"}

# Every row carries one of these three labels. Nothing is ever assumed to be experimental.
EXPERIMENTAL = "Experimental (published)"      # file is listed in data/raw/sources.csv with a citation
REFERENCE = "Reference data (cited database)"   # typical / specified values from a named database or handbook; never called "experimental"
UNVERIFIED = "Secondary data (original source not stated)"   # used, but never called experimental

# Values outside these ranges are treated as "not reported" (e.g. 30 C used as a placeholder for "step not applied").
PLAUSIBLE = {"austenitize_temp_C": (300, 1400), "temper_temp_C": (50, 900), "austenitize_time_min": (0.5, 100000),
             "temper_time_min": (0.5, 100000), "yield_strength": (20, 4000), "tensile_strength": (30, 4500),
             "elongation": (0, 100), "fatigue_strength": (20, 2500), "hardness": (10, 1000),
             "youngs_modulus": (1, 700), "density": (0.5, 25)}

RANDOM_STATE = 42
