"""Reading data files: recognise column names, convert to numbers, fix units."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import COMPOSITION_COLS, PLAUSIBLE, PROPERTY_COLS, TREATMENT_NUM, UNVERIFIED

# standard name -> other spellings found in real datasets (case and symbols are ignored)
COLUMN_ALIASES = {
    "material": ["material", "grade", "alloy", "steel", "name", "designation", "material grade"],
    "material_family": ["family", "class", "material class", "category"],
    "yield_strength": ["yield strength", "ys", "yield", "rp02", "yield stress", "yield strength mpa",
                       "0.2 proof stress", "yield strength 0.2"],
    "tensile_strength": ["tensile strength", "uts", "ultimate tensile strength", "ts", "rm", "tensile strength mpa",
                         "ultimate strength", "ultimate tensile strength mpa"],
    "youngs_modulus": ["youngs modulus", "young modulus", "elastic modulus", "modulus of elasticity", "e", "e gpa"],
    "density": ["density", "density g cm3", "rho"],
    "hardness": ["hardness", "hb", "brinell hardness", "brinell", "hardness hb", "hardness hv", "hv"],
    "elongation": ["elongation", "elongation %", "el", "percent elongation", "ductility", "total elongation"],
    "fatigue_strength": ["fatigue strength", "fatigue strength mpa", "fatigue limit", "fatigue"],
    "austenitize_temp_C": ["austenitize temp", "austenitizing temperature", "quenching temperature",
                           "hardening temperature", "through hardening temperature", "through hardening temperature c",
                           "solution treatment temperature", "quench temp"],
    "austenitize_time_min": ["austenitize time", "austenitizing time", "holding time", "through hardening time",
                             "through hardening time min"],
    "temper_temp_C": ["temper temp", "tempering temperature", "tempering temperature c", "temper temperature",
                      "aging temperature", "ageing temperature"],
    "temper_time_min": ["temper time", "tempering time", "tempering time min", "aging time", "ageing time"],
    "quench_method": ["quench", "quenchant", "quench medium", "cooling method", "quenching media"],
    "data_type": ["data type", "provenance"],
    "source": ["source", "reference", "paper", "doi", "citation"],
}
# NIMS / SteelBench abbreviations. Matched with exact capitalisation, because THT and THt (temperature / time) differ only by case.
EXACT_ALIASES = {"THT": "austenitize_temp_C", "THt": "austenitize_time_min", "TT": "temper_temp_C",
                 "Tt": "temper_time_min", "Fatigue": "fatigue_strength"}
COLUMN_ALIASES["austenitize_temp_C"] += ["austenitize t", "austenitise t", "austenitize temperature"]
COLUMN_ALIASES["temper_temp_C"] += ["temper t"]
COLUMN_ALIASES["material"] += ["steel type", "steel grade", "aisi"]
COLUMN_ALIASES["hardness"] += ["final hardness hrc post tempering", "final hardness hrc", "hardness hrc"]
COLUMN_ALIASES["yield_strength"] += ["sy"]
COLUMN_ALIASES["tensile_strength"] += ["su"]
COLUMN_ALIASES["elongation"] += ["a5"]
COLUMN_ALIASES["density"] += ["ro"]
COLUMN_ALIASES["treatment_type"] = ["heat treatment", "treatment", "condition temper"]
COLUMN_ALIASES["temper_time_s"] = ["tempering time s", "temper time s", "tempering time sec", "tempering time seconds"]
for _el in COMPOSITION_COLS:
    _low = _el.lower()
    COLUMN_ALIASES[_el] = [_low, f"{_low} wt", f"{_low} wt pct", f"{_low} pct", f"{_low} %", f"{_low} wt%", f"{_low} %wt",
                           f"{_low} at pct", f"{_low} weight percent"]


def _simplify(name) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9% ]", " ", str(name).lower())).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename recognised columns to the standard names; unrecognised columns are left as they are."""
    lookup = {}
    for standard, spellings in COLUMN_ALIASES.items():
        for spelling in [standard, *spellings]:
            lookup[_simplify(spelling)] = standard
    rename = {col: EXACT_ALIASES[col] for col in df.columns if col in EXACT_ALIASES}
    for col in df.columns:
        if col in rename:
            continue
        standard = lookup.get(_simplify(col))
        if standard and standard not in rename.values():
            rename[col] = standard
    return df.rename(columns=rename)


def read_table(path) -> pd.DataFrame:
    """Read a CSV or Excel file (tries several text encodings so unusual files still open)."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot read {path.name}")


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """Make every standard column exist and be numeric (or text where it should be)."""
    for col in COMPOSITION_COLS + PROPERTY_COLS + TREATMENT_NUM:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    df["quench_method"] = (df["quench_method"] if "quench_method" in df else pd.Series("none", index=df.index)) \
        .fillna("none").astype(str).str.strip().str.lower()
    for col, default in (("material_family", "Unknown"), ("source", "unknown"), ("treatment_type", ""),
                         ("data_type", UNVERIFIED)):
        df[col] = df[col].fillna(default) if col in df else default
    df["material"] = df["material"].astype(str).str.strip() if "material" in df else pd.Series(dtype=str)
    if df["youngs_modulus"].median() > 5000:          # looks like MPa -> convert to GPa
        df["youngs_modulus"] = df["youngs_modulus"] / 1000
    if df["density"].median() > 100:                  # looks like kg/m3 -> convert to g/cm3
        df["density"] = df["density"] / 1000
    for col, (low, high) in PLAUSIBLE.items():       # placeholders such as 30 C or 0 min mean "step not applied"
        df[col] = df[col].where(df[col].between(low, high))
    return df.reset_index(drop=True)


def quality_report(df: pd.DataFrame) -> None:
    """Printed by:  python main.py check"""
    print(f"Rows: {len(df)}   Materials: {df['material'].nunique()}   Sources: {df['source'].nunique()}")
    filled = df[COMPOSITION_COLS + PROPERTY_COLS + TREATMENT_NUM].notna().mean().mul(100).round(0)
    print("\nColumns with data (% of rows):")
    print(filled[filled > 0].astype(int).to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"Rows where yield strength > tensile strength (suspicious): {(df['yield_strength'] > df['tensile_strength']).sum()}")
    print("\nData labels:")
    print(df["data_type"].value_counts().to_string())
