"""
Builds the working dataset from the files in data/raw/.

Rules that are never relaxed (no value is ever filled in or estimated)
  * Only real files you put in data/raw/ are used. The program contains no built-in or generated data.
  * A file is labelled "Experimental (published)" only if it is listed in data/raw/sources.csv with a citation
    (or carries its own data_type column). Any other file is still used, but labelled "Secondary data".
  * Names and families derived from composition are announced in the "notes" under each file on the Data & files page.
"""
import numpy as np
import pandas as pd

from config import (COMPOSITION_COLS, EXPERIMENTAL, MANIFEST, PROPERTY_COLS, RAW_DIR, REFERENCE, TREATMENT_NUM,
                    UNVERIFIED)
from data_loader import clean_values, normalize_columns, read_table

STANDARD_COLUMNS = (["material", "material_family"] + COMPOSITION_COLS + PROPERTY_COLS + TREATMENT_NUM +
                    ["quench_method", "treatment_type", "data_type", "source"])
DATA_SUFFIXES = (".csv", ".xlsx", ".xls")


def raw_files():
    return sorted(p for p in RAW_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in DATA_SUFFIXES and p.name.lower() != "sources.csv"
                  and not p.name.startswith(("~", ".")))


def read_manifest() -> dict:
    """data/raw/sources.csv  ->  {file name in lower case: {citation, kind, family}}.
    Columns: file, citation (or source), and optionally kind (reference) and family."""
    if not MANIFEST.exists():
        return {}
    try:
        table = read_table(MANIFEST)
    except Exception:
        return {}
    table.columns = [str(c).strip().lower() for c in table.columns]
    table = table.rename(columns={"source": "citation"})
    if not {"file", "citation"} <= set(table.columns):
        return {}
    out = {}
    for _, row in table.iterrows():
        text = lambda c: "" if c not in table or str(row[c]).lower() == "nan" else str(row[c]).strip()   # noqa: E731
        if text("file") and text("citation"):
            out[text("file").lower()] = dict(citation=text("citation"), kind=text("kind").lower(), family=text("family"),
                                             ignore=[c.strip() for c in text("ignore_columns").split(";") if c.strip()])
    return out


# ---------------------------------------------------------------------------------------------
# material names and families built from composition (for files that have no grade names)
# ---------------------------------------------------------------------------------------------
def family_of(row) -> str:
    g = lambda e: row.get(e) if pd.notna(row.get(e)) else 0.0      # noqa: E731
    if g("Ti") >= 50:
        return "Titanium alloy"
    if g("Al") >= 50:
        return "Aluminium alloy"
    if g("Cu") >= 50:
        return "Copper alloy"
    if g("Mg") >= 50:
        return "Magnesium alloy"
    if g("Cr") >= 10.5:
        return "Stainless steel"
    if g("Ni") >= 8:
        return "High-alloy steel (e.g. maraging)"
    if sum(g(e) for e in ("Cr", "Ni", "Mo", "V", "W", "Co", "Ti", "Nb")) >= 1.0:
        return "Low-alloy steel"
    return "Carbon steel"


def family_from_text(material, desc="") -> str:
    """Material family from the grade name / description, for handbook-style tables that give no composition."""
    text = f"{material} {desc if isinstance(desc, str) else ''}".lower()
    rules = [("cast iron", "Cast iron"), ("grey ", "Cast iron"), ("nodular", "Cast iron"), ("malleable", "Cast iron"),
             ("alumin", "Aluminium alloy"), ("magnesium", "Magnesium alloy"), ("titanium", "Titanium alloy"),
             ("bronze", "Copper alloy"), ("brass", "Copper alloy"), ("copper", "Copper alloy"), ("cartridge", "Copper alloy"),
             ("muntz", "Copper alloy"), ("nickel", "Nickel alloy"),
             ("stainless", "Stainless steel"), ("corrosion", "Stainless steel"), ("heat-resist", "Stainless steel"),
             ("heat resist", "Stainless steel"), ("carbon", "Carbon steel"), ("alloy", "Low-alloy steel"),
             ("steel", "Steel")]
    return next((family for word, family in rules if word in text), "Unknown")


def label_of(row) -> str:
    family = str(row.get("material_family", ""))
    prefix = "Steel" if "steel" in family.lower() or not family or family in ("nan", "Unknown") else family
    parts = [f"C{row['C']:.2f}"] if pd.notna(row.get("C")) else []
    parts += [f"{e}{row[e]:.1f}" for e in ("Mn", "Cr", "Ni", "Mo", "Al", "Ti", "Cu", "Be", "Zn", "Mg")
              if pd.notna(row.get(e)) and row[e] >= 0.05 and e != "C"]
    return f"{prefix} " + " ".join(parts) if parts else "Unnamed material"


# ---------------------------------------------------------------------------------------------
def prepare_file(path, entry):
    """Read one data file and convert it to the standard layout. Returns (table, notes, columns)."""
    if "steelbench_full" in path.name.lower():
        raise ValueError("steelbench_full contains imputed (filled-in) values and is not experimental. "
                         "Use steelbench_core instead")
    entry = entry or {}
    citation = entry.get("citation")
    notes = []
    raw = read_table(path)
    df = normalize_columns(raw)
    dropped = [c for c in entry.get("ignore", []) if c in df]
    if dropped:
        df = df.drop(columns=dropped)
        notes.append(f"column(s) {', '.join(dropped)} deliberately not used (sources.csv: ignore_columns)")
    if "temper_time_s" in df:                       # seconds -> minutes
        if "temper_time_min" not in df:
            df["temper_time_min"] = pd.to_numeric(df["temper_time_s"], errors="coerce") / 60
            notes.append("tempering time was given in seconds and converted to minutes")
        df = df.drop(columns="temper_time_s")
    # Some compiled files label each row's origin (data_tier). Only rows marked as measured are experiments;
    # rows from specification sheets (standard minimum values, composition mid-ranges) are not.
    if "data_tier" in df:
        measured = df["data_tier"].astype(str).str.contains("measured", case=False)
        if not measured.all():
            dropped = df.loc[~measured, "data_tier"].value_counts()
            notes.append(f"{int((~measured).sum())} rows were excluded because they are not measurements "
                         f"({', '.join(f'{k}: {v}' for k, v in dropped.items())})")
            df = df[measured].reset_index(drop=True)
        if df.empty:
            raise ValueError("no row in this file is marked as measured")
    recognised = [c for c in df.columns if c in STANDARD_COLUMNS or c in ("data_type", "source")]
    ignored = [str(c) for c in df.columns if c not in recognised]
    if entry.get("family") and "material_family" not in df:
        df["material_family"] = entry["family"]
        notes.append(f"material family set to '{entry['family']}' from sources.csv")
    for col in COMPOSITION_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, limit, unit in (("youngs_modulus", 5000, "MPa -> GPa"), ("density", 100, "kg/m3 -> g/cm3")):
        if col in df:                                  # units are fixed per file, before files are combined
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].median() > limit:
                df[col] = df[col] / 1000
                notes.append(f"{col} converted from {unit}")
    has_composition = "C" in df and df["C"].notna().any()

    has_composition = has_composition or any(c in df and df[c].notna().any() for c in COMPOSITION_COLS)
    if "material_family" not in df and has_composition and not entry.get("family"):
        df["material_family"] = df.apply(family_of, axis=1)
        notes.append("material family (carbon / low-alloy / stainless ...) was derived from the composition")
    if "material_family" not in df and "material" in df and not entry.get("family"):
        description = df["Desc"] if "Desc" in df else pd.Series("", index=df.index)
        df["material_family"] = [family_from_text(m, d) for m, d in zip(df["material"], description)]
        notes.append("the file gives no composition: material family was guessed from the grade name / description by keyword rules")
    if "material" not in df:
        if has_composition:
            df["material"] = df.apply(label_of, axis=1)
            notes.append("the file has no grade names, so names were built from the composition (e.g. 'Steel C0.25 Cr1.0')")
        elif "material_family" in df:
            df["material"] = [f"{f} record {i + 1}" for i, f in enumerate(df["material_family"])]
            notes.append("the file has no grade names and no composition: rows are numbered within their family")
        else:
            raise ValueError("the file has no material/grade column, no composition columns and no family given in sources.csv")

    if "treatment_type" in df and "material" in df:          # grade + condition, e.g. "Steel SAE 1015 (annealed)"
        condition = df["treatment_type"].fillna("").astype(str).str.strip()
        df["material"] = df["material"].astype(str) + condition.map(lambda c: f" ({c})" if c else "")
    if "source" in df and df["source"].notna().any():          # the file names the paper of each row: keep it
        row_source = df["source"].astype(str).str.strip()
        df["source"] = row_source + f" (via {citation})" if citation else row_source
        notes.append("each row keeps the publication named in the file's own source column")
    else:
        df["source"] = citation or path.name
    if citation:
        kind = entry.get("kind")
        df["data_type"] = REFERENCE if kind == "reference" else UNVERIFIED if kind in ("secondary", "unverified") else EXPERIMENTAL
        if kind in ("secondary", "unverified"):
            notes.append("marked as secondary data in sources.csv (kind = secondary): used, but never called experimental")
    else:
        df["data_type"] = df["data_type"] if "data_type" in df else UNVERIFIED
        notes.append("not listed in data/raw/sources.csv, so its rows are labelled 'Secondary data'. "
                     "Add a line with the citation to mark it as experimental")
    for col in STANDARD_COLUMNS:
        if col not in df:
            df[col] = np.nan
    return df[STANDARD_COLUMNS], notes, dict(recognised=recognised, ignored=ignored)


def build_dataset():
    """Returns (table, info). info['mode'] is 'empty', 'experimental' or 'mixed'."""
    manifest = read_manifest()
    info = dict(mode="empty", files=[], warnings=[], provenance={}, rows=0)
    tables = []
    for path in raw_files():
        entry = dict(name=path.name, rows=0, dtype="", status="ok", notes=[], recognised=[], ignored=[])
        try:
            table, entry["notes"], columns = prepare_file(path, manifest.get(path.name.lower()))
            entry.update(columns)
            entry["rows"] = len(table)
            entry["dtype"] = table["data_type"].mode().iat[0] if len(table) else ""
            tables.append(table)
        except Exception as error:                      # one bad file must never stop the others
            entry["status"], entry["notes"] = "skipped", [f"could not be used: {error}"]
            info["warnings"].append(f"{path.name}: {error}")
        info["files"].append(entry)

    if not tables:
        return clean_values(pd.DataFrame(columns=STANDARD_COLUMNS)), info

    df = clean_values(pd.concat(tables, ignore_index=True))
    # a record that appears twice keeps the better-labelled copy (experimental before secondary)
    key = [c for c in ["material", "yield_strength", "tensile_strength", "hardness", "elongation", "fatigue_strength",
                       "austenitize_temp_C", "temper_temp_C", "C", "Mn", "Cr", "Ni", "Mo"] if c in df]
    before = len(df)
    df = (df.assign(_rank=(df["data_type"] != EXPERIMENTAL).astype(int)).sort_values("_rank", kind="stable")
            .drop_duplicates(subset=key).drop(columns="_rank").sort_index().reset_index(drop=True))
    if len(df) < before:
        info["warnings"].append(f"{before - len(df)} duplicate rows were removed (the same data in two files?)")
    info["provenance"] = {k: int(v) for k, v in df["data_type"].value_counts().items()}
    info["rows"] = len(df)
    labels = set(info["provenance"])
    info["mode"] = ("experimental" if labels == {EXPERIMENTAL} else
                    "published" if labels <= {EXPERIMENTAL, REFERENCE} else "mixed")
    return df, info


def banner(info) -> dict:
    if info["mode"] == "empty":
        return dict(level="warn", title="No data loaded yet",
                    text="Add your experimental data file to the data/raw folder, then press 'Reload data'. "
                         "The step-by-step guide is on this page.")
    if info["mode"] == "experimental":
        return dict(level="ok", title="Experimental data loaded",
                    text=f"All {info['rows']} rows come from files listed in data/raw/sources.csv with a citation.")
    if info["mode"] == "published":
        ref = info["provenance"].get(REFERENCE, 0)
        return dict(level="ok", title="Experimental and reference data loaded",
                    text=f"{info['rows'] - ref} experimental rows and {ref} reference-data rows (typical values from a cited "
                         "database, not experiments). Every row is cited in data/raw/sources.csv.")
    unverified = info["provenance"].get(UNVERIFIED, 0)
    return dict(level="warn", title="Some rows are secondary data",
                text=f"{unverified} of {info['rows']} rows come from files whose source is not declared. They are used, "
                     "but are labelled 'Secondary data' wherever they appear. Use the 'Experimental rows only' option to exclude them.")
