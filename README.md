# Data-Driven Machine Learning Recommender for Mechanical Material Selection

Implements classification and nearest-neighbor algorithms to recommend optimal engineering alloys based on design
constraints including yield strength, tensile strength, Young's modulus, and density.
**Extra module:** a heat-treatment recommender (give a material and wanted properties, get a heat treatment to try).

**This project contains NO data.** You add real, published, experimental files; every row is labelled with its source.
Nothing is AI-generated, estimated or filled in.

## Run it (Windows)
1. Double-click `STOP_ALL_PYTHON.bat` (closes any old copy of the app).
2. Double-click `START_APP.bat`. Wait up to a minute. Your browser opens by itself; if not, open the address printed
   after "READY".
3. First time you see **"No data loaded yet"** and **Version 3.2** in the footer. That is correct.

## Add the data
Open the **Data & files** page: it lists every download link. Save the files in `data\raw\` with these names, then press **Reload data**.

| Save as | Source |
|---|---|
| `steel_strength.csv` | https://figshare.com/articles/Steel_Strength_Data/7250453 (the DATASET csv, not .GZ) |
| `nims_fatigue.csv` | https://www.kaggle.com/datasets/chaozhuang/steel-fatigue-strength-prediction (cite Agrawal et al. 2014, doi:10.1186/2193-9772-3-8) |
| `steelbench_core.csv` | https://zenodo.org/records/18530558 (file `steelbench_core_open.csv`, renamed; never the "full" file) |

Citations are already in `data/raw/sources.csv`. A file is labelled **Experimental (published)** only if listed there.

## Young's modulus, density, hardness
None of the files above contain them, so material selection starts with yield/tensile (+ elongation). To add the rest, copy
values from a cited handbook or paper into `data/templates/reference_table_template.csv`, save it into `data\raw\`, and add
a line to `sources.csv` with `kind` = `reference` (label: "Reference data", never "experimental").

## Pages
Overview, Data analysis, Material selection (Random Forest vs k-NN family classifier + NearestNeighbors recommender),
Heat treatment recommender (Random Forest per property, inverse search over treatments, uncertainty, nearest real records), Data & files.
Press Ctrl+P on any page to save it as PDF. `python main.py charts` saves report figures to `outputs/`.

## Honest limits
- Predictions are statistical; check "R²" on the Overview page. The heat-treatment part is steel-only.
- The family classifier is weak when the data is mostly one family (families are derived from composition).
- Not tested here: the real downloaded files and Windows. If a column is "ignored" on the Data page, send a screenshot.
