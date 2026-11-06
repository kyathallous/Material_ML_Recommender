# HANDOFF (v3.2 rebuild) - read first
Project: Data-Driven Machine Learning Recommender for Mechanical Material Selection. Owner: 2nd-year mechanical engineering student,
new to Python/ML, Windows, Python 3.13, VS Code. Explain simply, one step at a time.

Requirements: (1) classification + nearest-neighbour alloy recommender on yield, tensile, E, density (summary line verbatim on every page);
(2) EXTRA: heat-treatment recommender (material + wanted properties -> treatment); (3) experimental, published, cited data only - nothing
synthetic, no filled-in/handbook-from-memory values; (4) data analysis + ML connected, pretty UI; user only adds data.

Layout: START_APP.bat / STOP_ALL_PYTHON.bat / run_app.py (free port from 5050), app.py (Flask), main.py (CLI), src/{config,data_loader,
data_manager,selection,heat_treatment,charts}.py, templates/, static/style.css, data/raw/sources.csv (file,citation,kind,family),
data/templates/reference_table_template.csv.

Rows in compiled files with a data_tier column are kept only if tier contains "measured" (real SteelBench file has emk_spec_verified = standards-sheet values, excluded).
Provenance labels: Experimental (published) = file listed in sources.csv; Published reference (handbook table) = kind=reference;
Unverified = unlisted. steelbench_full is rejected (imputed). config.PLAUSIBLE blanks placeholder values (e.g. 30 C). NIMS columns
THT/THt/TT/Tt/Fatigue are matched case-sensitively (EXACT_ALIASES). Selector offers only properties with >=10 values.

Tested (Linux, mock files shaped like the datasets, deleted afterwards): empty state, mock load, POSTs, CLI, path with spaces, cp1252, port busy.
NOT tested: real files (column names unseen; Zenodo file is named steelbench_core_open.csv), Windows .bat files, Python 3.13.
Next: user adds files -> screenshot of Data & files -> fix aliases -> read real CV R2 -> add E/density via reference table.
Rules: never add demo/synthetic/imputed values; never claim unread files were read; keep banners and labels.

## Update (real-data session, 2026-10-28)
- Real files seen on the user's PC: nims_fatigue.csv (437), steel_strength.csv (312), steelbench_core.csv (607 measured of 1360; 753 `emk_spec_verified`
  spec-sheet rows excluded; measured rows have NO heat-treatment temperatures; `nims_measured` rows have an unexplained tensile-only value: read DATASET_CARD.md).
- SteelBench `hardness` unit unknown -> excluded via sources.csv `ignore_columns`.
- Added tempering_hardness.csv (Kaggle rgerschtzsauer; 1466 rows; columns 'Source','Steel type','Tempering time (s)','Tempering temperature (ºC)','C (%wt)'...,
  'Final hardness (HRC) - post tempering'); each row names its original paper (kept in `source`); seconds converted to minutes; hardness unit label = HRC.
  Original upstream source/licence of the Kaggle compilation not verified.
- Heat model targets with real data: fatigue (NIMS), hardness (tempering file). Yield/tensile for heat treatment still lack data.

## Update 4 (v3.3): material selection polish + multi-family support
- Selection page: family filter, "experimental rows only", margin-over-limits column, method note. Selector.recommend(req,k,family,experimental_only).
- More elements (Cu, Be, Zn, Mg, Fe, Zr, Sn); families Copper/Magnesium derived; names built per family; files with a manifest `family` and no composition get "<family> record n".
- Candidate extra experimental files (NOT yet seen by anyone; sandbox cannot download): Cu alloys (Sci Data 2023, figshare doi 10.6084/m9.figshare.c.6475600.v1, per-record DOI),
  Al alloys (Mendeley b6br4yk6r3, origin of values unstated), Ti alloys (Zenodo 5727737, composition at.%, has E). None has density; only Ti has E. Expect to extend COMPOSITION/aliases after seeing headers.

## Update 5 (v3.4): Kaggle handbook-style table (Data.csv, 1552 rows: Std, Material, Heat treatment, Su, Sy, A5, Bhn, E[MPa], G, mu, Ro[kg/m3], Desc, HV)
- User asked to use it regardless of provenance to supply Young's modulus and density. Saved as kaggle_materials.csv, listed in sources.csv with kind=reference
  (label "Reference data (cited database)"; never experimental) and ignore_columns=hardness (HV would mix with HRC). Bhn/G/mu/pH/Desc ignored.
- Per-file unit conversion (MPa->GPa, kg/m3->g/cm3); family guessed from grade name/description (family_from_text); name = grade + (condition).
- Selector now ranks only by the properties the user constrained and searches a pool of rows that have them (no imputation); classifier trains on rows with all four.
- Values in that table are standard/typical values, not measurements.

## Update 6 (v3.6)
- User says Data.csv is the Autodesk material database (IDs are 32-hex GUIDs; NOT verified by me; proprietary library, cite and do not redistribute). Treated as reference data: sources.csv kind=reference,
  label "Reference data (cited database)"; mode "published" tag "EXPERIMENTAL + REFERENCE DATA". "secondary/unverified" kind still exists for undeclared sources.
- Selection page: info note that modulus/density come from reference data; "Experimental rows only" + modulus/density limit gives an explanatory message.
