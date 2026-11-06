PUT YOUR DATA FILES IN THIS FOLDER (csv or xlsx).

Expected names (see the app, page "Data & files", for download links):
  steel_strength.csv   nims_fatigue.csv   steelbench_core.csv

sources.csv decides the label of each file:
  - listed with a citation           -> "Experimental (published)"
  - listed with kind = secondary     -> "Secondary data (original source not stated)"
  - listed with kind = reference     -> "Reference data (cited database)"
  - not listed                       -> "Secondary data (original source not stated)"
Columns of sources.csv: file, citation, kind (optional), family (optional: used if the file has no material family).
Never put AI-generated, invented or remembered values in these files.
