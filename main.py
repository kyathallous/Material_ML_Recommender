"""
Command-line helper (the web app is the main way to use the project).

  python main.py check                         data-quality report
  python main.py charts                        save report-ready PNG charts into outputs/
  python main.py select --ys 450 --uts 550 --e 60 --density 3.0
  python main.py heat --c 0.4 --mn 0.8 --cr 1.0 --mo 0.2 --hardness 350
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from charts import REPORT_CHARTS  # noqa: E402
from config import OUT_DIR  # noqa: E402
from data_loader import quality_report  # noqa: E402
from data_manager import build_dataset  # noqa: E402
from heat_treatment import HeatTreatmentRecommender  # noqa: E402
from selection import MaterialSelector  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check", "charts", "select", "heat"])
    for name in ("ys", "uts", "e", "density", "elong", "hardness", "fatigue", "c", "mn", "si", "cr", "ni", "mo", "v"):
        p.add_argument("--" + name, type=float)
    p.add_argument("--material")
    p.add_argument("-k", type=int, default=5)
    a = p.parse_args()

    df, info = build_dataset()
    if info["mode"] == "empty":
        sys.exit("No data yet: put your experimental file in data/raw/ (see README.md, Step 3).")
    print(f"{info['rows']} rows | {info['provenance']}")

    if a.command == "check":
        quality_report(df)
    elif a.command == "charts":
        for name, make in REPORT_CHARTS.items():
            fig = make(df)
            if fig is not None:
                fig.savefig(OUT_DIR / f"{name}.png", dpi=160, facecolor="white", bbox_inches="tight")
                plt.close(fig)
                print("saved outputs/" + name + ".png")
    elif a.command == "select":
        req = {k: v for k, v in (("min_yield_strength", a.ys), ("min_tensile_strength", a.uts), ("min_youngs_modulus", a.e),
                                 ("max_density", a.density), ("min_elongation", a.elong)) if v is not None}
        selector = MaterialSelector(df)
        selector.train_classifiers()
        family = selector.predict_family(req)
        if family is not None:
            print((family.head(3) * 100).round(1).to_string())
        found = selector.recommend(req, k=a.k)
        print(found.drop(columns=["source"]).round(1).to_string() if len(found) else "No material meets every constraint.")
    else:
        heat = HeatTreatmentRecommender(df)
        if not heat.ok:
            sys.exit(heat.error)
        heat.train(evaluate=False)
        comp = {e.upper() if len(e) == 1 else e.capitalize(): getattr(a, e) for e in ("c", "mn", "si", "cr", "ni", "mo", "v")}
        targets = dict(yield_strength=a.ys, tensile_strength=a.uts, hardness=a.hardness, elongation=a.elong,
                       fatigue_strength=a.fatigue)
        res = heat.recommend(targets, material=a.material, composition=comp if any(v is not None for v in comp.values()) else None, k=a.k)
        print(res["recs"][["treatment", *heat.targets, "match_error"]].round(1).to_string())
        print("\nClosest records in your data:")
        print(res["evidence"].round(1).to_string())


if __name__ == "__main__":
    main()
