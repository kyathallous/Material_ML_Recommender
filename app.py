"""
Web app: Data-Driven Machine Learning Recommender for Mechanical Material Selection.
Start it with START_APP.bat (Windows), start_app.sh (Mac / Linux) or  python run_app.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from flask import Flask, redirect, render_template, request, url_for  # noqa: E402

import charts  # noqa: E402
from charts import to_svg  # noqa: E402
from config import EXPERIMENTAL, SELECTION_PROPS, APP_VERSION, COMPOSITION_COLS, PROP_LABEL, PROP_UNIT, PROPERTY_COLS, RAW_DIR, TREATMENT_NUM  # noqa: E402
from data_manager import banner, build_dataset  # noqa: E402
from heat_treatment import HeatTreatmentRecommender  # noqa: E402
from selection import MaterialSelector  # noqa: E402

app = Flask(__name__)
app.config.update(TEMPLATES_AUTO_RELOAD=True, SEND_FILE_MAX_AGE_DEFAULT=0)
STATE = {}
PAGES_WITHOUT_DATA = {"data", "reload_data", "health", "static"}
SELECT_FIELDS = [("ys", "Minimum yield strength", "MPa", "min_yield_strength"),
                 ("uts", "Minimum tensile strength", "MPa", "min_tensile_strength"),
                 ("e", "Minimum Young's modulus", "GPa", "min_youngs_modulus"),
                 ("density", "Maximum density", "g/cm³", "max_density"),
                 ("elong", "Minimum elongation", "%", "min_elongation")]
MODE_TAG = {"empty": ("NO DATA YET", "empty"), "experimental": ("EXPERIMENTAL DATA", "good"),
            "published": ("EXPERIMENTAL + REFERENCE DATA", "good"), "mixed": ("SOME SOURCES NOT STATED", "mixed")}


def load_all():
    """(Re)read data/raw and train every model. Runs at start-up and when 'Reload data' is pressed."""
    started = time.time()
    print("Loading data ...")
    df, info = build_dataset()
    print(f"  {info['mode']}: {info['rows']} rows from {len([f for f in info['files'] if f['status'] == 'ok'])} file(s)")
    selector, heat = MaterialSelector(df), HeatTreatmentRecommender(df)
    if info["mode"] != "empty":
        selector.train_classifiers()
        if heat.ok:
            heat.train()
        else:
            print("Heat-treatment recommender not available:", heat.error)
    STATE.update(df=df, info=info, sel=selector, ht=heat, banner=banner(info), loaded=time.strftime("%H:%M:%S"),
                 seconds=round(time.time() - started, 1))
    print(f"Ready ({STATE['seconds']} s).")


@app.before_request
def guard():
    if "df" not in STATE:
        load_all()
    if STATE["info"]["mode"] == "empty" and request.endpoint not in PAGES_WITHOUT_DATA:
        return redirect(url_for("data"))


@app.context_processor
def inject():
    tag, css = MODE_TAG[STATE["info"]["mode"]]
    return dict(banner=STATE["banner"], version=APP_VERSION, mode_tag=tag, mode_css=css, loaded=STATE["loaded"],
                has_ht=STATE["ht"].ok, PROP_LABEL=PROP_LABEL, PROP_UNIT=PROP_UNIT)


def number(name):
    text = request.form.get(name, "").strip().replace(",", ".")
    try:
        return float(text) if text else None
    except ValueError:
        return None


@app.route("/")
def overview():
    df, info, sel, ht = STATE["df"], STATE["info"], STATE["sel"], STATE["ht"]
    stats = dict(rows=len(df), materials=df["material"].nunique(), families=df["material_family"].nunique(),
                 files=len([f for f in info["files"] if f["status"] == "ok"]))
    return render_template("overview.html", page="overview", stats=stats, prov=info["provenance"],
                           prov_chart=to_svg(charts.provenance(info["provenance"])),
                           ashby=to_svg(charts.ashby(df)), strength=to_svg(charts.strength(df)),
                           cv=sel.cv_scores, best=sel.best_name, notice=sel.notice,
                           ht_metrics=ht.metrics if ht.ok else {}, ht_error=ht.error)


@app.route("/analysis")
def analysis():
    df = STATE["df"]
    return render_template("analysis.html", page="analysis", coverage=to_svg(charts.coverage(df)),
                           corr=to_svg(charts.correlation(df)), box=to_svg(charts.family_box(df)),
                           carbon_ys=to_svg(charts.composition_scatter(df, "C", "yield_strength")),
                           carbon_uts=to_svg(charts.composition_scatter(df, "C", "tensile_strength")))


@app.route("/select", methods=["GET", "POST"])
def select():
    sel = STATE["sel"]
    shown = [(name, label, unit, key) for name, label, unit, key in SELECT_FIELDS
             if sel.ready and (sel.accepts(key) or (name == "elong" and STATE["df"]["elongation"].notna().sum() >= 10))]
    df_all = STATE["df"]
    experimental = df_all[df_all["data_type"] == EXPERIMENTAL]
    reference_only = [p for p in sel.props if p in SELECTION_PROPS and experimental[p].notna().sum() == 0] if sel.ready else []
    ctx = dict(page="select", form={}, result=None, sel=sel, fields=shown, reference_only=reference_only)
    if request.method == "POST" and sel.ready:
        form = {name: (number(name) if name in {f[0] for f in shown} else None) for name, *_ in SELECT_FIELDS}
        k = int(number("k") or 5)
        family = request.form.get("family") or None
        only_exp = request.form.get("experimental_only") == "1"
        req = {key: form[f] for key, f in (("min_yield_strength", "ys"), ("min_tensile_strength", "uts"),
                                           ("min_youngs_modulus", "e"), ("max_density", "density"),
                                           ("min_elongation", "elong")) if form[f] is not None}
        ctx["form"] = {**{a: ("" if b is None else b) for a, b in form.items()}, "k": k, "family": family or "",
                       "experimental_only": only_exp}
        keys = {"youngs_modulus": "min_youngs_modulus", "density": "max_density"}
        blocked = [PROP_LABEL[p] for p in reference_only if keys.get(p) in req] if only_exp else []
        if not req:
            ctx["error"] = "Enter at least one requirement."
        elif blocked:
            ctx["error"] = (f"'Experimental rows only' cannot be combined with a {' / '.join(blocked)} limit: no experimental file "
                            "contains that property, it exists only in the reference data. Untick the box or clear that limit.")
        else:
            recs = sel.recommend(req, k=k, family=family, experimental_only=only_exp)
            proba = sel.predict_family(req)
            ctx["result"] = dict(
                recs=recs.to_dict("records"), n=len(recs),
                family=to_svg(charts.family_probabilities(proba)) if proba is not None else None,
                ashby=to_svg(charts.ashby(STATE["df"], picks=recs, density_limit=req.get("max_density"))),
                strength=to_svg(charts.strength(STATE["df"], picks=recs, req=req)))
    return render_template("select.html", **ctx)


@app.route("/heat", methods=["GET", "POST"])
def heat():
    ht = STATE["ht"]
    ctx = dict(page="heat", form={}, result=None, ht=ht)
    if ht.ok and request.method == "POST":
        material = request.form.get("material") or None
        composition = {e: number(f"c_{e}") for e in ht.comp_fields}
        typed = any(v is not None for v in composition.values())
        targets = {t: number(t) for t in ht.targets}
        k = int(number("k") or 4)
        ctx["form"] = {"material": material, "k": k, **{f"c_{e}": ("" if v is None else v) for e, v in composition.items()},
                       **{t: ("" if v is None else v) for t, v in targets.items()}}
        if all(v is None for v in targets.values()):
            ctx["error"] = "Enter at least one target property."
        elif not typed and not material:
            ctx["error"] = "Choose a material or type a composition."
        else:
            try:
                res = ht.recommend(targets, material=None if typed else material, composition=composition if typed else None, k=k)
            except ValueError as error:
                ctx["error"] = str(error)
            else:
                prop = next(iter(res["targets"]))
                records = ht.df if typed or not material else ht.df[ht.df["material"] == material]
                rows = [dict(treatment=r["treatment"], match_error=round(r["match_error"], 3),
                             preds=[(t, r[t], r[t + "_std"]) for t in ht.targets]) for r in res["recs"].to_dict("records")]
                ctx["result"] = dict(rows=rows, evidence=res["evidence"].to_dict("records"), warnings=res["warnings"],
                                     chart=to_svg(charts.tempering(records, res["recs"], prop)), prop=prop,
                                     label="your composition" if typed else material, targets=res["targets"])
    return render_template("heat.html", **ctx)


@app.route("/data")
def data():
    df, info = STATE["df"], STATE["info"]
    shown = COMPOSITION_COLS[:7] + PROPERTY_COLS + TREATMENT_NUM + ["quench_method"]
    coverage = []
    for col in shown:
        n = int((df[col].notna() & (df[col].astype(str) != "none")).sum())
        numeric = col != "quench_method" and n > 0
        coverage.append((col, n, round(100 * n / len(df)) if len(df) else 0,
                         f"{df[col].min():.4g}" if numeric else "", f"{df[col].max():.4g}" if numeric else ""))
    columns = ["material", "material_family", "yield_strength", "tensile_strength", "hardness", "elongation",
               "fatigue_strength", "austenitize_temp_C", "temper_temp_C", "data_type"]
    preview = df.groupby("source", sort=False).head(4).head(24)[columns].round(1).to_dict("records")
    return render_template("data.html", page="data", info=info, coverage=coverage, preview=preview, raw_dir=str(RAW_DIR),
                           sel=STATE["sel"], ht=STATE["ht"], seconds=STATE["seconds"])


@app.route("/reload", methods=["POST"])
def reload_data():
    load_all()
    return redirect(url_for("data"))


@app.route("/health")
def health():
    info = STATE["info"]
    return dict(version=APP_VERSION, mode=info["mode"], rows=info["rows"], provenance=info["provenance"],
                files=[f["name"] for f in info["files"]], heat_treatment_ready=STATE["ht"].ok)
