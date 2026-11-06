"""
HEAT-TREATMENT RECOMMENDER  (the extra requirement)

Input : a material (a grade name OR a typed chemical composition) and the properties you want
        (any of yield strength, tensile strength, hardness, elongation, fatigue strength that your data contains).
Output: heat treatments (austenitize temperature, quench, temper) predicted to give those properties, each with an
        uncertainty, plus the closest real records of your data as evidence.

How it works
  A. Forward model : one Random Forest per property, composition + treatment -> property.
  B. Inverse search: hundreds of candidate treatments, built from the ranges found in YOUR data, are passed through
                     the model; the ones closest to your targets are returned.
  C. Evidence      : the nearest records by composition, shown with their data label.
It adapts to the columns present: no quench column -> quench is not used; no tempering data -> no tempering step.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from config import COMPOSITION_COLS, HT_TARGETS, MODEL_DIR, RANDOM_STATE

MIN_ROWS = 30                      # smallest number of usable rows for a property model
SLOW_COOLS = {"furnace", "air"}    # anneal / normalise: no tempering step


def build_features(df: pd.DataFrame, quench_levels) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in COMPOSITION_COLS:
        X[col] = df[col].fillna(0.0)                       # element not reported = not present
    X["austenitize_temp_C"] = df["austenitize_temp_C"]
    X["austenitize_time_min"] = df["austenitize_time_min"]
    X["has_temper"] = df["temper_temp_C"].notna().astype(int)
    X["temper_temp_C"] = df["temper_temp_C"].fillna(0)
    X["temper_time_min"] = df["temper_time_min"].fillna(0)
    for level in quench_levels:
        X[f"quench_{level}"] = (df["quench_method"] == level).astype(int)
    return X


def describe(row) -> str:
    quench = str(row.get("quench_method", "none"))
    aus = f"{row['austenitize_temp_C']:.0f}°C" if pd.notna(row.get("austenitize_temp_C")) else None
    temper = row.get("temper_temp_C")
    tempered = pd.notna(temper) and temper > 0
    if quench == "furnace" and not tempered:
        return f"Full anneal{': heat to ' + aus if aus else ''}, furnace cool"
    if quench == "air" and not tempered:
        return f"Normalize{': heat to ' + aus if aus else ''}, air cool"
    parts = []
    if aus:
        parts.append(f"Austenitize {aus}")
    if quench not in ("none", "nan", ""):
        parts.append(f"{quench} quench")
    if tempered:
        minutes = row.get("temper_time_min")
        parts.append(f"temper {temper:.0f}°C" + (f" for {minutes:.0f} min" if pd.notna(minutes) and minutes > 0 else ""))
    return ", ".join(parts) if parts else "No heat treatment recorded"


class HeatTreatmentRecommender:
    def __init__(self, df: pd.DataFrame):
        self.error, self.model, self.metrics = None, None, {}
        self.targets, self.levels, self.materials, self.comp_fields, self.comp_range = [], [], [], ["C"], {}
        has_composition = df["C"].notna()
        has_treatment = df["austenitize_temp_C"].notna() | df["temper_temp_C"].notna()
        usable = df[has_composition & has_treatment]
        self.targets = [t for t in HT_TARGETS if usable[t].notna().sum() >= MIN_ROWS]
        if len(usable) < MIN_ROWS or not self.targets:
            self.error = (f"This module needs at least {MIN_ROWS} rows that have a composition (at least carbon), a "
                          f"heat-treatment temperature (austenitize or temper) and a measured property. Your data has "
                          f"{int(has_composition.sum())} rows with composition, and {int((has_composition & has_treatment).sum())} "
                          f"of them with a heat-treatment temperature.")
            self.df = usable.iloc[0:0]
            return
        self.df = usable[usable[self.targets].notna().any(axis=1)].reset_index(drop=True)
        self.levels = sorted(set(self.df["quench_method"].astype(str)) - {"none", "nan", ""})
        self.X = build_features(self.df, self.levels)
        self.scale = self.df[self.targets].std().replace(0, 1).fillna(1)
        counts = self.df["material"].value_counts()
        grades = sorted(counts[counts >= 3].index)
        self.materials = grades if 1 <= len(grades) <= 80 else []          # a dropdown only makes sense for real grades
        frequency = {e: float((self.df[e].fillna(0) >= 0.05).mean()) for e in COMPOSITION_COLS}
        self.comp_fields = [e for e in sorted(frequency, key=frequency.get, reverse=True) if frequency[e] >= 0.05][:8] or ["C"]
        self.comp_range = {e: (float(self.df[e].min()), float(self.df[e].max()))
                           for e in COMPOSITION_COLS if self.df[e].notna().any()}

    @property
    def ok(self):
        return self.error is None

    # ---------- training ----------
    def train(self, evaluate=True):
        if not self.ok:
            return
        make = lambda: make_pipeline(SimpleImputer(strategy="median"),  # noqa: E731
                                     RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE))
        self.model = {}
        # Rows of the same steel (same composition) share a group, so a steel is never in both the training and the test
        # part. A plain random split would let the model "see" the steel it is tested on and overstate R2.
        groups = pd.factorize(self.df[COMPOSITION_COLS].round(2).fillna(-1).apply(lambda r: "|".join(map(str, r.values)), axis=1))[0]
        print("Heat-treatment property models (5-fold cross-validation, grouped by steel):")
        for target in self.targets:
            has = self.df[target].notna()
            self.model[target] = make().fit(self.X[has], self.df.loc[has, target])
            if evaluate and has.sum() >= 10:
                folds = min(5, len(set(groups[has.values])))
                predicted = cross_val_predict(make(), self.X[has], self.df.loc[has, target],
                                              groups=groups[has.values], cv=GroupKFold(folds))
                self.metrics[target] = dict(r2=float(r2_score(self.df.loc[has, target], predicted)),
                                            mae=float(mean_absolute_error(self.df.loc[has, target], predicted)),
                                            n=int(has.sum()))
                print(f"  {target:<17} R2 = {self.metrics[target]['r2']:.3f}   MAE = {self.metrics[target]['mae']:.1f}"
                      f"   (n={int(has.sum())})")
        joblib.dump(dict(models=self.model, levels=self.levels), MODEL_DIR / "heat_treatment_models.joblib")

    def _predict(self, X):
        """Mean prediction and spread between the trees (the uncertainty) for every property."""
        out = {}
        for target, pipe in self.model.items():
            trees = np.stack([tree.predict(pipe[:-1].transform(X)) for tree in pipe[-1].estimators_])
            out[target], out[target + "_std"] = trees.mean(0), trees.std(0)
        return pd.DataFrame(out, index=X.index)

    # ---------- inputs ----------
    def composition_of(self, material=None, composition=None) -> dict:
        comp = {c: 0.0 for c in COMPOSITION_COLS}
        if composition and any(v is not None for v in composition.values()):
            comp.update({k: float(v) for k, v in composition.items() if v is not None})
            return comp
        rows = self.df[self.df["material"].str.lower() == str(material).lower()]
        if rows.empty:
            raise ValueError(f"'{material}' has no heat-treatment records. Choose another material or type a composition.")
        comp.update({k: float(v) for k, v in rows[COMPOSITION_COLS].mean().fillna(0).items()})
        return comp

    def _candidates(self, comp, material=None):
        """Candidate treatments built only from conditions that occur in the data."""
        d = self.df
        rows_of_material = d[d["material"].str.lower() == str(material).lower()] if material else d
        reference = rows_of_material if len(rows_of_material) >= 3 else d
        aus_known = reference["austenitize_temp_C"].dropna()
        aus_options = ([round((float(aus_known.median()) + shift) / 5) * 5 for shift in (-20, 0, 20)]
                       if len(aus_known) else [np.nan])
        aus_time = float(d["austenitize_time_min"].median()) if d["austenitize_time_min"].notna().any() else np.nan
        temper = d["temper_temp_C"].dropna()
        temper = temper[temper > 0]
        grid = list(np.arange(np.floor(temper.min() / 25) * 25, temper.max() + 1, 25)) if len(temper) else []
        if len(grid) > 40:
            grid = list(np.linspace(temper.min(), temper.max(), 40).round(-1))
        times = [float(t) for t in d["temper_time_min"].dropna()[lambda s: s > 0].value_counts().head(3).index] or [np.nan]
        rows = []
        for quench in self.levels or ["none"]:
            for aus in aus_options:
                base = dict(**comp, austenitize_temp_C=aus, austenitize_time_min=aus_time, quench_method=quench)
                if quench not in SLOW_COOLS:
                    rows += [dict(base, temper_temp_C=float(t), temper_time_min=tm) for t in grid for tm in times]
                if quench in SLOW_COOLS or not grid or (d["temper_temp_C"].isna() & (d["quench_method"] == quench)).any():
                    rows.append(dict(base, temper_temp_C=np.nan, temper_time_min=np.nan))
        return pd.DataFrame(rows)

    # ---------- the main function ----------
    def recommend(self, targets: dict, material=None, composition=None, k=4):
        if not self.ok:
            raise ValueError(self.error)
        if self.model is None:
            self.train(evaluate=False)
        targets = {t: float(v) for t, v in targets.items() if t in self.targets and v is not None}
        if not targets:
            raise ValueError("Enter at least one target property.")
        comp = self.composition_of(material, composition)
        warnings = [f"{e} = {comp[e]:.2f} wt% is outside the range in your data ({lo:.2f} to {hi:.2f}): "
                    "the prediction is an extrapolation" for e, (lo, hi) in self.comp_range.items()
                    if comp.get(e, 0) > 0 and not (lo - 0.02 <= comp[e] <= hi + 0.02)]
        cand = self._candidates(comp, material if (material and not composition) else None)
        predictions = self._predict(build_features(cand, self.levels))
        error = sum(((predictions[t] - v) / self.scale[t]) ** 2 for t, v in targets.items())
        cand = pd.concat([cand, predictions], axis=1)
        cand["match_error"] = np.sqrt(error / len(targets))
        cand["treatment"] = cand.apply(describe, axis=1)
        cand = cand.sort_values("match_error").drop_duplicates("treatment")
        signature = (cand[self.targets] / (self.scale * 0.1)).round(0).astype(int).astype(str).agg("_".join, axis=1)
        recs = cand.loc[~signature.duplicated()].head(k).reset_index(drop=True)      # skip near-identical predictions
        # evidence: nearest records by composition, then by closeness to the targets
        spread = self.df[COMPOSITION_COLS].std().replace(0, np.nan).fillna(1).clip(lower=0.05)
        distance = (((self.df[COMPOSITION_COLS].fillna(0) - pd.Series(comp)) / spread) ** 2).sum(axis=1) ** 0.5
        near = self.df.assign(_d=distance).sort_values("_d").head(max(40, k * 10)).copy()
        gap = 0
        for t, v in targets.items():
            gap = gap + ((near[t] - v) / self.scale[t]).abs().fillna(3)
        near["match_error"] = gap / len(targets)
        near["treatment"] = near.apply(describe, axis=1)
        evidence = (near.sort_values("match_error").drop_duplicates("treatment").head(k)
                    [["treatment"] + self.targets + ["data_type", "source", "match_error"]].reset_index(drop=True))
        return dict(recs=recs, evidence=evidence, composition=comp, warnings=warnings, targets=targets)
