"""
MATERIAL SELECTION  (classification + nearest neighbours)

  Uses only the properties your data really contains (at least 10 measured values each). Nothing is filled in.

  * Classification: Random Forest and K-Nearest-Neighbours predict the material FAMILY from yield strength,
    tensile strength, Young's modulus and density. The better model (cross-validation) is used.
  * Nearest neighbours: records that break a hard design constraint are removed; scikit-learn's NearestNeighbors
    then ranks the rest by closeness to the requirement (log-scaled, standardised). Only the properties you
    constrain are used, so a record is never penalised for a property you did not ask about.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import COMPOSITION_COLS, EXPERIMENTAL, MODEL_DIR, RANDOM_STATE, SELECTION_PROPS

CONSTRAINTS = (("min_yield_strength", "yield_strength", ">="), ("min_tensile_strength", "tensile_strength", ">="),
               ("min_youngs_modulus", "youngs_modulus", ">="), ("max_density", "density", "<="),
               ("min_elongation", "elongation", ">="))


class MaterialSelector:
    MIN_VALUES = 10          # a property is offered only if at least this many rows have a measured value

    def __init__(self, df: pd.DataFrame):
        self.props = [p for p in SELECTION_PROPS if df[p].notna().sum() >= self.MIN_VALUES]
        self.missing = [p for p in SELECTION_PROPS if p not in self.props]
        self.clf, self.cv_scores, self.best_name, self.notice = None, {}, None, None
        self.pool = df[df[self.props].notna().any(axis=1)].reset_index(drop=True) if len(self.props) >= 2 else df.iloc[0:0]
        self.ready = len(self.props) >= 2 and len(self.pool) >= 5
        if not self.ready:
            self.notice = ("Material selection needs at least 5 records that have at least two of: yield strength, "
                           "tensile strength, Young's modulus, density.")
            return
        self.df = self.pool.dropna(subset=self.props).reset_index(drop=True)      # complete rows: used by the classifier
        self.X, self.y = self.df[self.props], self.df["material_family"]
        logs = {p: np.log(self.pool[p].dropna()) for p in self.props}
        self.centre = {p: float(v.mean()) for p, v in logs.items()}
        self.spread = {p: float(v.std()) or 1.0 for p, v in logs.items()}     # log: properties span orders of magnitude

    def accepts(self, key: str) -> bool:
        """Can this constraint be applied? (only if that property exists in the data)"""
        return any(k == key and col in self.props for k, col, _ in CONSTRAINTS)

    @property
    def families(self):
        return sorted(self.pool["material_family"].unique()) if self.ready else []

    # ---------- classification ----------
    def train_classifiers(self):
        if not self.ready:
            return
        counts = self.y.value_counts()
        usable = counts[counts >= 5].index
        if len(usable) < 2:
            self.notice = ("Family classification needs at least 2 material families with 5 or more records that have "
                           f"all of {', '.join(self.props)}. Your data has: {({k: int(v) for k, v in counts.items()})}.")
            return
        keep = self.y.isin(usable)
        X, y = self.X[keep], self.y[keep]
        models = {
            "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                           RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
            "K-Nearest Neighbours": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                  KNeighborsClassifier(n_neighbors=5, weights="distance")),
        }
        folds = StratifiedKFold(n_splits=min(5, int(y.value_counts().min())), shuffle=True, random_state=RANDOM_STATE)
        print("Material-family classification (cross-validation accuracy):")
        for name, model in models.items():
            score = cross_val_score(model, X, y, cv=folds, scoring="accuracy")
            self.cv_scores[name] = float(score.mean())
            print(f"  {name:<22} {score.mean():.3f} +/- {score.std():.3f}")
        self.best_name = max(self.cv_scores, key=self.cv_scores.get)
        self.clf = models[self.best_name].fit(X, y)
        joblib.dump(self.clf, MODEL_DIR / "family_classifier.joblib")

    def _target(self, req):
        """Requirement vector over self.props; a property without a requirement is NaN."""
        key = {col: k for k, col, _ in CONSTRAINTS if col in SELECTION_PROPS}
        return np.array([req.get(key[p], np.nan) for p in self.props], dtype=float)

    def predict_family(self, req: dict):
        if self.clf is None:
            return None
        target = np.where(np.isnan(self._target(req)), self.X.median().values, self._target(req))
        probabilities = self.clf.predict_proba(pd.DataFrame([target], columns=self.props))[0]
        return pd.Series(probabilities, index=self.clf.classes_).sort_values(ascending=False)

    # ---------- nearest-neighbour recommendation ----------
    def recommend(self, req: dict, k: int = 5, family=None, experimental_only=False) -> pd.DataFrame:
        """req may contain: min_yield_strength, min_tensile_strength, min_youngs_modulus, max_density, min_elongation.
        family: only this material family.  experimental_only: only rows labelled 'Experimental (published)'."""
        if not self.ready:
            return pd.DataFrame()
        ok = pd.Series(True, index=self.pool.index)
        if family:
            ok &= self.pool["material_family"] == family
        if experimental_only:
            ok &= self.pool["data_type"] == EXPERIMENTAL
        for key, col, op in CONSTRAINTS:                    # a missing value never satisfies a limit
            if key in req:
                ok &= (self.pool[col] >= req[key]) if op == ">=" else (self.pool[col] <= req[key])
        feasible = self.pool[ok].copy()
        if feasible.empty:
            return feasible
        keys = {col: key for key, col, _ in CONSTRAINTS}
        used = [p for p in self.props if keys[p] in req]    # rank only by the properties the user constrained
        if used:
            z = np.zeros((len(feasible), len(used)))
            for j, p in enumerate(used):
                z[:, j] = (np.log(feasible[p].values) - np.log(req[keys[p]])) / self.spread[p]
            feasible["distance"] = np.sqrt((z ** 2).sum(axis=1))
        else:
            feasible["distance"] = 0.0
        feasible["score"] = feasible["distance"]
        if "density" in feasible and feasible["density"].notna().any():     # small bonus for strength per weight
            specific = feasible["yield_strength"] / feasible["density"]
            feasible["score"] -= (0.15 * specific / specific.max()).fillna(0)
        feasible = (feasible.sort_values("score").drop_duplicates(subset=["material", *self.props]))
        feasible["composition"] = feasible.apply(
            lambda r: ", ".join(f"{e} {r[e]:.2f}".rstrip("0").rstrip(".") for e in COMPOSITION_COLS
                                if pd.notna(r[e]) and r[e] >= 0.05), axis=1)
        columns = ["material", "material_family", "composition", *self.props, "elongation", "data_type", "source"]
        for key, col, op in CONSTRAINTS:                    # how much each pick exceeds the requirement
            if key in req and col in feasible:
                feasible[f"margin_{col}"] = (feasible[col] / req[key] - 1) * 100 if op == ">=" else (1 - feasible[col] / req[key]) * 100
        columns += [c for c in feasible if c.startswith("margin_")]
        return feasible.head(k)[list(dict.fromkeys(columns))].reset_index(drop=True)
