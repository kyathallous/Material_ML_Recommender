"""
Every chart, written once. Each function returns a matplotlib Figure (or None if the data cannot support it).
  * the web app turns a Figure into inline SVG:   to_svg(fig)
  * the report export saves the same Figures as PNG files:   python main.py charts
"""
import io
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from config import COMPOSITION_COLS, EXPERIMENTAL, PROP_LABEL, PROPERTY_COLS, REFERENCE, TREATMENT_NUM, UNVERIFIED  # noqa: E402

matplotlib.rcParams.update({
    "svg.fonttype": "none", "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#475569", "xtick.color": "#64748b", "ytick.color": "#64748b",
    "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 0.8, "axes.axisbelow": True,
})
PALETTE = ["#4f46e5", "#0d9488", "#f59e0b", "#e11d48", "#0ea5e9", "#7c3aed", "#65a30d", "#64748b"]
SIZE = (6.4, 4.2)


def to_svg(fig) -> str:
    """Figure -> SVG text that can be placed directly in a web page."""
    if fig is None:
        return ""
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    svg = buffer.getvalue()
    svg = svg[svg.find("<svg"):]
    svg = re.sub(r'<svg width="[\d.]+pt" height="[\d.]+pt"', "<svg", svg, count=1)
    return svg.replace("<svg", '<svg style="width:100%;height:auto"', 1)


def _families(df):
    families = sorted(df["material_family"].unique())
    return families, {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(families)}


def _label(prop):
    return f"{PROP_LABEL.get(prop, prop)}" + (" (MPa)" if "strength" in prop else "")


# ------------------------------------------------------------------ material selection
def ashby(df, picks=None, density_limit=None):
    """Stiffness against density (log-log): the classic material-selection chart."""
    data = df.dropna(subset=["density", "youngs_modulus"])
    if data.empty:
        return None
    families, colour = _families(data)
    fig, ax = plt.subplots(figsize=SIZE)
    for f in families:
        s = data[data["material_family"] == f]
        ax.scatter(s["density"], s["youngs_modulus"], s=22, alpha=0.55, color=colour[f], label=f, linewidths=0)
    if picks is not None and len(picks):
        ax.scatter(picks["density"], picks["youngs_modulus"], s=150, facecolors="none", edgecolors="#0f172a",
                   linewidths=1.8, label="Recommended", zorder=5)
    if density_limit:
        ax.axvline(density_limit, color="#e11d48", ls="--", lw=1, alpha=.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    lo_x, hi_x = data["density"].min(), data["density"].max()
    lo_y, hi_y = data["youngs_modulus"].min(), data["youngs_modulus"].max()
    ax.set_xlim(lo_x / 1.35, hi_x * 1.35)
    ax.set_ylim(lo_y / 1.5, hi_y * 1.5)
    ax.set_xticks([t for t in (1, 2, 3, 4, 5, 8, 10, 15, 20) if lo_x / 1.35 <= t <= hi_x * 1.35])
    ax.set_yticks([t for t in (10, 20, 50, 100, 200, 400, 800) if lo_y / 1.5 <= t <= hi_y * 1.5])
    ax.set_xlabel("Density (g/cm³)")
    ax.set_ylabel("Young's modulus (GPa)")
    ax.legend(fontsize=7.5, frameon=False)
    return fig


def strength(df, picks=None, req=None):
    data = df.dropna(subset=["tensile_strength", "yield_strength"])
    if data.empty:
        return None
    families, colour = _families(data)
    fig, ax = plt.subplots(figsize=SIZE)
    for f in families:
        s = data[data["material_family"] == f]
        ax.scatter(s["tensile_strength"], s["yield_strength"], s=22, alpha=0.55, color=colour[f], label=f, linewidths=0)
    if picks is not None and len(picks):
        ax.scatter(picks["tensile_strength"], picks["yield_strength"], s=150, facecolors="none", edgecolors="#0f172a",
                   linewidths=1.8, zorder=5)
    for key, draw in (("min_yield_strength", ax.axhline), ("min_tensile_strength", ax.axvline)):
        if req and key in req:
            draw(req[key], color="#e11d48", ls="--", lw=1, alpha=.7)
    ax.set_xlabel("Tensile strength (MPa)")
    ax.set_ylabel("Yield strength (MPa)")
    ax.legend(fontsize=7.5, frameon=False)
    return fig


def family_probabilities(proba):
    top = proba.head(4)[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 1.0 + 0.3 * len(top)))
    ax.barh(top.index, top.values * 100, color="#4f46e5", height=0.55)
    for i, v in enumerate(top.values * 100):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8.5, color="#334155")
    ax.set_xlim(0, 110)
    ax.grid(False)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    return fig


# ------------------------------------------------------------------ heat treatment
def tempering(records, recs, prop):
    """Records of the data (dots) and the recommended treatments (stars): property against tempering temperature."""
    data = records.dropna(subset=["temper_temp_C", prop])
    data = data[data["temper_temp_C"] > 0]
    starred = recs.dropna(subset=["temper_temp_C"]) if recs is not None else None
    if data.empty and (starred is None or starred.empty):
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, (quench, s) in enumerate(data.groupby(data["quench_method"].astype(str))):
        name = "records" if quench in ("none", "nan") else f"records, {quench} quench"
        ax.scatter(s["temper_temp_C"], s[prop], s=22, alpha=0.6, color=PALETTE[i % len(PALETTE)], label=name, linewidths=0)
    if starred is not None and len(starred):
        ax.scatter(starred["temper_temp_C"], starred[prop], s=170, marker="*", color="#f59e0b", edgecolors="#0f172a",
                   linewidths=0.8, zorder=6, label="Recommended (ML)")
    ax.set_xlabel("Tempering temperature (°C)")
    ax.set_ylabel(_label(prop))
    ax.legend(fontsize=7.5, frameon=False)
    return fig


# ------------------------------------------------------------------ data analysis
def provenance(counts):
    colour = {EXPERIMENTAL: "#16a34a", REFERENCE: "#4f46e5", UNVERIFIED: "#94a3b8"}
    fig, ax = plt.subplots(figsize=(6.4, 0.7))
    left, total = 0, sum(counts.values())
    for name, n in counts.items():
        ax.barh(0, n, left=left, color=colour.get(name, "#94a3b8"), height=0.55)
        if n / total > 0.08:
            ax.text(left + n / 2, 0, f"{n / total:.0%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left += n
    ax.axis("off")
    return fig


def coverage(df):
    """How many rows have a value for each column: tells you what the models can and cannot do."""
    cols = [c for c in PROPERTY_COLS + TREATMENT_NUM + ["C", "Mn", "Cr", "Ni", "Mo"] if c in df]
    have = (df[cols].notna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.27 * len(cols)))
    ax.barh(have.index, have.values, height=0.6,
            color=["#16a34a" if v >= 80 else "#f59e0b" if v >= 30 else "#e11d48" for v in have.values])
    for i, v in enumerate(have.values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8, color="#334155")
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of rows that have a value")
    ax.grid(False)
    return fig


def correlation(df):
    cols = [c for c in PROPERTY_COLS + ["C", "Mn", "Cr", "Ni", "Mo"] + TREATMENT_NUM
            if c in df and df[c].notna().sum() >= 10 and df[c].nunique() > 1]
    if len(cols) < 3:
        return None
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if np.isfinite(corr.iloc[i, j]):
                ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(corr.iloc[i, j]) > 0.6 else "#334155")
    fig.colorbar(image, ax=ax, shrink=0.8, label="correlation")
    return fig


def family_box(df, prop="yield_strength"):
    data = df.dropna(subset=[prop])
    families = [f for f in sorted(data["material_family"].unique()) if (data["material_family"] == f).sum() >= 3]
    if not families:
        return None
    fig, ax = plt.subplots(figsize=SIZE)
    boxes = ax.boxplot([data.loc[data["material_family"] == f, prop] for f in families], patch_artist=True, widths=0.55)
    for i, box in enumerate(boxes["boxes"]):
        box.set(facecolor=PALETTE[i % len(PALETTE)], alpha=0.35, edgecolor=PALETTE[i % len(PALETTE)])
    ax.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(_label(prop))
    return fig


def composition_scatter(df, x="C", y="yield_strength"):
    data = df.dropna(subset=[x, y])
    data = data[data[x] > 0]
    if len(data) < 10:
        return None
    families, colour = _families(data)
    fig, ax = plt.subplots(figsize=SIZE)
    for f in families:
        s = data[data["material_family"] == f]
        ax.scatter(s[x], s[y], s=20, alpha=0.6, color=colour[f], label=f, linewidths=0)
    ax.set_xlabel(f"{x} (wt %)")
    ax.set_ylabel(_label(y))
    ax.legend(fontsize=7.5, frameon=False)
    return fig


# Charts exported for a report:  python main.py charts
REPORT_CHARTS = {
    "01_coverage": coverage, "02_correlation": correlation, "03_ashby_stiffness_density": ashby,
    "04_yield_vs_tensile": strength, "05_yield_by_family": family_box,
    "06_carbon_vs_yield": lambda d: composition_scatter(d, "C", "yield_strength"),
    "07_carbon_vs_tensile": lambda d: composition_scatter(d, "C", "tensile_strength"),
}
