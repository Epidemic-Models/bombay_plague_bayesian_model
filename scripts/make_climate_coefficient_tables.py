# =============================================================================
# make_climate_coefficient_tables.py
# =============================================================================
"""
Build the climate-coefficient tables for the main text and the SI directly
from the final nested-sampling posteriors. Nothing is refitted.

Tables written (LaTeX, ready to \\input or paste):

  1. table_main_climate_coefficients.tex
       Main-text Table 2: climate effects on spillover and amplification in
       the climate formulation with linear temperature (the reported model),
       one row per climate variable, the two
       pathways side by side. Includes the temperature at which spillover
       peaks, in degrees Celsius.

  2. table_si_spillover_by_formulation.tex
       SI: the four spillover climate coefficients (a_T, a_TT, a_e, a_R) in
       every formulation of amplification. Supports the main-text sentence
       "the temperature peak in spillover appeared in every formulation,
       whereas the rainfall effect on spillover was resolved only in the
       formulations in which rainfall could also act on amplification".

  3. table_si_amplification_coefficients.tex
       SI (replaces Table S11): the climate coefficients on amplification in
       the two climate formulations (linear temperature, temperature peak)
       and the single-covariate formulations, with log Z.

  4. table_si_best_fit_parameters.tex
       SI (replaces Table S3): maximum-likelihood parameter values for all
       seven formulations.

Seven formulations
------------------
"all_climate" is the climate formulation with linear temperature and is the
reported model (main-text Table 2). "all_climate_quadT" adds a quadratic
temperature term to amplification; it appears in the SI tables only.
Delta log Z is reported relative to all_climate, matching main-text Table 1
(negative = higher evidence).

Also written: climate_coefficients_all_formulations.csv (every number used).
Every table is also printed to the terminal.

Conventions
-----------
* Posterior summaries use dynesty's importance weights directly:
  weighted median and 2.5% / 97.5% quantiles.
* "Resolved" (bold) means the 95% credible interval excludes zero.
* The temperature peak of spillover is computed per posterior sample as
  z* = -a_T / (2 a_TT) on the standardized scale, converted to degrees
  Celsius with the mean and standard deviation used to standardize the
  temperature covariate. Those two constants are recovered exactly from the
  prepared data (x_T is a linear function of z_T), so they always match the
  climate preprocessing.
* The fall of spillover one standard deviation above its peak is
  exp(a_TT), because beta_R is quadratic in z_T on the log scale.

Usage
-----
    python scripts/make_climate_coefficient_tables.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
RESULTS_ROOT = PROJECT_ROOT / "results"
MAIN_DIR = RESULTS_ROOT / "final_model_comparison_v2" / "nested_sampling"

# Same entries as posterior_analysis.py and make_bayesian_model_fits_figure.py.
MODELS = {
    "all_climate_quadT": {
        "label": "Climate, temperature peak",
        "path": RESULTS_ROOT / "all_climate_quadT_check" / "nested_sampling"
                / f"all_climate_quadT__{FIT_TARGET}__seeded.pkl",
    },
    "all_climate": {
        "label": "Climate, linear temperature",
        "path": MAIN_DIR / f"all_climate__{FIT_TARGET}__final_v2.pkl",
    },
    "humidity_only": {
        "label": "Vapour pressure only",
        "path": MAIN_DIR / f"humidity_only__{FIT_TARGET}__final_v2.pkl",
    },
    "rainfall_only": {
        "label": "Rainfall only",
        "path": RESULTS_ROOT / "rainfall_only_check" / "nested_sampling"
                / f"rainfall_only__{FIT_TARGET}__seeded.pkl",
    },
    "temperature_only": {
        "label": "Temperature only",
        "path": RESULTS_ROOT / "temperature_only_check" / "nested_sampling"
                / f"temperature_only__{FIT_TARGET}__seeded.pkl",
    },
    "constant": {
        "label": "Constant",
        "path": MAIN_DIR / f"constant__{FIT_TARGET}__final_v2.pkl",
    },
    "sigmoid": {
        "label": "Single shift",
        "path": MAIN_DIR / f"sigmoid__{FIT_TARGET}__final_v2.pkl",
    },
}

# Keep False for the final tables.
SKIP_MISSING_MODELS = False

# The reported model: main-text Table 2, and the reference for Delta log Z.
REFERENCE_MODEL = "all_climate"

OUTPUT_DIR = RESULTS_ROOT / "final_figures" / "climate_coefficient_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DECIMALS = 2

SPILLOVER_COEFS = [("aT", r"$a_T$"), ("aTT", r"$a_{TT}$"),
                   ("ae", r"$a_e$"), ("aR", r"$a_R$")]
AMPLIFICATION_COEFS = [("bR_H", "rainfall", r"$b_R^{(H)}$"),
                       ("be_H", "vapour pressure", r"$b_e^{(H)}$"),
                       ("bT_H", "temperature", r"$b_T^{(H)}$"),
                       ("bTT_H", "temperature$^2$", r"$b_{TT}^{(H)}$")]

# LaTeX names for the best-fit table, in display order.
PARAM_TEX = {
    "N0": r"$N_0$", "C_seed": r"$C_{\mathrm{seed}}$", "gamma": r"$\gamma$",
    "a0": r"$a_0$", "aT": r"$a_T$", "aTT": r"$a_{TT}$", "ae": r"$a_e$",
    "aR": r"$a_R$", "betaH0": r"$\beta_{H0}$",
    "bT_H": r"$b_T^{(H)}$", "bTT_H": r"$b_{TT}^{(H)}$",
    "be_H": r"$b_e^{(H)}$", "bR_H": r"$b_R^{(H)}$",
    "theta": r"$\theta$", "t_star": r"$t^*$", "kappa": r"$\kappa_{\mathrm{sig}}$",
    "psi": r"$\psi$", "nb_kappa": r"$\kappa_{\mathrm{NB}}$",
}


# =============================================================================
# Helpers
# =============================================================================

def load(model: str) -> dict:
    with open(MODELS[model]["path"], "rb") as f:
        d = pickle.load(f)
    if d.get("fit_target", FIT_TARGET) != FIT_TARGET:
        raise ValueError(f"{model}: fitted to {d['fit_target']!r}, not {FIT_TARGET!r}.")
    if d.get("betaH_mode", model) != model:
        raise ValueError(f"{model}: file says betaH_mode={d['betaH_mode']!r}.")
    res = d["results"]
    w = np.exp(res.logwt - res.logz[-1])
    d["_w"] = w / w.sum()
    d["_samples"] = np.asarray(res.samples)
    d["_logZ"] = float(d.get("logZ", res.logz[-1]))
    d["_maxlogL"] = float(np.max(res.logl))
    d["_best"] = np.asarray(res.samples[int(np.argmax(res.logl))])
    return d


def wquant(values: np.ndarray, w: np.ndarray, q=(0.025, 0.5, 0.975)) -> np.ndarray:
    order = np.argsort(values)
    cw = np.cumsum(w[order])
    return np.interp(q, cw, values[order])


def summarize(d: dict, name: str) -> dict | None:
    names = list(d["param_names"])
    if name not in names:
        return None
    v = d["_samples"][:, names.index(name)]
    lo, med, hi = wquant(v, d["_w"])
    sign = 1 if med >= 0 else -1
    p_sign = float(d["_w"][(v > 0) if sign > 0 else (v < 0)].sum())
    return {"median": med, "lo": lo, "hi": hi,
            "resolved": bool(lo > 0 or hi < 0), "p_sign": p_sign,
            "sign": ">" if sign > 0 else "<"}


def num(x: float, plus: bool = False) -> str:
    s = f"{x:.{DECIMALS}f}"
    if s.startswith("-0." ) and float(s) == 0:
        s = s[1:]
    if plus and not s.startswith("-"):
        s = "+" + s
    return s


def cell(s: dict | None, plus: bool = False, prefix: str = "") -> str:
    """'$\\mathbf{-2.22}$ ($-2.82$, $-1.87$)'; bold when resolved; '--' if absent."""
    if s is None:
        return "--"
    m = num(s["median"], plus)
    m = rf"$\mathbf{{{m}}}$" if s["resolved"] else f"${m}$"
    return f"{prefix}{m} (${num(s['lo'])}$, ${num(s['hi'])}$)"


def temperature_constants(df: pd.DataFrame) -> tuple[float, float]:
    """Mean and SD (degrees F) used to standardize temperature: x = mean + sd * z."""
    sd, mean = np.polyfit(df["zT"].to_numpy(float), df["xT"].to_numpy(float), 1)
    resid = np.max(np.abs(df["xT"] - (mean + sd * df["zT"])))
    if resid > 1e-6:
        raise RuntimeError("xT is not an exact linear function of zT; check the "
                           "climate preprocessing before converting to degrees C.")
    return float(mean), float(sd)


def write_and_show(path: Path, text: str, title: str) -> None:
    """Save a table and print it to the terminal."""
    path.write_text(text)
    bar = "=" * 90
    print(f"\n{bar}\n{title}\n  -> {path.name}\n{bar}")
    print(text.rstrip())


def f_to_c(f):
    return (np.asarray(f) - 32.0) * 5.0 / 9.0


# =============================================================================
# Main
# =============================================================================

def main():

    df = pd.read_csv(paths.MERGED_CSV)
    t_mean_f, t_sd_f = temperature_constants(df)
    t_mean_c, t_sd_c = float(f_to_c(t_mean_f)), t_sd_f * 5.0 / 9.0
    print(f"Temperature standardization: mean {t_mean_c:.2f} C, SD {t_sd_c:.2f} C")

    # ------------------------------------------------------------------ load
    fits = {}
    for model, spec in MODELS.items():
        if spec["path"].exists():
            fits[model] = load(model)
        elif SKIP_MISSING_MODELS:
            print(f"[WARNING] skipping {model}: {spec['path']} not found")
        else:
            raise SystemExit(f"Posterior for {model!r} not found:\n  {spec['path']}\n"
                             "Edit its path in MODELS or set SKIP_MISSING_MODELS = True.")

    order = sorted(fits, key=lambda m: fits[m]["_logZ"], reverse=True)
    if REFERENCE_MODEL not in fits:
        raise SystemExit(f"The reference model {REFERENCE_MODEL!r} is needed.")
    ref_logZ = fits[REFERENCE_MODEL]["_logZ"]

    # ------------------------------------------------------------ CSV of all
    rows = []
    for m in order:
        for name in [c for c, _ in SPILLOVER_COEFS] + [c for c, _, _ in AMPLIFICATION_COEFS]:
            s = summarize(fits[m], name)
            if s is not None:
                rows.append({"formulation": MODELS[m]["label"], "betaH_mode": m,
                             "logZ": fits[m]["_logZ"], "coefficient": name, **s})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "climate_coefficients_all_formulations.csv",
                              index=False)

    # ===================================================== 1. main Table 2
    ac = fits[REFERENCE_MODEL]
    names = list(ac["param_names"])
    aT = ac["_samples"][:, names.index("aT")]
    aTT = ac["_samples"][:, names.index("aTT")]
    if np.any(aTT >= 0):
        print("[WARNING] some posterior samples have a_TT >= 0 (no peak); "
              "they are excluded from the peak-temperature summary.")
    ok = aTT < 0
    w_ok = ac["_w"][ok] / ac["_w"][ok].sum()
    t_peak_c = f_to_c(t_mean_f + t_sd_f * (-aT[ok] / (2 * aTT[ok])))
    tp_lo, tp_med, tp_hi = wquant(t_peak_c, w_ok)
    drop_lo, drop_med, drop_hi = wquant(np.exp(aTT), ac["_w"])

    s = {k: summarize(ac, k) for k in ["aT", "aTT", "ae", "aR", "bT_H", "be_H", "bR_H"]}

    main_tex = rf"""\begin{{table}}[ht]
\centering
\caption{{Climate effects on spillover, $\beta_R(t)$, and amplification, $\beta_H(t)$, in the climate formulation with linear temperature. Values are posterior medians with 95\% credible intervals, on the log scale per standard deviation of each standardized climate variable. Bold entries have 95\% credible intervals that exclude zero. Rainfall acts in opposite directions on the two pathways, and no climate variable has a resolved effect of the same sign on both.}}
\label{{tab:climate_coefficients}}
\begin{{tabular}}{{lll}}
\toprule
Climate variable & Spillover, $\beta_R(t)$ & Amplification, $\beta_H(t)$ \\
\midrule
Rainfall
  & {cell(s['aR'], plus=True)}
  & {cell(s['bR_H'], plus=True)} \\[4pt]
Temperature
  & \begin{{tabular}}[t]{{@{{}}l@{{}}}}
      {cell(s['aT'], plus=True, prefix='linear: ')}\\
      {cell(s['aTT'], plus=True, prefix='quadratic: ')}\\
      peak at ${tp_med:.1f}^{{\circ}}$C (${tp_lo:.1f}$--${tp_hi:.1f}^{{\circ}}$C)
    \end{{tabular}}
  & {cell(s['bT_H'], plus=True)} \\[4pt]
Vapour pressure
  & {cell(s['ae'], plus=True)}
  & {cell(s['be_H'], plus=True)} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write_and_show(OUTPUT_DIR / "table_main_climate_coefficients.tex", main_tex,
                   "MAIN TEXT TABLE 2: climate effects, climate formulation with linear temperature")

    # Numbers quoted in the main-text paragraph on temperature
    temp_text = (
        f"Spillover peak: {tp_med:.1f} C (95% CrI {tp_lo:.1f}-{tp_hi:.1f} C); "
        f"study-period mean {t_mean_c:.1f} C; 1 SD = {t_sd_c:.2f} C. "
        f"One SD above the peak, spillover falls to {100 * drop_med:.0f}% "
        f"(95% CrI {100 * drop_lo:.0f}-{100 * drop_hi:.0f}%) of its maximum."
    )
    write_and_show(OUTPUT_DIR / "main_text_temperature_numbers.txt", temp_text + "\n",
                   "NUMBERS FOR THE MAIN-TEXT TEMPERATURE PARAGRAPH")

    # ========================== 2. SI: spillover coefficients by formulation
    lines = []
    for m in order:
        cells = [cell(summarize(fits[m], c)) for c, _ in SPILLOVER_COEFS]
        lines.append(f"{MODELS[m]['label']:<27} & " + " & ".join(cells) + r" \\")
    si_spill = (
        r"""\begin{table}[ht]
\centering
\small
\caption{Climate effects on spillover, $\beta_R(t)$, in each formulation of amplification, ordered by log-evidence. Values are posterior medians with 95\% credible intervals, on the log scale per standard deviation of each standardized climate variable. Bold entries have 95\% credible intervals that exclude zero.}
\label{tab:si_spillover_by_formulation}
\begin{tabular}{lllll}
\toprule
Formulation & $a_T$ (temperature) & $a_{TT}$ (temperature$^2$) & $a_e$ (vapour pressure) & $a_R$ (rainfall) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write_and_show(OUTPUT_DIR / "table_si_spillover_by_formulation.tex", si_spill,
                   "SI TABLE S15 (new): spillover coefficients in every formulation")

    # ================= 3. SI: amplification coefficients (replaces Table S11)
    amp_models = [m for m in order
                  if m in ("all_climate_quadT", "all_climate", "rainfall_only",
                           "humidity_only", "temperature_only")]
    lines = []
    for m in amp_models:
        d = fits[m]
        present = [(c, lab, tex) for c, lab, tex in AMPLIFICATION_COEFS
                   if summarize(d, c) is not None]
        for i, (c, lab, tex) in enumerate(present):
            sm = summarize(d, c)
            first = i == 0
            k = len(d["param_names"])
            lines.append(
                (f"{MODELS[m]['label']} & {k} & ${d['_logZ']:.2f}$ & " if first
                 else " & & & ")
                + f"{tex} ({lab}) & {cell(sm)} & "
                + f"$P({sm['sign']}0) = {sm['p_sign']:.3f}$ \\\\"
            )
    si_amp = (
        r"""\begin{table}[ht]
\centering
\small
\caption{Climate effects on amplification, $\beta_H(t)$, in the two climate formulations (linear temperature and temperature peak) and in the formulations with a single climate variable, ordered by log-evidence. $k$ is the number of fitted parameters. Values are posterior medians with 95\% credible intervals; bold entries exclude zero. $P$ is the posterior probability that the coefficient has the sign of its median. All formulations were fitted with one live point placed at the formulation's best fit from a preliminary run (Main Text Methods).}
\label{tab:si_amplification_coefficients}
\begin{tabular}{lrrlll}
\toprule
Formulation & $k$ & $\log Z$ & Coefficient & Median (95\% CrI) & $P$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write_and_show(OUTPUT_DIR / "table_si_amplification_coefficients.tex", si_amp,
                   "SI TABLE S11 (replacement): amplification coefficients")

    # =========================== 4. SI: best-fit parameters (replaces Table S3)
    all_params = [p for p in PARAM_TEX if any(p in fits[m]["param_names"] for m in order)]
    header = " & ".join(MODELS[m]["label"] for m in order)
    lines = []
    for p in all_params:
        vals = []
        for m in order:
            pn = list(fits[m]["param_names"])
            if p not in pn:
                vals.append("--")
                continue
            v = fits[m]["_best"][pn.index(p)]
            vals.append(f"{v:,.0f}" if p == "N0" else f"${v:.3f}$")
        lines.append(f"{PARAM_TEX[p]} & " + " & ".join(vals) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"$\log L$ & " + " & ".join(f"${fits[m]['_maxlogL']:.2f}$" for m in order) + r" \\")
    lines.append(r"$\log Z$ & " + " & ".join(f"${fits[m]['_logZ']:.2f}$" for m in order) + r" \\")
    si_best = (
        r"""\begin{table}[ht]
\centering
\scriptsize
\caption{Maximum-likelihood parameter values for the seven formulations of amplification, ordered by log-evidence. Each value is the highest-likelihood sample in the formulation's nested-sampling run. Parameters not present in a formulation are shown as --.}
\label{tab:si_best_fit_parameters}
\begin{tabular}{l""" + "r" * len(order) + r"""}
\toprule
Parameter & """ + header + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write_and_show(OUTPUT_DIR / "table_si_best_fit_parameters.tex", si_best,
                   "SI TABLE S3 (replacement): maximum-likelihood parameters")

    # ------------------------------------------------------------- report
    print(f"\nFormulations (ordered by log Z; Delta relative to "
          f"{MODELS[REFERENCE_MODEL]['label']}, negative = higher evidence):")
    for m in order:
        print(f"  {MODELS[m]['label']:<27} log Z = {fits[m]['_logZ']:.2f}  "
              f"(Delta = {ref_logZ - fits[m]['_logZ']:+.2f})")
    print("\nSpillover coefficients resolved (95% CrI excludes 0):")
    for m in order:
        res = [c for c, _ in SPILLOVER_COEFS if summarize(fits[m], c)["resolved"]]
        print(f"  {MODELS[m]['label']:<27} {', '.join(res) if res else 'none'}")
    print(f"\n[OK] Tables written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()