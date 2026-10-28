# =============================================================================
# check_rainfall_only_bounds.py
# =============================================================================
"""
Which parameters of the rainfall-only formulation sit at their prior bounds,
and did the widened-prior refit move them away?

Reads only the two saved runs (nothing is refitted):
    rainfall_only__plague_deaths_reported_fit__seeded.pkl        (main run)
    rainfall_only__plague_deaths_reported_fit__seeded_wide.pkl   (widened priors)

For every parameter it prints the prior bounds of both runs, the posterior
median and 95% credible interval, and flags a parameter as "AT BOUND" when
more than 5% of its posterior mass lies in the outer 2% of the prior range.

Usage
-----
    python scripts/check_rainfall_only_bounds.py
"""

import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIT_TARGET = "plague_deaths_reported_fit"
DIR = PROJECT_ROOT / "results" / "rainfall_only_check" / "nested_sampling"
RUNS = {
    "main": DIR / f"rainfall_only__{FIT_TARGET}__seeded.pkl",
    "wide": DIR / f"rainfall_only__{FIT_TARGET}__seeded_wide.pkl",
}
EDGE_FRACTION = 0.02   # outer 2% of the prior range
MASS_FLAG = 0.05       # flag if more than 5% of posterior mass is there


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d["results"]
    w = np.exp(res.logwt - res.logz[-1])
    w /= w.sum()
    bounds = d.get("bounds")
    if isinstance(bounds, dict):
        bounds = [bounds[n] for n in d["param_names"]]
    return d, np.asarray(res.samples), w, bounds, float(res.logz[-1])


def summary(v, w, lo_b, hi_b):
    o = np.argsort(v)
    q = np.interp([0.025, 0.5, 0.975], np.cumsum(w[o]), v[o])
    edge = EDGE_FRACTION * (hi_b - lo_b)
    m_lo = float(w[v <= lo_b + edge].sum())
    m_hi = float(w[v >= hi_b - edge].sum())
    flag = []
    if m_lo > MASS_FLAG:
        flag.append(f"AT LOWER BOUND ({100 * m_lo:.0f}%)")
    if m_hi > MASS_FLAG:
        flag.append(f"AT UPPER BOUND ({100 * m_hi:.0f}%)")
    return q, ", ".join(flag)


loaded = {k: load(p) for k, p in RUNS.items() if p.exists()}
for k, p in RUNS.items():
    if k not in loaded:
        print(f"[not found] {k}: {p}")

for k, (d, s, w, bounds, logz) in loaded.items():
    print(f"\n=== {k} run: log Z = {logz:.2f}")
    print(f"{'Parameter':<10}{'Prior bounds':<22}{'Median (95% CrI)':<34}Flag")
    for j, name in enumerate(d["param_names"]):
        lo_b, hi_b = map(float, bounds[j])
        q, flag = summary(s[:, j], w, lo_b, hi_b)
        print(f"{name:<10}{f'[{lo_b:g}, {hi_b:g}]':<22}"
              f"{f'{q[1]:.3f} ({q[0]:.3f}, {q[2]:.3f})':<34}{flag}")

if len(loaded) == 2:
    dm, dw = loaded["main"], loaded["wide"]
    changed = [n for j, n in enumerate(dm[0]["param_names"])
               if tuple(map(float, dm[3][j])) != tuple(map(float, dw[3][dw[0]['param_names'].index(n)]))]
    print(f"\nParameters whose bounds were widened: {', '.join(changed) or 'none'}")
    print(f"log Z change (wide - main): {dw[4] - dm[4]:+.2f}")