# =============================================================================
# check_rainfall_coefficient_correlation.py
# =============================================================================
"""
Posterior correlation between the climate coefficients of the two pathways in
the main model (climate formulation with linear temperature).

The key number is the correlation between a_R (rainfall effect on spillover)
and bR_H (rainfall effect on amplification):
  * strongly negative: the two effects partly trade off against each other;
  * weak: each effect is supported by the data on its own.

Reads only the saved posterior (nothing is refitted).

Usage
-----
    python scripts/check_rainfall_coefficient_correlation.py
"""

import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKL = (PROJECT_ROOT / "results" / "final_model_comparison_v2" / "nested_sampling"
       / "all_climate__plague_deaths_reported_fit__final_v2.pkl")

PAIRS = [("aR", "bR_H"), ("aT", "bT_H"), ("ae", "be_H"),
         ("aR", "be_H"), ("aR", "bT_H"), ("ae", "bR_H")]


def weighted_corr(x, y, w):
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    return cov / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2))


def weighted_spearman(x, y, w):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return weighted_corr(rx, ry, w)


with open(PKL, "rb") as f:
    d = pickle.load(f)
res = d["results"]
names = list(d["param_names"])
w = np.exp(res.logwt - res.logz[-1])
w /= w.sum()
s = np.asarray(res.samples)

print("Posterior correlations between spillover and amplification coefficients")
print("(main model; importance-weighted)\n")
print(f"{'Pair':<16}{'Pearson r':>12}{'Spearman rho':>15}")
for a, b in PAIRS:
    x, y = s[:, names.index(a)], s[:, names.index(b)]
    print(f"{a + ' - ' + b:<16}{weighted_corr(x, y, w):>12.3f}{weighted_spearman(x, y, w):>15.3f}")

r = weighted_corr(s[:, names.index("aR")], s[:, names.index("bR_H")], w)
print()
if r < -0.5:
    print(f"a_R and bR_H are strongly negatively correlated (r = {r:.2f}): the two rainfall")
    print("effects partly trade off; describe them as a shift in the source of infection.")
elif r < -0.2:
    print(f"a_R and bR_H are moderately negatively correlated (r = {r:.2f}).")
else:
    print(f"a_R and bR_H are only weakly correlated (r = {r:.2f}): each rainfall effect")
    print("is supported by the data on its own.")