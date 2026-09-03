# =============================================================================
# posterior_predictive.py
# =============================================================================
"""
Posterior predictive check for one fitted beta_H formulation: draw
parameter vectors from the (importance-weighted) posterior, simulate the
implied weekly-death curve for each, and plot the spread against the
observed data. Also produces a corner plot of the parameter posterior.

Usage:
    python scripts/posterior_predictive.py --betaH_mode humidity_only --n-draws 200
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import paths  # noqa: E402
from likelihood import PlagueLikelihood  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betaH_mode", required=True)
    ap.add_argument("--fit-target", default="plague_deaths_reported_fit",
                     help="must match the fit_target used in the run_nested_sampling.py run you're loading")
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stem = paths.run_stem(args.betaH_mode, args.fit_target)
    pkl_path = paths.NESTED_SAMPLING_DIR / f"{stem}.pkl"
    if not pkl_path.exists():
        raise SystemExit(
            f"No saved results at {pkl_path}. Run run_nested_sampling.py with "
            f"--betaH_mode {args.betaH_mode} --fit-target {args.fit_target} first."
        )

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    results = d["results"]
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    weights = weights / weights.sum()

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(samples), size=args.n_draws, p=weights, replace=True)
    draws = samples[idx]

    merged_df = pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
    lik = PlagueLikelihood(merged_df, betaH_mode=args.betaH_mode, fit_target=d["fit_target"])

    paths.ensure_output_dirs()

    # --- posterior predictive plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, theta in enumerate(draws):
        mu = lik.predict_weekly_deaths(theta)
        ax.plot(lik.merged_df[paths.DATE_COL], mu, color="C1", alpha=0.05,
                label="Posterior draws" if i == 0 else None)
    ax.plot(lik.merged_df[paths.DATE_COL], lik.y_obs, "ko-", markersize=3, linewidth=1,
            label=f"Observed ({d['fit_target']})")
    ax.set_xlabel("Week ending")
    ax.set_ylabel("Weekly deaths")
    ax.set_title(f"Posterior predictive check: {args.betaH_mode} ({d['fit_target']})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ppc_path = paths.FIGURES_DIR / f"posterior_predictive_{stem}.png"
    fig.savefig(ppc_path, dpi=150)
    print(f"[OK] Saved: {ppc_path}")

    # --- corner plot ---
    try:
        import corner
        fig2 = corner.corner(draws, labels=d["param_names"], show_titles=True)
        corner_path = paths.FIGURES_DIR / f"corner_{stem}.png"
        fig2.savefig(corner_path, dpi=150)
        print(f"[OK] Saved: {corner_path}")
    except ImportError:
        print("[skip] corner not installed (pip install corner) -- skipping corner plot")


if __name__ == "__main__":
    main()