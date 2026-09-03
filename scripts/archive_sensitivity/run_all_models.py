# =============================================================================
# run_all_models.py
# =============================================================================
"""
Run nested sampling for all four beta_H formulations.

Each model is fitted independently using scripts/run_nested_sampling.py.

Models:
    1. constant
    2. sigmoid
    3. humidity_only
    4. all_climate

Usage:
    python scripts/run_all_models.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_nested_sampling.py"


MODELS = [
    "constant",
    "sigmoid",
    "humidity_only",
    "all_climate",
]


def main():

    for model in MODELS:

        print("\n" + "=" * 70)
        print(f"Running beta_H model: {model}")
        print("=" * 70)

        command = [
            sys.executable,
            str(RUN_SCRIPT),
            "--betaH_mode",
            model,
        ]

        subprocess.run(
            command,
            check=True,
            cwd=PROJECT_ROOT,
        )

    print("\n" + "=" * 70)
    print("All four beta_H models completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()