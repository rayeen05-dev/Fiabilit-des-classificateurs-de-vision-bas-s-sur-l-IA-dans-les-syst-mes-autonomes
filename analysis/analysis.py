#!/usr/bin/env python3
"""
analysis.py from Table 2.1
Generates robustness curve plots from verification results
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


csv_path = "results/results.csv"


df = pd.read_csv(csv_path)


# For one verification instance:
# UNSAT = certified robust
# TIMEOUT = not certified (solver ran out of time, NOT a counterexample)
# SAT = counterexample found (not robust)

df["marabou_unsat"] = (
    df["marabou_result"]
    .str.upper()
    .eq("UNSAT")
    .astype(int) * 100
)

df["crown_unsat"] = (
    df["crown_result"]
    .str.upper()
    .eq("UNSAT")
    .astype(int) * 100
)


plt.figure(figsize=(8, 5))

plt.plot(
    df["epsilon"],
    df["marabou_unsat"],
    marker="o",
    label="Marabou"
)

plt.plot(
    df["epsilon"],
    df["crown_unsat"],
    marker="o",
    label="αβ-CROWN"
)


plt.xlabel("ε")

# CORRECTION Q2.2 : renommer l'axe Y
plt.ylabel("% certifié robuste (UNSAT)")
plt.title("Profil de robustesse : Marabou vs αβ-CROWN")

plt.ylim(-10, 110)

plt.grid(True)
plt.legend()

# CORRECTION Q2.2 : ajouter note explicative
plt.figtext(
    0.5, -0.05,
    "Note : 0% pour Marabou indique un timeout (robustesse non prouvée),\n"
    "et non la présence d'un contre-exemple (SAT).",
    ha="center",
    fontsize=8,
    color="gray",
    style="italic"
)

plt.tight_layout()

Path("results/plots").mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    "results/plots/robustness_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

if __name__ == "__main__":
    print("analysis.py - generates robustness curve plots from verification results")
