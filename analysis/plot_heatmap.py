#!/usr/bin/env python3
"""
plot_heatmap.py from Table 2.1
Generates heatmap visualization from verification data
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


original_path = "/home/rayen/stage/auto_adv_testing/data/cifar10/cifar/test/2402_truck.png"

adv_path = "results/images/adv.png"


out_dir = Path("results/plots")
out_dir.mkdir(parents=True, exist_ok=True)


# Load images
orig = np.array(
    Image.open(original_path)
    .convert("RGB")
).astype(float) / 255.0


adv = np.array(
    Image.open(adv_path)
    .convert("RGB")
).astype(float) / 255.0


# Difference
diff = np.abs(orig - adv)


# Save images
plt.figure(figsize=(4,4))
plt.imshow(orig)
plt.axis("off")
plt.title("Image originale")
plt.savefig(
    out_dir/"original.png",
    bbox_inches="tight",
    dpi=300
)
plt.close()


plt.figure(figsize=(4,4))
plt.imshow(adv)
plt.axis("off")
plt.title("Image perturbée")
plt.savefig(
    out_dir/"adversarial.png",
    bbox_inches="tight",
    dpi=300
)
plt.close()


plt.figure(figsize=(5,4))
plt.imshow(diff.mean(axis=2))
plt.axis("off")
plt.title("Carte de perturbation |x-x_adv|")
plt.colorbar()

plt.savefig(
    out_dir/"perturbation_heatmap.png",
    bbox_inches="tight",
    dpi=300
)

plt.close()


print("Heatmap generated")

if __name__ == "__main__":
    print("plot_heatmap.py - generates heatmap visualization from verification data")
