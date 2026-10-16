#!/usr/bin/env python3
"""
generate_vnnlib() function implementation from Table 2.1
Generates VNNLIB files for neural network verification
"""

import numpy as np
from PIL import Image
from torchvision import transforms


def generate_vnnlib(image_path, true_class, epsilon, output_path):
    """Generate VNNLIB file for verification.
    
    Args:
        image_path: Path to input image
        true_class: True class label
        epsilon: Perturbation bound
        output_path: Path to save VNNLIB file
    """
    img = Image.open(image_path).convert("RGB")
    x = transforms.ToTensor()(img).numpy()  # (3, 32, 32) NCHW
    flat = x.flatten()  # 3072 values in NCHW order

    with open(output_path, 'w') as f:
        # declare input variables
        for i in range(3072):
            f.write(f"(declare-const X_{i} Real)\n")
        # declare output variables
        for i in range(10):
            f.write(f"(declare-const Y_{i} Real)\n")

        # input bounds
        f.write("\n")
        for i, v in enumerate(flat):
            lb = max(0.0, float(v) - epsilon)
            ub = min(1.0, float(v) + epsilon)
            f.write(f"(assert (<= X_{i} {ub}))\n")
            f.write(f"(assert (>= X_{i} {lb}))\n")

        # output constraint: true class must dominate all others
        f.write("\n; Output constraints\n")
        for c in range(10):
            if c != true_class:
                f.write(f"(assert (>= Y_{true_class} Y_{c}))\n")


if __name__ == "__main__":
    generate_vnnlib(
        "/home/rayen/stage/auto_adv_testing/data/cifar10/cifar/test/2402_truck.png",
        true_class=9,  # truck
        epsilon=0.003,
        output_path="specs/truck_eps0.003.vnnlib"
    )
    print("Done")
