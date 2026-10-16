#!/usr/bin/env python3
"""
collect_margin_spectrum.py from Section 2.7.1
Collects and analyzes the margin spectrum of test images
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Constants from the original pipeline
classes = [
    "speed_20", "speed_30", "speed_50", "speed_60", 
    "speed_70", "speed_80", "end_speed_80", "speed_100", 
    "speed_120", "no_overtaking",
]
class_to_idx = {c: i for i, c in enumerate(classes)}
test_dir = "./data/test"


def get_image(path):
    """Helper function to load and preprocess an image"""
    img = Image.open(path).convert("RGB")
    x = transforms.ToTensor()(img)      # (3,32,32)
    x = x.unsqueeze(0)                  # (1,3,32,32)
    return x.numpy().astype(np.float32)


def collect_margin_spectrum(session):
    """Collect margin spectrum for all correctly classified test images.
    
    Args:
        session: ONNX Runtime inference session
        
    Returns:
        DataFrame with margin spectrum data
    """
    inp = session.get_inputs()[0].name
    margins = []
    image_data = []

    for fname in os.listdir(test_dir):
        if not fname.endswith(".png"):
            continue
        cls = fname.rsplit("_", 1)[1].replace(".png", "")
        if cls not in class_to_idx:
            continue
        c = class_to_idx[cls]
        x = get_image(os.path.join(test_dir, fname))
        scores = session.run(None, {inp: x})[0][0]
        pred = int(np.argmax(scores))
        
        if pred == c:
            s = np.sort(scores)[::-1]
            margin = float(s[0] - s[1])  # confidence margin
            margins.append(margin)
            image_data.append({
                "image": fname,
                "class": classes[c],
                "margin": margin,
                "scores": scores.tolist()
            })

    df = pd.DataFrame(image_data)
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Save margin spectrum data
    df.to_csv("results/margin_spectrum.csv", index=False)
    
    # Generate summary statistics
    summary = {
        "total_images": len(df),
        "mean_margin": float(df["margin"].mean()),
        "std_margin": float(df["margin"].std()),
        "min_margin": float(df["margin"].min()),
        "max_margin": float(df["margin"].max()),
        "median_margin": float(df["margin"].median()),
        "q25_margin": float(df["margin"].quantile(0.25)),
        "q75_margin": float(df["margin"].quantile(0.75)),
    }
    
    with open("results/margin_summary.json", "w") as f:
        import json
        json.dump(summary, f, indent=2)
    
    # Create margin histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df["margin"], bins=30, alpha=0.7, edgecolor="black")
    plt.xlabel("Confidence Margin")
    plt.ylabel("Frequency")
    plt.title("Margin Spectrum of CIFAR-10 Test Images")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/margin_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Create scatter plot (margin vs class distribution)
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    
    for idx, row in df.iterrows():
        class_idx = class_to_idx[row["class"]]
        plt.scatter(row["margin"], class_idx, 
                   c=[colors[class_idx]], s=50, alpha=0.7, edgecolors="black")
    
    plt.xlabel("Confidence Margin")
    plt.ylabel("Class Index")
    plt.title("Margin Distribution by Class")
    plt.yticks(range(10), classes)
    plt.grid(True, alpha=0.3)
    plt.savefig("results/margin_by_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Margin spectrum collected for {len(df)} correctly classified images")
    print(f"Summary statistics: mean margin = {summary['mean_margin']:.4f}")
    
    return df


if __name__ == "__main__":
    print("collect_margin_spectrum.py - collects and analyzes margin spectrum")
    print("Usage: Import and call collect_margin_spectrum(session)")
