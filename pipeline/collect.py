#!/usr/bin/env python3
"""
collect() function from Table 2.1
Collects correctly classified test images for verification
"""

import os
import json
import numpy as np
from PIL import Image
from torchvision import transforms

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


def collect_most_confident(session, selected_images=None):
    """Collect single most-confident correctly-classified test image.
    
    Alternative to collect() - collects from GTSRB pickle data
    
    Args:
        session: ONNX Runtime session
        selected_images: Optional list of pre-selected images
        
    Returns:
        List of (filename, class_idx, image_array, margin) tuples
    """
    inp = session.get_inputs()[0].name
    candidates = []

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
            margin = float(s[0] - s[1])
            candidates.append((fname, c, x, margin))

    if not candidates:
        return []

    best = max(candidates, key=lambda t: t[3])
    print(f"Most confident image: {best[0]} | class: {classes[best[1]]} "
          f"| margin: {best[3]:.4f}")
    return [best]


def load_selected_images():
    """Load images listed in selected_images.json from the gtsrb pickle."""
    import pickle
    with open("/home/rayen/stage/VeriX/models/gtsrb.pickle", "rb") as f:
        data = pickle.load(f)
    x_test = data["x_test"]

    with open("selected_images.json") as f:
        selected = json.load(f)

    images = []
    for item in selected:
        idx = int(item["file"].split("_")[0])
        img = x_test[idx].astype(np.float32) / 255.0  # (32,32,3)
        x = img[np.newaxis, ...]                      # (1,32,32,3)
        images.append((
            item["file"],
            int(item["true_class"]),
            x,
            float(item["margin"]),
        ))
    return images


def collect(session, use_gtsrb=False):
    """collect() function from Table 2.1
    
    Main collection function that selects correctly classified images
    for verification based on confidence margins.
    
    Args:
        session: ONNX Runtime inference session
        use_gtsrb: If True, use GTSRB data from pickle file
        
    Returns:
        List of (filename, class_idx, image_array, margin) tuples
    """
    if use_gtsrb:
        return load_selected_images()
    else:
        return collect_most_confident(session)


if __name__ == "__main__":
    print("collect() function - collect correctly classified test images")
    print("Usage: Import and call collect(session, use_gtsrb=False)")
