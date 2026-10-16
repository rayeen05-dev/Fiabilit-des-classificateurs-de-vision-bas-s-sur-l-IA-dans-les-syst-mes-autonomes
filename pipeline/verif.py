#!/usr/bin/env python3
"""
verif() function implementation from Table 2.1
Main verification loop for adversarial examples
"""

import os
import sys
import io
import time
import json
import pickle
import argparse
import resource
import multiprocessing
import csv
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from maraboupy import Marabou
import onnxruntime as ort
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


def collect_most_confident(session):
    """Collect single most-confident correctly-classified test image."""
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
    return [(best[0], best[1], best[2], best[3])]


def load_selected_images(session):
    """Load images listed in selected_images.json from the gtsrb pickle."""
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


def save_plot(x, adv_flat, c_true, c_adv, epsilon, session, out_dir, fname):
    """Plot for a single SAT counter-example."""
    inp_name = session.get_inputs()[0].name

    adv = adv_flat.reshape(32, 32, 3)
    adv_tensor = adv[np.newaxis].astype(np.float32)

    scores = session.run(None, {inp_name: adv_tensor})[0][0]
    adv_pred = int(np.argmax(scores))

    target_score = float(scores[c_adv])
    winner_score = float(scores[adv_pred])
    gap = target_score - winner_score
    agrees = (adv_pred == c_adv)

    print("\n==============================")
    print("SAT / float32 analysis")
    print("==============================")
    print("True class   :", classes[c_true])
    print("Target class :", classes[c_adv])
    print("Winner class :", classes[adv_pred])
    print("Target score :", target_score)
    print("Winner score :", winner_score)
    print("Gap          :", gap)
    print("Float32 agrees with SAT :", agrees)

    os.makedirs(os.path.join(out_dir, "analysis"), exist_ok=True)
    np.save(os.path.join(out_dir, "analysis", "adv_logits.npy"), scores)

    orig = x.reshape(32, 32, 3)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"{fname} | eps={epsilon:.3f} | true: {classes[c_true]} "
        f"-> model sees: {classes[adv_pred]}",
        fontsize=11,
    )

    axes[0].imshow(np.clip(orig, 0, 1))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(adv, 0, 1))
    axes[1].set_title("Adversarial")
    axes[1].axis("off")

    colors = ["steelblue"] * 10
    colors[c_adv] = "green"
    colors[adv_pred] = "red"

    axes[2].barh(classes, scores, color=colors)
    axes[2].invert_yaxis()
    axes[2].axvline(0, color="black")
    axes[2].set_title("Float32 logits")
    axes[2].text(
        0.02, 0.02,
        f"Target : {classes[c_adv]}\n"
        f"Winner : {classes[adv_pred]}\n"
        f"Gap = {gap:.6f}\n"
        f"Agreement = {agrees}",
        transform=axes[2].transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.9),
    )

    plt.tight_layout()
    plot_name = (
        f"{fname.replace('.png', '')}"
        f"_eps{epsilon:.3f}"
        f"_target_{classes[c_adv]}.png"
    )
    os.makedirs(os.path.join(out_dir, "adv_image"), exist_ok=True)
    plt.savefig(
        os.path.join(out_dir, "adv_image", plot_name),
        dpi=150, bbox_inches="tight",
    )
    plt.close()


def generate_robustness_plot(out_dir, all_results, epsilons):
    """
    Certified radius for an image = largest epsilon in the sweep such that
    every target class returned 'unsat' at that epsilon AND at every smaller
    tested epsilon.
    """
    csv_path = os.path.join(out_dir, "analysis", "verification_results.csv")
    if not os.path.exists(csv_path):
        print("[plot] verification_results.csv not found, skipping.")
        return
    if not all_results:
        print("[plot] no results to plot.")
        return

    df = pd.read_csv(csv_path)
    sorted_eps = sorted(set(float(e) for e in epsilons))

    def cert_radius(sub):
        radius = 0.0
        for eps in sorted_eps:
            block = sub[np.isclose(sub["epsilon"], eps)]
            if len(block) == 0:
                continue
            if (block["status"] == "unsat").all():
                radius = eps
            else:
                break  # stop at first ε that breaks certification
        return radius

    radii = df.groupby("image").apply(cert_radius).to_dict()

    rows = []
    for entry in all_results:
        fname = entry["fname"]
        rows.append({
            "image":  fname,
            "class":  entry["class"],
            "margin": float(entry["margin"]),
            "radius": float(radii.get(fname, 0.0)),
        })

    data = pd.DataFrame(rows)
    if data.empty:
        print("[plot] empty data frame, skipping plot.")
        return

    # regression
    if len(data) >= 2 and data["margin"].std() > 0:
        slope, intercept, r_value, _, _ = stats.linregress(
            data["margin"], data["radius"]
        )
        r2 = r_value ** 2
    else:
        slope, intercept, r2 = 0.0, float(data["radius"].mean()), 0.0

    # figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        data["margin"], data["radius"],
        c=data["margin"], cmap="coolwarm",
        s=90, edgecolors="black", linewidth=0.6, zorder=3,
    )

    if data["margin"].std() > 0:
        x_line = np.linspace(data["margin"].min(), data["margin"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "--",
                color="steelblue", alpha=0.7, zorder=2)

    for _, row in data.iterrows():
        label = row["image"].replace(".png", "") + f"_{row['class']}"
        ax.annotate(label, (row["margin"], row["radius"]),
                    fontsize=7, xytext=(5, 5), textcoords="offset points")

    ax.text(
        0.03, 0.95, f"R² = {r2:.3f}\nn = {len(data)}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow",
                  edgecolor="olive", alpha=0.9),
    )

    if data["radius"].mean() > 0:
        eps_var = data["radius"].max() - data["radius"].min()
        pct = 100 * eps_var / data["radius"].mean()
        ax.text(
            0.65, 0.08,
            f"Variation ε : {eps_var:.4f}\n({pct:.1f}% de la moyenne)",
            transform=ax.transAxes, fontsize=8, color="gray",
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Marge softmax", fontsize=10)

    ax.set_xlabel("Marge softmax (confiance de classification)", fontsize=11)
    ax.set_ylabel("Rayon de robustesse certifié (ε)", fontsize=11)
    ax.set_title(
        "Relation entre marge softmax et robustesse certifiée",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "analysis", "robustness_vs_margin.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {plot_path}")

    summary_csv = os.path.join(out_dir, "analysis", "robustness_summary.csv")
    data.to_csv(summary_csv, index=False)
    print(f"[plot] Summary CSV: {summary_csv}")


def verif(x, c_true, epsilon, timeout, model, session, out_dir, fname):
    """Verification loop over the 9 adversarial classes."""
    flat_x = x.flatten()
    results = {}

    adversarial_classes = [c for c in range(10) if c != c_true]

    analysis_dir = os.path.join(out_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    csv_path = os.path.join(analysis_dir, "verification_results.csv")
    write_header = not os.path.exists(csv_path)

    for c_adv in adversarial_classes:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        p = multiprocessing.Process(
            target=_worker,
            args=(model, flat_x, c_true, c_adv, epsilon, timeout, return_dict),
        )

        t0 = time.time()
        p.start()
        p.join(timeout + 10)
        elapsed = round(time.time() - t0, 1)

        if p.is_alive():
            p.terminate()
            p.join()
            status = "timeout"
        else:
            status = return_dict.get("status", "error")

        winner_class = ""
        target_score = ""
        winner_score = ""
        gap = ""
        agrees = ""

        if status == "sat" and "adv_flat" in return_dict:
            adv_flat = np.array(return_dict["adv_flat"])
            inp_name = session.get_inputs()[0].name

            adv = adv_flat.reshape(32, 32, 3)
            adv_tensor = adv[np.newaxis].astype(np.float32)
            scores = session.run(None, {inp_name: adv_tensor})[0][0]

            adv_pred = int(np.argmax(scores))
            winner_class = classes[adv_pred]
            target_score = float(scores[c_adv])
            winner_score = float(scores[adv_pred])
            gap = target_score - winner_score
            agrees = (adv_pred == c_adv)

            save_plot(x, adv_flat, c_true, c_adv, epsilon,
                      session, out_dir, fname)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "image", "epsilon", "true_class", "target_class",
                    "status", "time_s", "winner_class",
                    "target_score", "winner_score", "gap", "float32_agrees",
                ])
                write_header = False
            writer.writerow([
                fname, epsilon, classes[c_true], classes[c_adv],
                status, elapsed, winner_class,
                target_score, winner_score, gap, agrees,
            ])

        results[c_adv] = {"status": status, "time_s": elapsed}

        print(f"{classes[c_true]} vs {classes[c_adv]:12s} "
              f"-> {status.upper()} ({elapsed}s)")

    return results


def main():
    """Main function for the verification pipeline."""
    global_start = time.time()

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="simplified onnx model")
    p.add_argument("--epsilon", type=float, nargs="+",
                   default=[0.01, 0.03])
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--out_dir", default="./results_tt")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "adv_image"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "analysis"), exist_ok=True)

    if not os.path.isfile(args.model):
        print(f"ERROR : model not found : {args.model}")
        return

    session = ort.InferenceSession(args.model)

    images = load_selected_images(session)
    if not images:
        print("ERROR , no correctly classified images found")
        return

    all_results = []
    for fname, c_true, x, margin in images:
        print("\n===================================")
        print(f"Image  : {fname}")
        print(f"Margin : {margin:.6f}")
        print("===================================")

        img_entry = {
            "fname":    fname,
            "class":    classes[c_true],
            "margin":   margin,
            "epsilons": {},
        }

        for eps in args.epsilon:
            print(f"\nEPSILON {eps}")
            ver = verif(
                x, c_true, eps, args.timeout,
                args.model, session, args.out_dir, fname,
            )
            img_entry["epsilons"][str(eps)] = {
                classes[k]: v for k, v in ver.items()
            }

        all_results.append(img_entry)

        # 🔄 refresh plot + summary after every image
        generate_robustness_plot(args.out_dir, all_results, args.epsilon)

        # 💾 also checkpoint the partial report
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "config":    vars(args),
            "results":   all_results,
            "partial":   True,
        }
        with open(os.path.join(args.out_dir, "report_partial.json"), "w") as f:
            json.dump(checkpoint, f, indent=2)

    # ---- final plot (redundant but safe) ----
    generate_robustness_plot(args.out_dir, all_results, args.epsilon)

    # ---- resources / report ----
    total_queries = len(images) * len(args.epsilon) * 9
    runtime = time.time() - global_start
    resource_info = {
        "total_queries":       total_queries,
        "total_runtime_s":     round(runtime, 2),
        "peak_memory_mb":      round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2
        ),
        "avg_time_per_query_s": round(runtime / total_queries, 2)
                                if total_queries > 0 else 0,
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "config":    vars(args),
        "resources": resource_info,
        "results":   all_results,
    }

    i = 1
    while os.path.isfile(os.path.join(args.out_dir, f"report_{i}.json")):
        i += 1
    out = os.path.join(args.out_dir, f"report_{i}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report saved to {out}")


if __name__ == "__main__":
    main()
