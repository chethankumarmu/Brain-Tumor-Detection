"""
Creates every chart and metric file used by the web page. Run once after training:

    python make_charts.py --data_dir "D:\\chethan\\Projects\\Brain Tumor Detection\\Epic and CSCR hospital Dataset"

Writes into  static/charts/  and  static/charts/metrics.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets

from gradcam import GradCAM, array_to_base64_png  # noqa: F401  (kept for parity with predict.py)
from predict import MODEL_PATH, display_name, load_model
from train_brain_tumor import make_transforms

HERE = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(HERE, "static", "charts")
COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]


def loss_axes(ax, hist, title, ymax=None, mark_best=None):
    ax.plot(hist["epochs"], hist["train_loss"], c="b", label="Train loss")
    ax.plot(hist["epochs"], hist["val_loss"], c="r", label="Validation loss")
    if mark_best:
        ax.axvline(mark_best, c="g", linestyle="--", label=f"Best epoch ({mark_best})")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    if ymax:
        ax.set_ylim(-0.05, ymax)
    ax.grid(True)
    ax.legend()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, name), dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    os.makedirs(CHART_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = load_model()
    model, classes, img_size = cache["model"], cache["classes"], cache["img_size"]
    names = [display_name(c) for c in classes]
    metrics = {}

    # ---------- 1. loss curves: before (600-epoch reference) and after (our run) ----------
    ref = json.load(open(os.path.join(HERE, "reference_600_epochs.json")))
    hist_path = os.path.join(HERE, "train_history.json")
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        ours = {"epochs": [x["epoch"] for x in h["history"]],
                "train_loss": [x["train_loss"] for x in h["history"]],
                "val_loss": [x["val_loss"] for x in h["history"]]}
        best = h["best_epoch"]
    else:
        ours = json.load(open(os.path.join(HERE, "history_ours.json")))
        best = ours["best_epoch"]
    ymax = max(max(ref["val_loss"]), max(ours["val_loss"])) * 1.05
    fig, ax = plt.subplots(figsize=(9, 5))
    loss_axes(ax, ours, f"Our run: {len(ours['epochs'])} rounds, stopped early", ymax, best)
    save(fig, "loss_ours.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    loss_axes(ax, ref, "Reference: 600 epochs, no fixes (overfitting)", ymax)
    save(fig, "loss_reference.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    loss_axes(axes[0], ref, "BEFORE: 600 epochs, no fixes", ymax)
    loss_axes(axes[1], ours, "AFTER: early stopping + fixes", ymax, best)
    save(fig, "loss_before_after.png")

    # ---------- 2. test set predictions ----------
    test_set = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=make_transforms(img_size, False))
    loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    all_probs, all_true = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            all_probs.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
            all_true.append(labels.numpy())
    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_true)
    y_pred = probs.argmax(axis=1)
    order_classes = test_set.classes
    # map folder order -> display names (test_set.classes is the folder order)
    names_in_test = [display_name(c) for c in order_classes]

    # ---------- 3. per-image probability plot (like the reference) ----------
    correct_prob = probs[np.arange(len(y_true)), y_true]
    idx = np.argsort(y_true, kind="stable")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(correct_prob[idx], color="#1f77b4", linewidth=1)
    start = 0
    for k, name in enumerate(names_in_test):
        count = int((y_true == k).sum())
        ax.axvspan(start, start + count, alpha=0.08, color=COLORS[k % 4])
        ax.text(start + count / 2, 0.02, name, ha="center", fontsize=11)
        start += count
        if k < len(names_in_test) - 1:
            ax.axvline(start, c="r", linestyle="--")
    ax.set_title("Probability given to the CORRECT class, every test scan (grouped by class)", fontsize=13)
    ax.set_xlabel("Test scan (sorted by true class)")
    ax.set_ylabel("Probability of the correct class")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    save(fig, "per_image_probability.png")

    # ---------- 4. confusion matrix and per-class scores ----------
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names_in_test)), names_in_test, rotation=30, ha="right")
    ax.set_yticks(range(len(names_in_test)), names_in_test)
    for i in range(len(names_in_test)):
        for j in range(len(names_in_test)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test confusion matrix (accuracy {(y_true == y_pred).mean():.3f})")
    fig.colorbar(im)
    save(fig, "confusion_matrix.png")
    report = classification_report(y_true, y_pred, target_names=names_in_test, digits=3, output_dict=True, zero_division=0)
    metrics["test_accuracy"] = float((y_true == y_pred).mean())
    metrics["per_class"] = {n: {k: round(float(report[n][k]), 3) for k in ("precision", "recall", "f1-score", "support")}
                            for n in names_in_test}

    # ---------- 5. training / test split bars ----------
    tr = datasets.ImageFolder(os.path.join(args.data_dir, "train"))
    labels = np.array(tr.targets)
    idx_train, idx_val = train_test_split(np.arange(len(labels)), test_size=0.15, random_state=42, stratify=labels)
    counts = {"Train": np.bincount(labels[idx_train], minlength=len(names_in_test)),
              "Validation": np.bincount(labels[idx_val], minlength=len(names_in_test)),
              "Test": np.bincount(y_true, minlength=len(names_in_test))}
    metrics["split_counts"] = {k: {n: int(v[i]) for i, n in enumerate(names_in_test)} for k, v in counts.items()}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(names_in_test))
    w = 0.27
    for k, (label, vals) in enumerate(counts.items()):
        bars = ax.bar(x + (k - 1) * w, vals, w, label=label)
        ax.bar_label(bars, fontsize=9)
    ax.set_xticks(x, names_in_test)
    ax.set_ylabel("Number of scans")
    ax.set_title("How the scans are split for each class")
    ax.legend()
    save(fig, "split_counts.png")

    # ---------- 6. feature maps: one sample scan, early and deeper layer ----------
    sample_path = test_set.samples[0][0]
    from PIL import Image
    x_in = make_transforms(img_size, False)(Image.open(sample_path).convert("RGB")).unsqueeze(0).to(device)
    layer_outputs = {}
    with torch.no_grad():
        out = x_in
        for i, block in enumerate(model.features):
            out = block(out)
            layer_outputs[i] = out[0].cpu().numpy()
    for layer_idx, tag in [(0, "early"), (3, "deeper")]:
        maps = layer_outputs[layer_idx]
        n = min(16, maps.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(14, 14))
        for k, ax in enumerate(axes.flat):
            ax.axis("off")
            if k < n:
                ax.imshow(maps[k], cmap="viridis")
        fig.suptitle(f"Feature maps, layer {layer_idx} ({tag}): first {n} of {maps.shape[0]} filters", fontsize=14)
        save(fig, f"feature_maps_{tag}.png")

    # ---------- 7. Grad-CAM examples: one scan per class ----------
    from gradcam import overlay_heatmap
    gradcam = cache["gradcam"]
    fig, axes = plt.subplots(2, len(names_in_test), figsize=(4 * len(names_in_test), 8))
    for k in range(len(names_in_test)):
        rows = np.where(y_true == k)[0]
        if len(rows) == 0:
            continue
        path, _ = test_set.samples[rows[0]]
        xi = make_transforms(img_size, False)(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        cam, pred = gradcam(xi)
        scan = np.array(Image.open(path).convert("RGB").resize((img_size, img_size)))
        axes[0, k].imshow(scan, cmap="gray")
        axes[0, k].set_title(f"True: {names_in_test[k]}")
        axes[1, k].imshow(overlay_heatmap(scan, cam))
        axes[1, k].set_title(f"Model: {names[pred]}")
    for ax in axes.flat:
        ax.axis("off")
    save(fig, "gradcam_examples.png")

    with open(os.path.join(CHART_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print("Charts written to", CHART_DIR)
    print("Test accuracy:", round(metrics["test_accuracy"], 4))


if __name__ == "__main__":
    main()
