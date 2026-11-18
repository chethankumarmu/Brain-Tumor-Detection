"""
Classify one MRI image with the trained model. Also returns a Grad-CAM heatmap and warnings.

Command line:
    python predict.py "D:\\some folder\\scan.jpg"

From Python (used by the web page):
    from predict import predict_scan
    result = predict_scan("scan.jpg")
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

from gradcam import GradCAM, array_to_base64_png, overlay_heatmap
from train_brain_tumor import build_model, make_transforms

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_best.pt")
DISPLAY_NAMES = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary",
                 "notumor": "No Tumor", "No_Tumor": "No Tumor", "no_tumor": "No Tumor"}
_cache = {}


def load_model():
    if "model" not in _cache:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model = build_model(len(checkpoint["classes"]), pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        _cache.update(model=model, device=device, classes=checkpoint["classes"],
                      img_size=checkpoint["img_size"],
                      gradcam=GradCAM(model, model.features[-1]))   # last conv block of EfficientNet-B0
    return _cache


def display_name(folder_name):
    return DISPLAY_NAMES.get(folder_name, folder_name)


def scan_warnings(image_rgb, probs_sorted):
    warnings = []
    r, g, b = [image_rgb[..., i].astype(np.float32) for i in range(3)]
    colourfulness = np.mean(np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b)))
    if colourfulness > 25:
        warnings.append("This looks like a colour photo, not a brain MRI. The result may be unreliable.")
    top1, top2 = probs_sorted[0], probs_sorted[1]
    if top1 < 0.60:
        warnings.append("Low confidence: no class reaches 60%. Treat this result with caution.")
    elif top1 - top2 < 0.25:
        warnings.append("Close call: the two most likely classes are near each other.")
    return warnings


def predict_scan(image_path):
    cache = load_model()
    transform = make_transforms(cache["img_size"], train=False)
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(cache["device"])

    with torch.no_grad():
        probs = torch.softmax(cache["model"](x), dim=1)[0].cpu().numpy()
    names = cache["classes"]
    probabilities = {display_name(n): float(p) for n, p in zip(names, probs)}
    order = np.argsort(-probs)
    best = display_name(names[order[0]])

    cam, _ = cache["gradcam"](x, class_idx=int(order[0]))
    scan = np.array(image.resize((cache["img_size"], cache["img_size"])))
    overlay = overlay_heatmap(scan, cam)

    return {
        "best": best,
        "probabilities": probabilities,
        "warnings": scan_warnings(scan, probs[order]),
        "scan_png": array_to_base64_png(scan),
        "gradcam_png": array_to_base64_png(overlay),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python predict.py "path\\to\\scan.jpg"')
        sys.exit(1)
    result = predict_scan(sys.argv[1])
    print("Most likely:", result["best"])
    for name, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:12s} {p*100:5.1f}%")
    for w in result["warnings"]:
        print("WARNING:", w)
    print("Heatmap saved as heatmap.png")
    import base64
    with open("heatmap.png", "wb") as f:
        f.write(base64.b64decode(result["gradcam_png"]))
