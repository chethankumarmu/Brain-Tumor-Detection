"""
Web page for the brain tumor classifier. Runs on your PC only.

Start:   python app.py
Open:    http://127.0.0.1:5000
"""
import json
import os
import tempfile

from flask import Flask, jsonify, render_template, request

from predict import predict_scan

HERE = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(HERE, "static", "charts")
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DISCLAIMER = ("Research prototype for learning and demonstration only. It is NOT a medical diagnosis. "
              "Results must be reviewed by a qualified radiologist.")
LIMITS = ("The model was trained on one dataset. It can miss tumors, and it can be confident and still be wrong. "
          "Glioma and meningioma are the two classes it most often confuses.")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", disclaimer=DISCLAIMER, limits=LIMITS)


@app.route("/charts")
def charts():
    metrics = {}
    metrics_path = os.path.join(CHART_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    charts_list = [
        ("loss_before_after.png", "Loss curves: before and after"),
        ("loss_ours.png", "Our run (early stopping)"),
        ("loss_reference.png", "Reference run (600 epochs, overfitting)"),
        ("per_image_probability.png", "Probability of the correct class, every test scan"),
        ("confusion_matrix.png", "Confusion matrix (test set)"),
        ("split_counts.png", "Scans per class in each split"),
        ("feature_maps_early.png", "Feature maps: early layer"),
        ("feature_maps_deeper.png", "Feature maps: deeper layer"),
        ("gradcam_examples.png", "Grad-CAM examples (one per class)"),
    ]
    available = [(f, t) for f, t in charts_list if os.path.exists(os.path.join(CHART_DIR, f))]
    return render_template("charts.html", metrics=metrics, charts=available, limits=LIMITS, disclaimer=DISCLAIMER)


@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files.get("scan")
    if file is None or file.filename == "":
        return jsonify(error="Please choose an image first."), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error="Please upload a JPG, PNG, BMP or TIFF image."), 400
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        file.save(path)
        result = predict_scan(path)
    except Exception as exc:  # unreadable file, corrupt image, etc.
        return jsonify(error=f"Could not read this image: {exc}"), 400
    finally:
        os.remove(path)
    result["disclaimer"] = DISCLAIMER
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
