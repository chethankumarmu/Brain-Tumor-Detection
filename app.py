"""
Web page for the brain tumor classifier. Runs on your PC only.

Start:   python app.py
Open:    http://127.0.0.1:5000
"""
import json
import os
import tempfile

from flask import Flask, jsonify, redirect, render_template, request

from predict import predict_scan

HERE = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(HERE, "static", "charts")
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DISCLAIMER = ("Research prototype for learning and demonstration only. It is NOT a medical diagnosis. "
              "Results must be reviewed by a qualified radiologist.")
LIMITS = ("The model was trained on one dataset. It can miss tumors, and it can be confident and still be wrong. "
          "Glioma and meningioma are the two classes it most often confuses.")

# (file, title, caption shown under the card)
CHARTS = [
    ("loss_before_after.png", "Loss curves: before and after",
     "Before: 600 epochs with no fixes. After: early stopping and the fixes."),
    ("confusion_matrix.png", "Confusion matrix, test set", "Test set: scans the model never saw in training."),
    ("confusion_matrix_train.png", "Confusion matrix, training set",
     "For comparison only, not a fair score."),
    ("split_3d.png", "Scans per class: train and test (3D bars)", None),
    ("per_image_probability.png", "Probability of the correct class, every test scan",
     "Each dot is one test scan. Dots near the top are confident and correct."),
    ("feature_maps_early.png", "Feature maps: early layer", "Simple patterns such as edges and brightness."),
    ("feature_maps_deeper.png", "Feature maps: deeper layer", "More complex shapes that the model combines."),
    ("gradcam_examples.png", "Grad-CAM examples, one scan per class",
     "Where the model looked for each class."),
]

app = Flask(__name__)


@app.route("/")
def index():
    metrics = {}
    metrics_path = os.path.join(CHART_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    charts = []
    for file, title, caption in CHARTS:
        if not os.path.exists(os.path.join(CHART_DIR, file)):
            continue
        if caption is None and "split_counts" in metrics:
            sc = metrics["split_counts"]
            caption = (f"Scans per class: train {sum(sc['Train'].values()):,}, "
                       f"test {sum(sc['Test'].values()):,}")
        charts.append({"file": file, "title": title, "caption": caption or ""})
    return render_template("index.html", metrics=metrics, charts=charts, limits=LIMITS, disclaimer=DISCLAIMER)


@app.route("/charts")
def charts_redirect():
    # The charts used to live on their own page. They are now a tab on the main page.
    return redirect("/?tab=charts")


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
