"""
app.py  —  Flask web UI for ResUNetCBAMASPP nerve segmentation
Run:  python web_app/app.py
Open: http://127.0.0.1:5000
"""

import sys, os, io, base64

# Project root + model definition (src/) on the import path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from flask import Flask, request, jsonify, render_template

from train_best import ResUNetCBAMASPP, apply_clahe, use_amp

# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH = os.path.join(_ROOT, "models", "best_model_v2.pth")
IMG_SIZE   = 256
THRESHOLD  = 0.5

# =============================================================================
# LOAD MODEL  (once at startup)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ResUNetCBAMASPP().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"Model loaded on {device}")

# =============================================================================
# INFERENCE HELPERS
# =============================================================================
def preprocess(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes → grayscale float32 256×256 with CLAHE."""
    arr  = np.frombuffer(file_bytes, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    return apply_clahe(img)


def predict_tta(img_np: np.ndarray) -> np.ndarray:
    """4-fold TTA → probability map H×W float32."""
    t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

    def fwd(x):
        if use_amp:
            with torch.amp.autocast("cuda"):
                return torch.sigmoid(model(x))
        return torch.sigmoid(model(x))

    with torch.no_grad():
        p  = fwd(t)
        p += fwd(torch.flip(t, [-1])).flip([-1])
        p += fwd(torch.flip(t, [-2])).flip([-2])
        p += fwd(torch.rot90(t, 2, [-2,-1])).rot90(-2, [-2,-1])

    return (p / 4.0).squeeze().cpu().numpy()


def to_base64(img_uint8: np.ndarray, fmt: str = "png") -> str:
    """Encode a numpy H×W or H×W×3 uint8 image to base64 PNG/JPEG string."""
    ok, buf = cv2.imencode(f".{fmt}", img_uint8)
    return base64.b64encode(buf.tobytes()).decode()


def make_heatmap(prob: np.ndarray) -> np.ndarray:
    """Apply 'hot' colormap to a [0,1] probability map → H×W×3 uint8 BGR."""
    colored = (cm.hot(prob)[:, :, :3] * 255).astype(np.uint8)   # RGB
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def make_overlay(img_gray: np.ndarray, pred_bin: np.ndarray,
                 prob: np.ndarray) -> np.ndarray:
    """
    Semi-transparent probability fill (red) + bright green contour on top.
    Returns H×W×3 uint8 BGR.
    """
    base = cv2.cvtColor((img_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # semi-transparent heat fill
    heat  = make_heatmap(prob)
    alpha = np.clip(prob * 0.55, 0, 0.55)[:, :, np.newaxis]
    blend = (base * (1 - alpha) + heat * alpha).astype(np.uint8)

    # solid green contour
    cnts, _ = cv2.findContours(pred_bin.astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blend, cnts, -1, (0, 230, 80), 2)
    return blend


def compute_metrics(pred_bin: np.ndarray,
                    gt: np.ndarray | None) -> dict:
    """Return dice / iou / precision / recall (or None if no GT)."""
    if gt is None:
        return {}
    p, g  = pred_bin.ravel().astype(float), gt.ravel().astype(float)
    tp    = (p * g).sum()
    fp    = (p * (1-g)).sum()
    fn    = ((1-p) * g).sum()
    return {
        "dice"     : round(float((2*tp+1)/(2*tp+fp+fn+1)), 4),
        "iou"      : round(float((tp+1)/(tp+fp+fn+1)),     4),
        "precision": round(float(tp/(tp+fp+1e-6)),         4),
        "recall"   : round(float(tp/(tp+fn+1e-6)),         4),
    }


# =============================================================================
# FLASK APP
# =============================================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit


@app.route("/")
def index():
    return render_template("index.html",
                           device=str(device).upper(),
                           model_name="ResUNetCBAMASPP")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file    = request.files["image"]
    img_bytes = file.read()

    # Optional ground-truth mask upload
    gt = None
    if "mask" in request.files and request.files["mask"].filename:
        mask_bytes = request.files["mask"].read()
        marr = np.frombuffer(mask_bytes, np.uint8)
        mimg = cv2.imdecode(marr, cv2.IMREAD_GRAYSCALE)
        if mimg is not None:
            gt = (cv2.resize(mimg, (IMG_SIZE, IMG_SIZE)) > 127).astype(np.float32)

    try:
        img      = preprocess(img_bytes)
        prob     = predict_tta(img)
        pred_bin = (prob > THRESHOLD).astype(np.float32)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Build result images
    img_u8    = (img * 255).astype(np.uint8)
    heatmap   = make_heatmap(prob)
    overlay   = make_overlay(img, pred_bin, prob)
    pred_u8   = (pred_bin * 255).astype(np.uint8)

    results = {
        "input"    : to_base64(img_u8),
        "heatmap"  : to_base64(heatmap),
        "overlay"  : to_base64(overlay),
        "prediction": to_base64(pred_u8),
        "coverage" : round(float(pred_bin.mean() * 100), 2),
        "threshold": THRESHOLD,
    }

    if gt is not None:
        results["gt"]      = to_base64((gt * 255).astype(np.uint8))
        results["metrics"] = compute_metrics(pred_bin, gt)

    return jsonify(results)


if __name__ == "__main__":
    print(f"\n  Nerve Segmentation Web UI")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Device : {device}")
    print(f"  Open   : http://127.0.0.1:5000\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
