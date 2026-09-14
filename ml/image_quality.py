"""ImageQualityChecker — independent gate before any verification claim.

Poor OCR from a blurry/dark photo must NEVER be treated as evidence of
fakery. If the image is insufficient for reliable analysis the pipeline
returns NEEDS_REVIEW with "Insufficient image quality...".

Metrics (OpenCV + numpy only, no new dependencies):
  resolution / blur (Laplacian variance) / brightness / contrast / glare.
Output: {"overall": 0..1, <metric>: 0..1, "status": PASS|REVIEW|FAIL, ...}
Thresholds come from config/documentVerification.json.
"""
import os
import json

_CONFIG = None


def _config():
    global _CONFIG
    if _CONFIG is None:
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "config", "documentVerification.json")
        with open(here) as f:
            _CONFIG = json.load(f)["image_quality"]
    return _CONFIG


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def check_image_quality(image_path):
    """Score a photo file. Returns dict with overall + per-metric scores."""
    cfg = _config()
    w = cfg["weights"]
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"overall": 0.5, "status": "REVIEW",
                "note": "OpenCV unavailable — quality gate skipped",
                "resolution": 0.5, "blur": 0.5, "brightness": 0.5,
                "contrast": 0.5, "glare": 0.5}
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"overall": 0.0, "status": "FAIL",
                "note": "Unreadable image file",
                "resolution": 0.0, "blur": 0.0, "brightness": 0.0,
                "contrast": 0.0, "glare": 0.0}
    h, w_px = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resolution: longest side 1800px+ ideal (matches OCR upscale target).
    resolution = _clamp(max(h, w_px) / 1800.0)

    # Blur: variance of Laplacian. Sharp document photos sit well above ~300;
    # heavy blur collapses toward 0. Calibrated floor, not a magic proof.
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur = _clamp(blur_var / 300.0)

    # Brightness: mean gray; documents are bright paper, so the ideal band
    # leans bright (~110–210). Dark scans fade below; only near-blown
    # means wash out (flash glare is measured separately).
    mean = float(gray.mean())
    if mean < 110:
        brightness = _clamp(mean / 110.0)
    elif mean > 210:
        brightness = _clamp(1.0 - (mean - 210.0) / 90.0)
    else:
        brightness = 1.0

    # Contrast: std-dev; healthy document photos spread well across grays.
    contrast = _clamp(float(gray.std()) / 50.0)

    # Glare: near-blown pixels (>245). But a clean white scan is ALSO mostly
    # >245 — that is background, not flash. Flash glare coexists with darker
    # surroundings (high contrast + local blown blob), so a uniformly bright
    # page is explicitly rescued.
    blown = float((gray > 245).mean())
    if blown > 0.5 and contrast < 0.5:
        glare = 0.9  # plain white background, not a flash hotspot
    else:
        glare = _clamp(1.0 - blown / 0.08)

    overall = round(w["resolution"] * resolution + w["blur"] * blur
                    + w["brightness"] * brightness + w["contrast"] * contrast
                    + w["glare"] * glare, 3)
    if overall >= cfg["pass"]:
        status = "PASS"
    elif overall >= cfg["review"]:
        status = "REVIEW"
    else:
        status = "FAIL"
    return {"overall": overall, "status": status,
            "resolution": round(resolution, 3), "blur": round(blur, 3),
            "blur_variance": round(float(blur_var), 1),
            "brightness": round(brightness, 3), "mean_gray": round(mean, 1),
            "contrast": round(contrast, 3),
            "glare": round(glare, 3), "blown_fraction": round(blown, 4)}


def digital_pdf_quality():
    """Digital PDFs carry a text layer — no photo to grade. Neutral."""
    return {"overall": 1.0, "status": "N/A",
            "note": "digital PDF text layer — no photo quality to assess",
            "resolution": 1.0, "blur": 1.0, "brightness": 1.0,
            "contrast": 1.0, "glare": 1.0}
