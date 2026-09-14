"""TamperAnalyzer — humble screening signals, never proof of forgery.

Where technically feasible with installed libs (OpenCV/Pillow/numpy only):
  - QR detection + decoding (cv2.QRCodeDetector). Reports three SEPARATE
    facts: detected / readable / data_consistent. Cryptographic authenticity
    is NEVER claimed: signature_valid is always False with an explicit note.
  - ELA-ish recompression response (JPEG resave difference).
  - Noise/sharpness inconsistency across quadrants (copy-paste hint).

Output: {"status": PASS|REVIEW|NOT_AVAILABLE, "score": 0..1, ...}.
A REVIEW here means "needs a human look", never "fake".
"""
import os


def _qr_check(image_path, text):
    """Detect + decode QR. Compare payload vs OCR facts for consistency."""
    out = {"detected": False, "readable": False, "data_consistent": None,
           "signature_valid": False,
           "note": "Cryptographic authenticity not established"}
    try:
        import cv2
    except ImportError:
        out["note"] += " (OpenCV unavailable)"
        return out
    try:
        img = cv2.imread(image_path)
        if img is None:
            return out
        det = cv2.QRCodeDetector()
        data, points, _ = det.detectAndDecode(img)
        if points is not None and len(points) > 0:
            out["detected"] = True
        if data:
            out["readable"] = True
            out["payload_preview"] = data[:120]
            import re
            nums = set(re.findall(r"\d{4,}", data)) & set(re.findall(r"\d{4,}", text))
            names = any(w.lower() in data.lower() for w in
                        re.findall(r"[A-Z][a-z]{3,}", text)[:20]) if text else False
            out["data_consistent"] = bool(nums or names)
    except Exception as e:
        out["note"] += f" (qr error: {e})"
    return out


def _ela_regions(image_path, max_regions=3):
    """ELA hotspot boxes: recompress, threshold the difference map, take
    significant contours. Coordinates normalized 0..1 (viewer-agnostic)."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import io
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((800, 800))
        w, h = img.size
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        buf.seek(0)
        re = Image.open(buf)
        diff = np.abs(np.asarray(img).astype(int) - np.asarray(re).astype(int)).sum(axis=2)
        thr = diff.mean() + 2.5 * diff.std()
        mask = ((diff > max(thr, 12)).astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < 0.002 * w * h or area > 0.5 * w * h:
                continue
            boxes.append((area, {"x": round(x / w, 3), "y": round(y / h, 3),
                                 "width": round(bw / w, 3),
                                 "height": round(bh / h, 3),
                                 "signals": ["compression_mismatch"],
                                 "severity": "MEDIUM",
                                 "label": "compression response differs here"}))
        boxes.sort(reverse=True)
        return [b for _, b in boxes[:max_regions]]
    except Exception:
        return []


def _sharpness_regions(image_path, max_regions=3):
    """4×4 sharpness grid: cells wildly off the median get flagged with
    normalized boxes (copy-paste/local-edit hint, shadows aside)."""
    try:
        import cv2
        import numpy as np
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return []
        h, w = gray.shape
        if min(h, w) < 200:
            return []
        gh, gw = h // 4, w // 4
        vars_ = []
        for i in range(4):
            for j in range(4):
                cell = gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
                vars_.append(cv2.Laplacian(cell, cv2.CV_64F).var() + 1e-6)
        med = float(np.median(vars_))
        out = []
        for idx, v in enumerate(vars_):
            if v < med / 8 or v > med * 8:
                i, j = divmod(idx, 4)
                out.append({"x": round(j / 4, 3), "y": round(i / 4, 3),
                            "width": 0.25, "height": 0.25,
                            "signals": ["local_noise_mismatch"],
                            "severity": "MEDIUM",
                            "label": "sharpness/noise differs from surroundings"})
                if len(out) >= max_regions:
                    break
        return out
    except Exception:
        return []


def _noise_inconsistency(image_path):
    """Max/min sharpness ratio across quadrants. Natural photos ≈ 1-3×;
    pasted regions often diverge. Returns (ratio, flag)."""
    try:
        import cv2
        import numpy as np
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None, False
        h, w = gray.shape
        quads = [gray[:h // 2, :w // 2], gray[:h // 2, w // 2:],
                 gray[h // 2:, :w // 2], gray[h // 2:, w // 2:]]
        vars_ = [cv2.Laplacian(q, cv2.CV_64F).var() + 1e-6 for q in quads]
        ratio = max(vars_) / min(vars_)
        return round(float(ratio), 2), bool(ratio > 8.0)
    except Exception:
        return None, False


def _overlap(a, b):
    """Normalized-box IoU > 0."""
    ax2, ay2, bx2, by2 = (a["x"] + a["width"], a["y"] + a["height"],
                          b["x"] + b["width"], b["y"] + b["height"])
    ix = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    iy = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    return ix > 0.005 and iy > 0.005


def _merge_regions(ela_boxes, sharp_boxes):
    """Overlapping boxes from independent methods merge — and overlapping
    independent signals escalate severity (§6)."""
    regions = list(ela_boxes)
    for s in sharp_boxes:
        hit = next((r for r in regions if _overlap(r, s)), None)
        if hit:
            for sig in s["signals"]:
                if sig not in hit["signals"]:
                    hit["signals"].append(sig)
            hit["severity"] = "HIGH" if len(hit["signals"]) >= 2 else "MEDIUM"
            hit["label"] = ("multiple independent signals overlap here — "
                            "needs a human look")
        else:
            regions.append(s)
    return regions[:4]


def analyze(image_path, text, is_digital_pdf=False):
    """Run all feasible tamper screens. Returns evidence dict."""
    if is_digital_pdf or not image_path or not os.path.exists(image_path):
        return {"status": "NOT_AVAILABLE", "score": None, "regions": [],
                "note": "Pixel forensics need the source photo (N/A for digital PDFs)"}
    qr = _qr_check(image_path, text or "")
    regions = _merge_regions(_ela_regions(image_path),
                             _sharpness_regions(image_path))

    flags = []
    for r in regions:
        where = f"region x={r['x']} y={r['y']} w={r['width']} h={r['height']}"
        flags.append(f"{where}: {', '.join(r['signals'])}")
    if qr["detected"] and not qr["readable"]:
        flags.append("QR-like region detected but unreadable")

    score = None
    if regions:
        score = round(1.0 - 0.25 * sum(len(r["signals"]) for r in regions), 3)
        score = max(0.0, score)
    status = "REVIEW" if flags else "PASS"
    return {"status": status, "score": score, "flags": flags,
            "regions": regions, "qr": qr,
            "note": "Screening signals only — none of this proves forgery. "
                    "Regions are 'suspicious, needs review', never 'forged'." if flags
                    else "No tamper indicators found (screening only, not proof)"}
