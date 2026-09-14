"""
Document Verification ML Service
================================
College project — runs locally, no deployment.

Pipeline:
  1. Preprocess image (OpenCV) — grayscale, denoise, adaptive threshold
  2. OCR with PaddleOCR (primary) — returns text + confidence
  3. Syntax verification — regex / format checks per document type
  4. Semantic verification — keyword coverage (TF-IDF cosine via scikit-learn),
     logical consistency (dates, age, ID checksums), text coherence

Usage:
  python3 verify_document.py <file_path> <document_type>

  document_type: aadhaar | pan | passport | marksheet | generic

Output:
  Single JSON object printed to stdout (last line). Node.js backend parses it.
"""

import sys
import os
import re
import json
from datetime import datetime


# ---------------------------------------------------------------------------
# Document templates: required keywords + syntax patterns
# ---------------------------------------------------------------------------

# Document templates live in doc_templates.py (single source of truth shared
# with classifier.py and verifiers.py — no duplicated verification logic).
from doc_templates import TEMPLATES, SPECIFIC_TYPES

GENERIC_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
GENERIC_DATE = re.compile(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})")


# ---------------------------------------------------------------------------
# 1. Preprocessing + OCR
# ---------------------------------------------------------------------------

def deskew(gray):
    """Straighten rotated scans. Returns deskewed image (no-op if angle tiny)."""
    import cv2
    import numpy as np
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5 or abs(angle) > 20:  # huge angles = probably a photo, don't touch
        return gray
    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(image_path):
    """Photo → clean B/W page: EXIF-orientation, denoise, deskew, contrast
    (CLAHE), upscale, adaptive threshold, white border. Returns temp path."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return image_path  # OpenCV optional — skip preprocessing
    # PIL honors EXIF orientation (cv2.imread silently ignores it, so phone
    # portraits would arrive sideways without this).
    try:
        from PIL import Image, ImageOps
        pil = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return image_path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1. Denoise (keeps edges, kills sensor noise)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # 2. Deskew rotated scans
    try:
        gray = deskew(gray)
    except Exception:
        pass
    # 3. Contrast: CLAHE fixes shadows / yellow indoor light
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass
    # 4. Upscale: OCR engines want tall characters (x-height ≥ ~30px)
    h, w = gray.shape
    if max(h, w) < 1800:
        scale = 1800 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # 5. Binarize + slight sharpen
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 41, 12)
    # 6. White border: characters touching the edge get missed otherwise
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20,
                              cv2.BORDER_CONSTANT, value=255)
    tmp = image_path + "_pre.png"
    cv2.imwrite(tmp, gray)
    return tmp


# ---------------------------------------------------------------------------
# 1b. OCR output cleanup (fixes complaints #2: misspellings)
# ---------------------------------------------------------------------------

# Vocabulary the corrector may snap words to: template keywords + common labels.
CORRECTOR_VOCAB = sorted({
    "government", "india", "unique", "identification", "authority", "uidai",
    "aadhaar", "card", "proof", "identity", "citizenship", "verification",
    "authentication", "verify", "scanning", "scan", "offline", "code",
    "masked", "resident", "download", "update", "valid",
    "passport", "republic", "income", "tax", "department",
    "permanent", "account", "number", "certificate", "board", "central",
    "secondary", "university", "examination", "examinations", "cbse",
    "school", "college", "marks", "statement", "cum", "grade", "subject",
    "theory", "practical", "additional", "scholastic", "achievements",
    "achieved", "obtained", "result", "controller", "roll", "registration",
    "enrolment", "session", "division", "course", "standard", "school",
    "mother", "father", "guardian", "birth", "date", "gender", "male",
    "female", "transgender", "address", "nationality", "indian", "expiry",
    "issue", "place", "signature", "authorized",
    "year", "month", "village", "district", "state", "road", "street", "nagar",
    "mathematics", "science", "english", "hindi", "social", "sanskrit",
    "physics", "chemistry", "biology", "computer", "music", "literature",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "sharma", "verma", "gupta", "singh", "kumar", "patel", "reddy", "khan",
})


def fix_id_confusions(text):
    """OCR mixes 0/O, 1/I/l, 5/S, 8/B inside numbers — normalize digit runs."""
    import re
    def fix_run(m):
        run = m.group(0)
        if sum(c.isdigit() for c in run) < len(run) / 2:
            return run  # mostly letters — leave alone
        return (run.replace("O", "0").replace("o", "0").replace("D", "0")
                   .replace("I", "1").replace("l", "1").replace("|", "1")
                   .replace("S", "5").replace("B", "8").replace("A", "4"))
    return re.sub(r"[0-9OoDIlaAISB|]{4,}", fix_run, text)


def correct_ocr_text(text):
    """Lightweight spell-fix using only stdlib difflib.

    - Only touches pure-letter words ≥4 chars (never names-in-particular?
      no — names ARE corrected only if very close to vocab, cutoff 0.88).
    - Never touches tokens containing digits (IDs, dates stay byte-exact).
    """
    import re
    import difflib
    if not text:
        return text
    # Stray apostrophes from OCR ("OB'TAINED", "Muthet'") block matching.
    text = re.sub(r"(?<=[A-Za-z])['’`\"](?=[A-Za-z])", "", text)
    # Stray @ glued to words ("THi@ MARKS").
    text = re.sub(r"(?<=[A-Za-z])@+", "", text)
    # Trailing stray apostrophe ("Mother' Name").
    text = re.sub(r"(?<=[A-Za-z])['’`\"](?=\s|$)", "", text)

    def fix_word(m):
        w = m.group(0)
        if len(w) < 4 or not w.isalpha():
            return w
        low = w.lower()
        if low in CORRECTOR_VOCAB:
            return w
        hit = difflib.get_close_matches(low, CORRECTOR_VOCAB, n=1, cutoff=0.80)
        if hit:
            fixed = hit[0]
            if w[0].isupper():
                fixed = fixed.capitalize()
            return fixed
        return w

    text = re.sub(r"[A-Za-z]{4,}", fix_word, text)
    text = fix_id_confusions(text)
    # Collapse spaced-out digits "2 3 4 5" that OCR splits ("2 345" kept as-is)
    text = re.sub(r"(?<=\d) (?=\d)", "", text)
    return text


def canonicalize_labels(text):
    """Rewrite OCR-mangled *labels* to their canonical form for display.

    Only touches near-certain label mangles (word boundaries, often with
    surrounding context), never numbers/dates/names. Applied to the text
    used for checks AND shown in the frontend, so users see "Roll No."
    instead of "Rull Na.". Matching patterns keep their own OCR tolerance
    as well — this is purely about correct display.
    """
    import re
    if not text:
        return text
    rules = [
        (r"\bRull\b", "Roll"), (r"\bRell\b", "Roll"),
        (r"\bR0ll\b", "Roll"), (r"\bRoII\b", "Roll"),
        (r"\bRgn\b", "Reg"),
        (r"\bD0B\b", "DOB"),
        (r"\bGennder\b", "Gender"), (r"\bGendar\b", "Gender"),
        (r"\bAdhsar\b", "Aadhaar"), (r"\bAadhar\b", "Aadhaar"),
        (r"\bAdhar\b", "Aadhaar"),
        (r"\bMuthet\b", "Mother"),
        (r"\blng\b", "Lang"),
        (r"\bTHi\b", "The"),
        (r"\bNa\.", "No."),                       # "Rull Na." → "Roll No."
        (r"\bCUN\b(?=\s+CERTIFICATE)", "CUM"),     # "MARKS CUM CERTIFICATE"
    ]
    for pat, repl in rules:
        text = re.sub(pat, repl, text)
    return text


def coherence_of(text):
    """0..1 score: how much does this look like real words (for engine voting)."""
    import re
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    if not words:
        return 0.0
    alpha = sum(1 for w in words if w.isalpha()) / len(words)
    avg_len = sum(len(w) for w in words) / len(words)
    vocab_hit = sum(1 for w in words if w in CORRECTOR_VOCAB) / len(words)
    score = 0.5 * alpha + 0.2 * (1.0 if 3.5 <= avg_len <= 9 else 0.3) \
        + 0.3 * min(vocab_hit * 4, 1.0)
    return round(score, 3)


def extract_text_pdf(pdf_path):
    """Try embedded-text extraction first (digital PDFs), else render page 1."""
    # Attempt 1: PyMuPDF (best — also renders scanned pages)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        parts = [page.get_text() for page in doc]
        text = "\n".join(parts).strip()
        if len(text) > 50:
            return text, 0.95, "pymupdf-text"
        # Scanned PDF — render first page to PNG for OCR
        pix = doc[0].get_pixmap(dpi=200)
        img_path = pdf_path + "_page1.png"
        pix.save(img_path)
        return img_path, 0.0, "render-for-ocr"
    except ImportError:
        pass  # fall through to pypdf
    except Exception:
        pass  # fall through to pypdf
    # Attempt 2: pypdf (lightweight, pure python — digital PDFs only)
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        parts = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(parts).strip()
        if len(text) > 20:
            return text, 0.90, "pypdf-text"
        return "", 0.0, "pdf-error: scanned PDF needs PyMuPDF+PaddleOCR (no text layer found)"
    except Exception as e:
        return "", 0.0, f"pdf-error: {e}"


def run_tesseract(image_path):
    """Tesseract with document-tuned config. Tries PSM 6 (uniform block),
    falls back to PSM 3 (fully automatic) if output is garbage."""
    import pytesseract
    from PIL import Image
    img = Image.open(image_path)
    base_cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    last = None
    for cfg in (base_cfg, "--oem 1 --psm 3"):
        try:
            data = pytesseract.image_to_data(img, config=cfg,
                                             output_type=pytesseract.Output.DICT)
        except Exception as e:
            last = e
            continue
        words, confs = [], []
        for w, c in zip(data.get("text", []), data.get("conf", [])):
            try:
                c = float(c)
            except (TypeError, ValueError):
                continue
            if w and w.strip() and c >= 0:
                words.append(w)
                confs.append(c / 100.0)
        full_text = pytesseract.image_to_string(img, config=cfg)
        if full_text and full_text.strip():
            avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.6
            boxes = [{"text": w, "confidence": round(c, 3)}
                     for w, c in zip(words[:50], confs[:50])]
            # PSM 6 result good enough? keep it, else try PSM 3 below.
            if coherence_of(full_text) >= 0.45 or cfg.endswith("psm 3"):
                return full_text, avg_conf, boxes
    raise RuntimeError(f"tesseract produced no usable text ({last})")


def run_rapidocr(image_path):
    """Fallback OCR via RapidOCR-ONNX (pip only, no system binary needed).
    Returns (full_text, avg_confidence, line_details)."""
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    result, _ = ocr(image_path)
    lines, scores, boxes = [], [], []
    if result:
        for item in result:
            # item: [box_points, text, confidence]
            try:
                txt, conf = item[1], float(item[2])
            except (IndexError, TypeError, ValueError):
                continue
            lines.append(txt)
            scores.append(conf)
            boxes.append({"text": txt, "confidence": round(conf, 3)})
    full_text = "\n".join(lines)
    avg_conf = round(sum(scores) / len(scores), 3) if scores else 0.0
    return full_text, avg_conf, boxes


def run_paddleocr(image_path):
    """Returns (full_text, avg_confidence, line_details)."""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = ocr.ocr(image_path, cls=True)
    lines, scores, boxes = [], [], []
    if result and result[0]:
        for line in result[0]:
            box, (txt, conf) = line[0], line[1]
            lines.append(txt)
            scores.append(float(conf))
            boxes.append({"text": txt, "confidence": round(float(conf), 3)})
    full_text = "\n".join(lines)
    avg_conf = round(sum(scores) / len(scores), 3) if scores else 0.0
    return full_text, avg_conf, boxes


def do_ocr(file_path):
    """Unified OCR entry. Returns dict with text, confidence, engine."""
    ext = os.path.splitext(file_path)[1].lower()
    engine = "paddleocr"

    if ext == ".pdf":
        out, conf, how = extract_text_pdf(file_path)
        if how in ("pymupdf-text", "pypdf-text"):
            return {"text": out, "confidence": conf, "engine": f"{how.split('-')[0]} (digital pdf)"}
        if how.startswith("pdf-error"):
            return {"text": "", "confidence": 0.0, "engine": how}
        file_path = out  # rendered image → continue to OCR

    processed = preprocess_image(file_path)
    temp_files = [processed] if processed != file_path else []
    cleanup = lambda: [os.remove(p) for p in temp_files if os.path.exists(p)]
    try:
        # Orientation invariance: EXIF is already honored in preprocessing,
        # but scans/photos can still be sideways or upside-down. Try 0°,
        # then 90/180/270 via lossless cv2 rotates, and let the coherence
        # vote pick the winner. Early exit the moment a read is clearly good
        # (common case stays at exactly one OCR pass per engine).
        candidates = []  # (text, conf, boxes, "engine+rot")
        errors = []
        orientations = [(processed, 0)]
        try:
            import cv2
            base = cv2.imread(processed, cv2.IMREAD_COLOR)
            if base is not None:
                codes = [(cv2.ROTATE_90_CLOCKWISE, 90),
                         (cv2.ROTATE_180, 180),
                         (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)]
                for code, deg in codes:
                    rp = f"{processed}_r{deg}.png"
                    if cv2.imwrite(rp, cv2.rotate(base, code)):
                        orientations.append((rp, deg))
                        temp_files.append(rp)
        except Exception as e:
            errors.append(f"rotate [{e}]")

        for img_path, deg in orientations:
            for name, fn in (("paddleocr", run_paddleocr),
                             ("rapidocr", run_rapidocr),
                             ("tesseract", run_tesseract)):
                try:
                    t, c, b = fn(img_path)
                    if t and t.strip():
                        tag = name if deg == 0 else f"{name}+rot{deg}"
                        candidates.append((t, c, b, tag))
                except Exception as e:
                    if deg == 0:
                        errors.append(f"{name} [{e}]")
            # Early exit: a clearly good read needs no further rotations.
            if candidates and max(coherence_of(t) for t, _, _, _ in candidates) >= 0.6:
                break

        if not candidates:
            return {"text": "", "confidence": 0.0,
                    "engine": "ocr-unavailable: " + " | ".join(errors) + ". "
                              "FIX: (a) digital PDFs work without OCR — try one; "
                              "(b) for photos run: pip install rapidocr-onnxruntime "
                              "(pip-only, easiest) OR brew install tesseract + pip install pytesseract "
                              "OR pip install -r ml/requirements.txt (full PaddleOCR).",
                    "lines": []}

        # Vote: highest coherence wins, engine confidence breaks ties.
        # Spell-fix + ID-fix applied to the winner only (complaint #2).
        ranked = sorted(candidates,
                        key=lambda cb: (coherence_of(cb[0]), cb[1]),
                        reverse=True)
        raw_text, conf, boxes, winner = ranked[0]
        engine_used = winner + (" (best of %d)" % len(candidates)
                                if len(candidates) > 1 else "")
        # MRZ union: the coherence vote favors dictionary words, but MRZ is
        # non-dictionary BY DESIGN — a rotated read holding the MRZ would
        # otherwise lose the vote and take the MRZ down with it. Scan every
        # candidate so machine-readable evidence always survives.
        mrz_lines, mrz_hint = [], ""
        try:
            from cross_field import find_mrz_lines, detect_mrz
            for t, _, _, _ in candidates:
                for pair in find_mrz_lines(t.upper().replace(" ", "")):
                    if pair not in mrz_lines:
                        mrz_lines.append(pair)
            for t, _, _, _ in sorted(candidates,
                                     key=lambda cb: (coherence_of(cb[0]), cb[1]),
                                     reverse=True):
                blob = detect_mrz(t)
                if blob and len(blob) > len(mrz_hint):
                    mrz_hint = blob
        except Exception:
            pass
        text = canonicalize_labels(correct_ocr_text(raw_text))
        return {"text": text, "confidence": conf, "engine": engine_used,
                "lines": boxes[:50],
                "enginesTried": [c[3] for c in candidates],
                "mrz_lines": mrz_lines[:2], "mrz_hint": mrz_hint}
    except Exception as e:
        return {"text": "", "confidence": 0.0, "engine": f"ocr-error: {e}", "lines": []}
    finally:
        try:
            cleanup()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 2. Syntax verification (format / regex)
# ---------------------------------------------------------------------------

def verhoeff_suggest(num_str):
    """Single-digit OCR-error correction for 12-digit numbers.

    Tries likely OCR confusions first (1↔7, 3↔8, 4↔9, 6↔8 …), then every
    digit exhaustively. Returns (corrected, position, old, new) or None.
    Real Aadhaar numbers with one misread digit self-heal; random 12-digit
    strings almost never do (1-in-10 chance per position), so this stays
    honest: it can't launder a fake number into a valid one.
    """
    confusions = {"0": "68", "1": "7", "2": "7", "3": "8", "4": "9",
                  "5": "6", "6": "058", "7": "1", "8": "3609", "9": "48"}
    tried = set()

    def attempt(i, new_ch):
        if new_ch == num_str[i] or (i, new_ch) in tried:
            return None
        tried.add((i, new_ch))
        cand = num_str[:i] + new_ch + num_str[i + 1:]
        if verhoeff_check(cand):
            return (cand, i, num_str[i], new_ch)
        return None

    for i, ch in enumerate(num_str):  # confusion-first (most likely first)
        for new_ch in confusions.get(ch, ""):
            hit = attempt(i, new_ch)
            if hit:
                return hit
    for i, ch in enumerate(num_str):  # exhaustive fallback
        for new_ch in "0123456789":
            hit = attempt(i, new_ch)
            if hit:
                return hit
    return None


def verhoeff_check(num_str):
    """Verhoeff checksum — real Aadhaar numbers pass this. Used as a signal."""
    d = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
         [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
         [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
         [9,8,7,6,5,4,3,2,1,0]]
    p = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
         [8,9,1,6,0,4,3,7,2,5],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
         [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]
    c = 0
    for i, ch in enumerate(reversed(num_str)):
        c = d[c][p[i % 8][int(ch)]]
    return c == 0


def _display_value(name, raw):
    """Clean check value for the frontend: collapse whitespace and mask
    sensitive identifiers (Aadhaar/VID/PAN/passport/mobile) middle-out."""
    val = " ".join(str(raw).split())[:60]
    if name in ("aadhaar_number", "vid", "pan_number", "passport_no",
                "dl_no", "mobile", "roll_no_format"):
        from number_checks import mask_id
        # Mask the identifier chunk (spaces ignored): 7260 1815 9081 → 72••••••••81.
        import re as _re
        m = _re.search(r"(?:[A-Z0-9][ ]?){8,}", val)
        if m:
            flat = m.group(0).replace(" ", "")
            val = val.replace(m.group(0), mask_id(flat))
    return val


def syntax_checks(text, doc_type):
    """Regex checks → list of {check, status, message, value} + syntax_score."""
    tpl = TEMPLATES.get(doc_type, TEMPLATES["generic"])
    checks = []
    passed = 0
    total = 0

    # Alias patterns: alternate evidence that rescues a primary check.
    # e.g. OCR shreds "Roll No" label but the 8-digit CBSE number survives.
    alias_of = {"roll_no_format": "roll_no", "grade_token": "grade",
                "subject_marks": "marks", "marks_words": "marks",
                "mrz": "passport_no"}
    alias_hits = {}

    for name, pat in tpl["patterns"].items():
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if not m:
            # Generic is a catch-all: absent optional patterns are simply
            # irrelevant (no email on a marksheet!), so don't list them —
            # listing them is what produced the confusing "WARN email" noise.
            if doc_type == "generic" and name not in tpl.get("required", []):
                continue
            if name in alias_of:
                continue  # backup evidence: silence when absent
            total += 1
            is_required = name in tpl.get("required", [])
            checks.append({"check": f"syntax:{name}", "status": "fail" if is_required else "warn",
                           "message": f"{'Missing required' if is_required else 'Not found (optional)'}: {name.replace('_',' ')}",
                           "value": None})
            continue
        if name in alias_of:
            # Backup evidence: don't score alone, may rescue the primary below.
            alias_hits[alias_of[name]] = (name, " ".join(m.group(0).split())[:60])
            continue
        total += 1
        extra = ""
        status = "pass"
        if name == "aadhaar_number":
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) == 12 and digits[0] != "0" and digits[0] != "1":
                if verhoeff_check(digits):
                    extra = " (Verhoeff checksum valid)"
                else:
                    # One misread digit is the norm in OCR, not fraud:
                    # try to self-heal a single-digit error.
                    fix = verhoeff_suggest(digits)
                    if fix:
                        _, pos, old_ch, new_ch = fix
                        extra = (f" (valid after allowing one likely OCR misread "
                                 f"at digit {pos + 1}: {old_ch}→{new_ch})")
                    else:
                        status = "warn"
                        extra = (" (checksum doesn't validate — most likely an "
                                 "OCR digit error; please verify the number visually)")
            else:
                status = "warn"
                extra = " (must be 12 digits, not starting with 0/1)"
        if name == "pan_number":
            pan = m.group(0).upper()
            # 4th char = holder type (P=individual, C=company, etc.)
            if pan[3] not in "PCHABGJLFTT":
                status = "warn"
                extra = " (4th character should be holder type, e.g. P)"
        checks.append({"check": f"syntax:{name}", "status": status,
                       "message": f"Found {name.replace('_',' ')}{extra}",
                       "value": _display_value(name, m.group(0))})
        if status == "pass":
            passed += 1
        else:
            passed += 0.4
    # Rescue: primary failed/warned but alternate OCR-proof evidence passed.
    for c in checks:
        base = c["check"].split(":", 1)[1]
        if base in alias_hits and c["status"] in ("fail", "warn"):
            old = c["status"]
            alias_name, alias_val = alias_hits[base]
            c["status"] = "pass"
            c["message"] = (f"Found {base.replace('_',' ')} "
                            f"(via {alias_name.replace('_',' ')}: {alias_val})")
            c["value"] = c["value"] or alias_val
            passed += 1.0 if old == "fail" else 0.6  # warn already earned 0.4
    # Generic quality signals
    total += 1
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) >= 10:
        checks.append({"check": "syntax:text_volume", "status": "pass",
                       "message": f"Readable text extracted ({len(words)} words)", "value": len(words)})
        passed += 1
    else:
        checks.append({"check": "syntax:text_volume", "status": "fail",
                       "message": "Very little readable text — image may be blurry/dark",
                       "value": len(words)})

    syntax_score = round(passed / total, 3) if total else 0.0
    return checks, syntax_score


def _corpus_similarities(text, types):
    """Delegates to classifier.similarities — one implementation only."""
    from classifier import similarities
    return similarities(text, types)


def template_similarity(text, tpl_name_or_tpl):
    """TF-IDF cosine between one template and document text (0..1)."""
    name = tpl_name_or_tpl if isinstance(tpl_name_or_tpl, str) \
        else next((t for t in TEMPLATES if TEMPLATES[t] is tpl_name_or_tpl),
                  "generic")
    return _corpus_similarities(text, list(TEMPLATES))[name]


def detect_document_type(text):
    """Score text against every template → (best_type, {type: score}).

    Generic is the fallback: it only wins when the classifier is not
    confident, so a marksheet is never again called 'Generic'."""
    from classifier import classify
    result = classify(text)
    sims = result["raw_similarities"]
    if not result["confident"]:
        return "generic", sims
    return result["predicted"], sims


# ---------------------------------------------------------------------------
# 3. Semantic verification (meaning / consistency)
# ---------------------------------------------------------------------------

def semantic_checks(text, doc_type, ocr_conf, is_best_fit=False):
    """Keyword coverage (TF-IDF) + logic consistency + coherence."""
    tpl = TEMPLATES.get(doc_type, TEMPLATES["generic"])
    checks = []
    score_parts = []

    # --- 3a. Keyword / template similarity via TF-IDF cosine ---
    sim = template_similarity(text, tpl)
    # Generic is a catch-all bucket, so its bar is lower — and it can never
    # FAIL: "we don't know what this is" is not a failure, just neutral.
    pass_at, warn_at = (0.08, 0.03) if doc_type == "generic" else (0.15, 0.05)
    if sim >= pass_at:
        score_parts.append(sim)
        checks.append({"check": "semantic:template_match", "status": "pass",
                       "message": f"Document reads like a {tpl['label']} (similarity {sim:.2f})",
                       "value": round(sim, 3)})
    elif sim >= warn_at:
        score_parts.append(sim)
        msg = (f"Partial match to {tpl['label']} template (similarity {sim:.2f}) — "
               f"this is still the best-fit type, OCR noise is diluting the wording"
               if is_best_fit else
               f"Weak match to {tpl['label']} template (similarity {sim:.2f}) — wrong doc type?")
        checks.append({"check": "semantic:template_match", "status": "warn",
                       "message": msg,
                       "value": round(sim, 3)})
    elif doc_type == "generic":
        score_parts.append(warn_at)  # neutral credit, not zero
        checks.append({"check": "semantic:template_match", "status": "warn",
                       "message": f"No template resemblance (similarity {sim:.2f}) — "
                                  f"uncharacterized document, neutral (not a failure)",
                       "value": round(sim, 3)})
    else:
        score_parts.append(sim)
        checks.append({"check": "semantic:template_match", "status": "fail",
                       "message": f"Text does not resemble a {tpl['label']} (similarity {sim:.2f})",
                       "value": round(sim, 3)})

    # --- 3b. Required-field semantics: does the doc actually identify someone? ---
    low = text.lower()
    has_name_like = bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text))
    has_id_num = bool(re.search(r"\d{4,}", text))
    has_date = bool(GENERIC_DATE.search(text))
    present = sum([has_name_like, has_id_num, has_date])
    field_score = present / 3
    score_parts.append(field_score)
    checks.append({"check": "semantic:entity_completeness", "status": "pass" if present >= 2 else ("warn" if present == 1 else "fail"),
                   "message": f"Identity signals: name={'yes' if has_name_like else 'no'}, id-number={'yes' if has_id_num else 'no'}, date={'yes' if has_date else 'no'}",
                   "value": f"{present}/3"})

    # --- 3c. Date logic: no future DOB, expiry after issue ---
    date_score = 0.5  # neutral default
    dates = []
    for m in GENERIC_DATE.finditer(text):
        try:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y += 2000 if y < 50 else 1900
            dates.append(datetime(y, mo, d))
        except ValueError:
            continue
    now = datetime.now()
    if dates:
        future = [d for d in dates if d > now]
        if future:
            checks.append({"check": "semantic:date_logic", "status": "fail",
                           "message": f"{len(future)} date(s) lie in the future — suspicious",
                           "value": [d.strftime("%d/%m/%Y") for d in future][:3]})
            date_score = 0.0
        else:
            checks.append({"check": "semantic:date_logic", "status": "pass",
                           "message": f"All {len(dates)} date(s) are plausible (none in future)",
                           "value": [d.strftime("%d/%m/%Y") for d in dates][:3]})
            date_score = 1.0
    else:
        checks.append({"check": "semantic:date_logic", "status": "warn",
                       "message": "No dates found to validate", "value": None})
    score_parts.append(date_score)

    # --- 3d. Coherence: ratio of dictionary-like words (OCR garbage detection) ---
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    if words:
        alpha_ratio = sum(1 for w in words if re.fullmatch(r"[a-z]+", w)) / len(words)
        avg_len = sum(len(w) for w in words) / len(words)
        coherent = alpha_ratio > 0.7 and 3.5 <= avg_len <= 9
        coh_score = 1.0 if coherent else 0.3
        checks.append({"check": "semantic:coherence", "status": "pass" if coherent else "warn",
                       "message": f"Text coherence {'looks natural' if coherent else 'looks garbled — retake photo'} (alpha-ratio {alpha_ratio:.2f})",
                       "value": round(alpha_ratio, 3)})
    else:
        coh_score = 0.0
        checks.append({"check": "semantic:coherence", "status": "fail",
                       "message": "No readable words — cannot assess meaning", "value": 0})
    score_parts.append(coh_score)

    semantic_score = round(sum(score_parts) / len(score_parts), 3)
    return checks, semantic_score


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Legacy score (kept for the transition period; the pipeline now decides via
# EvidenceAggregator + DecisionEngine in evidence.py).
# ---------------------------------------------------------------------------

def final_verdict(ocr_conf, syntax_score, semantic_score):
    overall = round(0.35 * ocr_conf + 0.35 * syntax_score + 0.30 * semantic_score, 3)
    if overall >= 0.70 and ocr_conf >= 0.5:
        verdict = "VERIFIED"
    elif overall >= 0.40:
        verdict = "NEEDS_REVIEW"
    else:
        verdict = "REJECTED"
    return verdict, overall


def _ocr_evidence(ocr_conf, has_text):
    """OCR evidence: readability only — NEVER fraud evidence (§5)."""
    import json as _j, os as _o  # local to avoid touching module imports
    cfg_path = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)),
                            "config", "documentVerification.json")
    try:
        with open(cfg_path) as f:
            th = _j.load(f)["ocr"]
    except Exception:
        th = {"pass": 0.60, "review": 0.35}
    if not has_text:
        return {"status": "FAIL", "score": 0.0,
                "note": "No text extracted — readability failure, not fraud evidence"}
    if ocr_conf >= th["pass"]:
        return {"status": "PASS", "score": ocr_conf, "note": "Text readable"}
    if ocr_conf >= th["review"]:
        return {"status": "REVIEW", "score": ocr_conf,
                "note": "Text partially readable"}
    return {"status": "FAIL", "score": ocr_conf,
            "note": "Text barely readable — readability failure, not fraud evidence"}


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: verify_document.py <file_path> <document_type>"}))
        return
    file_path, requested = sys.argv[1], sys.argv[2].lower()
    if requested not in list(TEMPLATES) + ["auto"]:
        requested = "auto"
    if not os.path.exists(file_path):
        print(json.dumps({"error": f"File not found: {file_path}"}))
        return

    from image_quality import check_image_quality, digital_pdf_quality
    from classifier import classify
    from verifiers import get_verifier
    from cross_field import validate as cross_validate
    from tamper import analyze as tamper_analyze
    from number_checks import scan_text as fab_scan
    from evidence import aggregate, decide

    is_pdf = os.path.splitext(file_path)[1].lower() == ".pdf"

    # ---- Layer 1: OCR (needed before quality-gating the pixels) ----
    ocr = do_ocr(file_path)
    text = ocr.get("text", "")
    ocr_conf = float(ocr.get("confidence", 0.0) or 0.0)
    engine = ocr.get("engine", "unknown")
    is_digital = is_pdf and "digital pdf" in engine

    # ---- Layer 2: Image Quality Gate (poor image ≠ fake, §3) ----
    if is_digital:
        quality = digital_pdf_quality()
        photo_path = None
    else:
        photo_path = file_path
        quality = check_image_quality(file_path)

    if not text.strip():
        # No text at all: image too poor → NEEDS_REVIEW, never SUSPICIOUS.
        if quality.get("status") == "FAIL":
            decision, reasons = "NEEDS_REVIEW", [
                "Insufficient image quality for reliable verification — retake the photo."]
        else:
            decision, reasons = "NEEDS_REVIEW", [
                f"OCR could not read any text ({engine}). "
                "Try better lighting / higher resolution."]
        print(json.dumps({
            "documentType": requested if requested != "auto" else "generic",
            "verdict": decision, "decision": decision, "reasons": reasons,
            "overallScore": 0.0, "ocrConfidence": 0.0,
            "syntaxScore": 0.0, "semanticScore": 0.0,
            "engine": engine, "extractedText": "",
            "image_quality": quality,
            "document_classification": {"predicted": "generic",
                                        "confidence": 0.0, "confident": False},
            "evidence": {"ocr": _ocr_evidence(0.0, False)},
            "checks": [{"check": "ocr:readable", "status": "fail",
                        "message": reasons[0], "value": None}],
        }))
        return

    # ---- Layer 3: Document Classification (probabilistic, §4) ----
    # The MRZ union rides along for classification only: it proves type,
    # but it is machine text, not prose — checks still run on human text.
    cls_input = text + ("\n" + ocr.get("mrz_hint", "") if ocr.get("mrz_hint") else "")
    cls = classify(cls_input)
    detected = cls["predicted"] if cls["confident"] else "generic"
    if requested == "auto":
        doc_type, auto_used, cls_auto = detected, True, True
    else:
        doc_type, auto_used, cls_auto = requested, False, False
    classification = {"predicted": doc_type,
                      "confidence": cls["confidence"] if doc_type == cls.get("best_raw") else cls["confidence"],
                      "alternatives": cls["alternatives"],
                      "raw_similarities": cls["raw_similarities"],
                      "confident": cls["confident"] or not auto_used,
                      "auto": auto_used}

    # ---- Layer 4: Document-Specific Verification (adapter, §6) ----
    verifier = get_verifier(doc_type)
    struct = verifier.verify(text)
    syn_checks = struct["checks"]
    syntax_score = struct["score"]
    if struct.get("extra"):
        syn_checks = syn_checks + struct["extra"]

    # ---- Layer 5b: Fabrication screen (type-independent, §2E) ----
    fab_hits = fab_scan(text)
    # Carve-out: the statistical tell (entropy) is suppressed for numbers
    # matching a known structured format (e.g. zero-padded licence serials
    # inside a valid DL number). Crisp tells still fire.
    if fab_hits:
        from doc_templates import KNOWN_STRUCTURED_FORMATS
        structured = set()
        for pats in KNOWN_STRUCTURED_FORMATS.values():
            for pat in pats:
                for m in re.finditer(pat, text, re.IGNORECASE):
                    structured.add(re.sub(r"\s+", "", m.group(0)).upper())
        kept = [h for h in fab_hits
                if not (h.get("kind") == "entropy" and
                        any(h["number"] in s for s in structured))]
        if len(kept) != len(fab_hits):
            fab_hits = kept
    fab_status = "FAIL" if fab_hits else "PASS"

    # ---- Layer 5: Semantics (kept) ----
    sem_checks, semantic_score = semantic_checks(text, doc_type, ocr_conf,
                                                 is_best_fit=(detected == doc_type))
    if auto_used:
        sem_checks.insert(0, {"check": "semantic:auto_detect", "status": "pass",
                              "message": f"Auto-detected as {TEMPLATES[doc_type]['label']} "
                                         f"(confidence {cls['confidence']:.2f})",
                              "value": doc_type})
    elif detected != requested and detected != "generic":
        gap = cls["raw_similarities"].get(detected, 0) - cls["raw_similarities"].get(requested, 0)
        if gap > 0.12:
            hint_msg = (f"You selected {TEMPLATES[requested]['label']} but this looks "
                        f"more like a {TEMPLATES[detected]['label']} — switch type or use Auto-detect")
            sem_checks.insert(0, {"check": "semantic:type_hint", "status": "warn",
                                  "message": hint_msg,
                                  "value": detected})
            classification["type_hint_msg"] = hint_msg
    sem_status = ("PASS" if semantic_score >= 0.6 else
                  "REVIEW" if semantic_score >= 0.35 else "FAIL")

    # ---- Layer 6: Cross-Field Validation (§12) ----
    # Strict MRZ pairs found in ANY orientation candidate are validated,
    # even if the winning read dropped them.
    cross = cross_validate(text, doc_type, struct.get("fields", {}),
                           mrz_lines=ocr.get("mrz_lines") or None)

    # ---- Layer 7: Tamper Analysis (§13, humble by design) ----
    tamper = tamper_analyze(photo_path, text, is_digital_pdf=is_digital)

    # ---- Layers 8–9: Evidence Aggregation + Decision (§14–15) ----
    evidence = {
        "ocr": _ocr_evidence(ocr_conf, True),
        "structure": {"status": struct["status"], "score": syntax_score},
        "semantic": {"status": sem_status, "score": semantic_score},
        "cross_field": {"status": cross["status"],
                        "conflicts": cross.get("conflicts", [])},
        "tamper": {"status": tamper["status"], "score": tamper.get("score"),
                   "flags": tamper.get("flags", []), "qr": tamper.get("qr")},
        "fabrication": {"status": fab_status, "hits": fab_hits,
                        "note": "Long ID numbers screened for fabricated patterns "
                                "(repetition/sequential/low variety)" if not fab_hits else
                                f"{len(fab_hits)} fabricated-looking number(s) found"},
        "cryptographic": {"status": "NOT_AVAILABLE",
                          "note": "Cryptographic authenticity not established — "
                                  "no official validator configured"},
        "image_quality": {"status": quality["status"] if quality["status"] != "N/A" else "PASS",
                          "score": quality["overall"]},
    }
    agg_score, breakdown = aggregate(doc_type, evidence)
    if classification.get("type_hint_msg"):
        classification["type_hint"] = classification["type_hint_msg"]
    decision, reasons, findings, risk, confidence = decide(
        doc_type, evidence, agg_score, classification, quality,
        struct_checks=syn_checks)
    # PAN honesty: format ≠ genuine.
    if doc_type == "pan" and decision == "VERIFIED":
        reasons.append("Note: valid PAN format does not mean the PAN is genuine; "
                       "no registry check performed.")
    from evidence import load_config as _load_cfg
    mode = _load_cfg().get("verification_mode", "strict")
    critical = [f for f in findings if f["severity"] == "CRITICAL"]
    warnings = [f for f in findings if f["severity"] == "WARNING"]
    highs = [f for f in findings if f["severity"] == "SUSPICIOUS"]
    passed = [f for f in findings if f["severity"] in ("PASS", "INFO")]
    explanation = reasons[0] if reasons else decision
    limitations = [f["message"] for f in findings if f["severity"] == "INFO"]

    print(json.dumps({
        "documentType": doc_type,
        "requestedType": requested,
        "detectedType": detected,
        "documentLabel": TEMPLATES[doc_type]["label"],
        "verdict": decision, "decision": decision, "reasons": reasons,
        "risk_level": risk, "confidence": confidence,
        "verification_mode": mode,
        "overallScore": agg_score,
        "ocrConfidence": ocr_conf,
        "syntaxScore": syntax_score,
        "semanticScore": semantic_score,
        "engine": engine,
        "extractedText": text[:4000],
        "image_quality": quality,
        "document_classification": classification,
        "evidence": evidence,
        "evidence_breakdown": breakdown,
        "findings": findings,
        "critical_findings": critical,
        "warnings": warnings,
        "high_findings": highs,
        "passed_checks": passed,
        "suspicious_regions": tamper.get("regions", []),
        "limitations": limitations,
        "explanation": explanation,
        "cross_field": cross,
        "tamper": tamper,
        "checks": syn_checks + sem_checks,
        "lines": ocr.get("lines", [])[:30],
    }))


if __name__ == "__main__":
    main()
