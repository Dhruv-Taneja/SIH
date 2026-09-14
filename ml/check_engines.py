"""Report which OCR/text backends are usable in this environment.

Used by server.js /api/health so the frontend can warn when photo OCR
is unavailable (digital PDFs still work via pypdf/PyMuPDF).

Usage: python3 ml/check_engines.py   → prints single JSON line.
"""
import importlib.util
import json
import shutil


def _try_import(name):
    # Check availability WITHOUT importing: real imports of cv2 /
    # onnxruntime / sklearn load 300MB+ and OOM small servers.
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def main():
    engines = {
        "paddleocr": _try_import("paddleocr"),
        "rapidocr": _try_import("rapidocr_onnxruntime"),
        "pytesseract_lib": _try_import("pytesseract"),
        "tesseract_binary": shutil.which("tesseract") is not None,
        "pymupdf": _try_import("fitz"),
        "pypdf": _try_import("pypdf"),
        "opencv": _try_import("cv2"),
        "sklearn": _try_import("sklearn"),
    }
    engines["image_ocr_ready"] = bool(
        engines["paddleocr"] or engines["rapidocr"]
        or (engines["pytesseract_lib"] and engines["tesseract_binary"])
    )
    engines["pdf_text_ready"] = bool(engines["pymupdf"] or engines["pypdf"])
    print(json.dumps(engines))


if __name__ == "__main__":
    main()
