# 🛡️ DocuVerify — Document Verification (College Project, Local Only)

Upload / scan a document → **PaddleOCR** extracts text → **syntax checks** (regex, formats, Verhoeff checksum) → **semantic checks** (TF-IDF template match, entity completeness, date logic, coherence) → verdict: `VERIFIED` / `NEEDS_REVIEW` / `REJECTED`.

> ⚠️ **Do not deploy.** Local demo only.

## Architecture (layered pipeline)

```
Browser (public/) — upload / camera scan / evidence UI
   │  POST /api/upload (multipart: document + documentType)
   ▼
Node.js + Express (server.js) — limits, temp files, python-shell bridge
   │  python3 ml/verify_document.py <file> <type>
   ▼
Python pipeline (ml/):
  1. OCR (image_quality-gated): OpenCV preprocess → PaddleOCR → RapidOCR → Tesseract (best read wins).
     Orientation-invariant: EXIF honored, deskewed, 0/90/180/270° tried with
     coherence voting + early exit (upright stays at one pass per engine).
  2. ImageQualityChecker (image_quality.py): resolution/blur/brightness/contrast/glare.
     Poor image ⇒ NEEDS_REVIEW ("insufficient quality") — NEVER fraud evidence.
  3. DocumentClassifier (classifier.py): corpus-fit TF-IDF → softmax probabilities
     over types + confidence. Uncertain ⇒ NEEDS_REVIEW, never forced.
  4. Verifier adapters (verifiers.py + doc_templates.py): Aadhaar/PAN/Passport/
     Marksheet/Certificate/Generic share one interface — add types without rewrites.
  5. CrossFieldValidator (cross_field.py): repeated facts must agree; ICAO 9303
     MRZ check digits (real math, no API); marksheet arithmetic (total=sum, % check).
  6. TamperAnalyzer (tamper.py): QR detected≠readable≠authentic; ELA + noise screens.
     Signals only — never proof of forgery.
  7. EvidenceAggregator + DecisionEngine (evidence.py, config in
     ml/config/documentVerification.json): weighted evidence per doc type →
     VERIFIED (strong evidence, no contradiction) / NEEDS_REVIEW (cannot verify)
     / SUSPICIOUS (contradictions on a readable document).
```

Product principle: **"AI-assisted document screening"** — the system communicates
uncertainty. "Cannot verify" is NOT "fake". Cryptographic authenticity is reported
as `NOT_AVAILABLE` ("not established") unless an official validator is configured;
a valid format (e.g. PAN) is never presented as genuine.

## Supported documents

| Type | Structure | Cross-field emphasis |
|---|---|---|
| Aadhaar | 12-digit + Verhoeff (+1-digit OCR self-heal), DOB (OCR-tolerant label), gender, PIN | repeated numbers must agree |
| PAN | ABCDE1234F, holder-type letter, DOB | repeated numbers must agree |
| Passport | Passport no., MRZ, nationality | OCR vs MRZ + ICAO check digits |
| Driving Licence | DL no. (SS RR YYYY serial), DOB, validity, issue date | repeated numbers must agree; validity-after-DOB |
| Marksheet | Roll no. (label-tolerant), year, marks/grades | total=sum, percentage math |
| Certificate | Cert ID, recipient, date, course | date/field consistency |
| Generic | Found-pattern evidence only (no absent-field noise) | light checks |

## Run locally

**1. Backend deps**
```bash
npm install
```

**2. Python ML deps** (Python 3.9+; first run downloads PaddleOCR models ~100MB)
```bash
pip install -r ml/requirements.txt
```

**3. Start**
```bash
npm start
# open http://localhost:3000
```

Health check: `http://localhost:3000/api/health`

## Organization accounts (local only, optional)

- Anyone can verify without an account. Organizations may sign up / sign in
  from the **Organization sign in** panel; uploads are then logged under the org name.
- Credentials: scrypt-hashed, stored in `data/users.json`; sessions are opaque
  HttpOnly cookies (`data/sessions.json`, 7-day expiry). Nothing leaves this machine.
- Demo-grade auth for a college project — do not reuse these patterns for production.

## Try it (no account needed)

1. Pick document type (e.g. Aadhaar).
2. Drag-drop a JPG/PNG/PDF **or** click *Scan with camera* → Capture.
3. Hit **Verify document** → see verdict, score bars, per-check list, OCR text.

> Tip: first OCR run is slow (model download). Blurry photos → `NEEDS_REVIEW` with a "retake photo" hint. That's the coherence check working.

## For your project report / viva

- **OCR:** PaddleOCR (`use_angle_cls=True`), OpenCV preprocessing in `preprocess_image()`.
- **Syntax:** regexes in `TEMPLATES` + `syntax_checks()` + Verhoeff checksum for Aadhaar.
- **Semantics:** `semantic_checks()` — TF-IDF cosine similarity vs template, entity completeness (name/id/date), future-date detection, alpha-ratio coherence.
- **Scoring:** `overall = 0.35·OCR + 0.35·syntax + 0.30·semantic`; ≥0.70 + OCR≥0.50 → VERIFIED; ≥0.40 → NEEDS_REVIEW; else REJECTED.
- **JS↔Python:** `python-shell` v5 Promise API in `processDocument()`.

## Files

```
server.js              Express + multer + python-shell bridge
public/                Landing + verify UI with evidence sections
ml/verify_document.py  Pipeline entry (OCR, orchestration, legacy compat)
ml/doc_templates.py    Single source of patterns/keywords (no duplication)
ml/classifier.py       Probabilistic document classifier
ml/verifiers.py        Adapter per doc type (common interface)
ml/cross_field.py      Cross-field validator + ICAO MRZ math
ml/tamper.py           QR + ELA + noise screens (signals, not proof)
ml/evidence.py         Aggregator + decision engine
ml/image_quality.py    Independent quality gate
ml/check_engines.py    Engine availability probe (/api/health)
ml/config/documentVerification.json   Thresholds + weights (calibrate here)
tests/test_pipeline.py Unit tests — synthetic data only, no real IDs
sample/                Synthetic demo docs (fictional data)
uploads/               Temp uploads (auto-created, auto-cleaned, never stored)
```

## Tests + threshold calibration

```bash
python3 -m unittest discover -s tests -v   # 15 tests: checksum self-heal, MRZ,
                                           # arithmetic, classifier, quality, decisions
```

Thresholds/weights live in `ml/config/documentVerification.json` (documented
there). Calibrate on a labeled validation set (genuine / blurry / rotated /
altered-field / inconsistent-field cases, all synthetic) and report precision,
recall, F1 per doc type — not accuracy alone.

## Privacy (identity documents stay local)

- 100% local processing; uploads are never sent anywhere.
- Temp files deleted after every request; nothing is stored, no document DB.
- Server logs filenames only — never extracted text/numbers.
- 10 MB limit, image/PDF-only filter, UUID filenames (no path traversal).
- Python runs isolated per request; malformed files fail safe with JSON errors.

## Troubleshooting

**"OCR could not read any text (ocr-unavailable…)" on JPG/PNG photos?**
Photos need an OCR engine; digital PDFs don't (they carry a text layer).
This project tries engines in order — PaddleOCR → RapidOCR → Tesseract —
and keeps the best read. Install any ONE (easiest first):

```bash
# Option A — easiest, pip only, no system packages:
pip install rapidocr-onnxruntime

# Option B — classic Tesseract (macOS):
brew install tesseract && pip install pytesseract

# Option C — full PaddleOCR (biggest download, ~500MB):
pip install -r ml/requirements.txt
```

Check what's installed: open `http://localhost:3000/api/health` and look at
`engines.image_ocr_ready` (must be `true` for photos) and `engines.pdf_text_ready`.
The web UI also shows a warning banner on load when photo OCR is missing.

**For the sharpest scans:** hold the phone parallel to the page, fill the frame,
use daylight/white light (no yellow bulb + no shadow), wipe the camera lens.
The app deskews, denoises and auto-corrects spelling, but a blurry photo still
caps your score — garbage in, NEEDS_REVIEW out.

- `ML service failed…` → you skipped `npm install`, or `python3` isn't on PATH.
- Blurry/dark photos → `NEEDS_REVIEW` with a "retake photo" hint (coherence check working).
- Camera button fails → open via `http://localhost:3000` (not `file://`), allow camera permission, use Chrome/Edge.
