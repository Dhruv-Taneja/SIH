# DocuVerify - Document Verification System

A web app to verify Indian documents like Aadhaar, PAN, Passport, Driving Licence, Marksheets and Certificates. Upload a scan or photo and it extracts the text using OCR and checks whether the details look valid.

Built for Smart India Hackathon (SIH).

## Features

- Upload document image (JPG, PNG) or PDF, or scan directly with camera
- Automatic text extraction with OCR
- Format and checksum validation (Aadhaar Verhoeff check, PAN format, passport MRZ check digits, marksheet total/percentage check)
- Document type detection
- Clear result: Verified, Needs Review or Rejected with score and reasons
- Shows extracted text and individual checks
- Organization login to keep track of verifications

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Node.js + Express
- ML Pipeline: Python, PaddleOCR, RapidOCR, OpenCV, scikit-learn
- File handling: Multer

## Supported Documents

- Aadhaar Card
- PAN Card
- Passport
- Driving Licence
- Marksheet
- Certificate
- Generic documents

## How to Run

1. Install backend dependencies:

```bash
npm install
```

2. Install Python dependencies (Python 3.9+ required):

```bash
pip install -r ml/requirements.txt
```

3. Start the server:

```bash
npm start
```

Open http://localhost:3000 in your browser.

Health check: http://localhost:3000/api/health

## How to Use

1. Select the document type.
2. Upload a file or scan with camera.
3. Click Verify Document to see the result, scores and extracted text.

Note: The first OCR run takes time because it downloads the models. For best results use a clear, well-lit photo.

## Organization Login

Organizations can sign up and sign in from the sign in panel. Verifications done while signed in are tagged with the organization name. Guest verification works without login.

Accounts are stored locally in `data/users.json` with hashed passwords.

## Project Structure

```
server.js              Express server and upload handling
public/                Frontend (index, styles, app logic)
ml/verify_document.py  Main verification pipeline
ml/doc_templates.py    Document patterns and keywords
ml/classifier.py       Document type classifier
ml/verifiers.py        Checks for each document type
ml/cross_field.py      Cross field validation and MRZ checks
ml/tamper.py           Tamper and quality signals
ml/evidence.py         Final scoring and decision
ml/image_quality.py    Image quality check
ml/check_engines.py    OCR engine status check
ml/config/documentVerification.json   Thresholds and weights
tests/test_pipeline.py Unit tests
sample/                Sample documents for testing
uploads/               Temporary uploads (auto deleted)
```

## Testing

```bash
python3 -m unittest discover -s tests -v
```

## Notes

- Uploaded files are temporary and deleted after verification.
- Max file size is 10 MB. Only images and PDFs allowed.
- If photo OCR does not work, install an OCR engine:

```bash
pip install rapidocr-onnxruntime
```

or

```bash
brew install tesseract
pip install pytesseract
```
