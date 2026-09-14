"""Shared document templates: keywords + syntax patterns per document type.

Imported by verify_document.py, classifier.py and verifiers.py so the
patterns live in exactly one place (no duplicated verification logic).
"""

TEMPLATES = {
    "aadhaar": {
        "label": "Aadhaar Card",
        "keywords": ["government of india", "aadhaar", "uidai", "dob", "date of birth",
                     "male", "female", "address", "vid", "proof of identity",
                     "verification", "authentication", "offline", "qr code", "masked"],
        "patterns": {
            "aadhaar_number": r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b",
            "vid": r"\bVID\s*[:\-]?\s*\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",
            "dob": r"\b(?:D[o0]B|D\.?\s*[o0O]\.?\s*B\.?|Date of Birth|D\.O\.B)[\s:]*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b",
            "gender": r"\b(MALE|FEMALE|TRANSGENDER)\b",
            "pincode": r"\b[1-9]\d{5}\b",
            "mobile": r"\b[6-9]\d{9}\b",
        },
        "required": ["aadhaar_number", "dob", "gender"],
    },
    "pan": {
        "label": "PAN Card",
        "keywords": ["income tax department", "permanent account number", "pan",
                     "govt of india", "date of birth", "father", "signature"],
        "patterns": {
            "pan_number": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
            "dob": r"\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b",
            "name": r"\b(?:Name|NAME)[\s:]*([A-Z][A-Za-z ]{2,40})\b",
        },
        "required": ["pan_number"],
    },
    "passport": {
        "label": "Passport",
        "keywords": ["republic of india", "passport", "nationality", "indian",
                     "date of birth", "place of birth", "date of issue", "date of expiry"],
        "patterns": {
            # Indian numbers are 1 letter + 7 digits (A1234567) or 2 letters
            # + 6 digits (AN192841) — both must match.
            "passport_no": r"\b[A-Z]{1,2}\d{6,7}\b",
            "dob": r"\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b",
            "mrz": r"[A-Z0-9<]{30,}",
            "nationality": r"\bINDIAN\b",
        },
        "required": ["passport_no"],
    },
    "driving_licence": {
        "label": "Driving Licence",
        "keywords": ["driving", "licence", "license", "transport", "motor",
                     "rto", "validity", "authorisation", "authorization",
                     "holder", "lmv", "mcwg", "nt", "inv", "issued"],
        "patterns": {
            "dl_no": r"\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{7}\b",
            "dob": r"\b(?:D[o0]B|D\.?\s*[o0O]\.?\s*B\.?|Date of Birth|D\.O\.B)[\s:]*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b",
            "validity": r"\b(?:Validity|Valid(?:\s*(?:Till|Upto))?|\(?NT\)?|\(?INV\)?)[\s:]*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b",
            "issue_date": r"\b(?:Issue\s*Date|DOI)[\s:]*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b",
            "name": r"\b(?:Name|NAME)[\s:]*([A-Z][A-Za-z ]{2,40})\b",
            "blood_group": r"\bBlood\s*Gr(?:ou)?p[\s:]*([ABO]{1,2}[+-])\b",
        },
        "required": ["dl_no"],
    },
    "marksheet": {
        "label": "Marksheet",
        "keywords": ["board", "central", "secondary", "examination", "cbse",
                     "university", "marks", "grade", "subject",
                     "theory", "practical", "roll", "enrolment", "enrollment",
                     "registration", "result", "pass", "division", "session",
                     "school", "college", "class", "affiliation", "statement",
                     "birth", "dob", "date", "semester", "total", "maximum"],
        "patterns": {
            # Label-based: OCR often mangles the label itself ("Roll No" →
            # "Rull Na", "Reg. No" → "Rgn.No"), so variants are included and
            # up to ~12 chars of debris allowed between label and number.
            "roll_no": r"\b(?:Roll|Rull|Rell|R0ll|RoII|Reg(?:istration)?|Rgn|Enrol[ml](?:ment)?|Enroll?ment)[\s\.:No\-–—a-zA-Z]{0,12}?(\d[\d\-\/]{3,19})\b",
            # Fallback: CBSE roll numbers are 7–8 straight digits.
            "roll_no_format": r"\b\d{7,8}\b",
            "year": r"\b(19\d{2}|20[0-3]\d)\b",
            "percentage": r"\b\d{1,3}(?:\.\d{1,2})?\s?%",
            "marks": r"\b(\d{1,3}\s?\/\s?\d{2,4})(?![\/\-\.]\d)\b",
            # Table-proof: OCR usually loses the "/" in mark tables, but
            # subject names + nearby numbers survive ("MATHEMATICS ... 095").
            "subject_marks": r"\b(?:MATHEMATICS|SCIENCE|ENGLISH|HINDI|SOCIAL|SANSKRIT|PHYSICS|CHEMISTRY|BIOLOGY|COMPUTER|MUSIC|LITERATURE|LANGUAGE|COURSE)\b[\s\S]{0,80}?\b(\d{2,3})\b",
            # Marks in words ("NINETY THREE") + standalone CBSE grades (A1–E2).
            "marks_words": r"\b(?:TWENTY|THIRTY|FORTY|FIFTY|SIXTY|SEVENTY|EIGHTY|NINETY)(\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE))?\b",
            "grade": r"\bGrade[\s:]*([A-E][+]?)\b",
            "grade_token": r"\b[ABCDE][12]\b",
            "dob": r"\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b",
        },
        "required": ["roll_no", "year"],
    },
    "certificate": {
        "label": "Certificate",
        "keywords": ["certificate", "certified", "awarded", "presented",
                     "completion", "achievement", "course", "training",
                     "issuer", "authority", "signatory", "seal", "recipient",
                     "grade", "valid", "credential"],
        "patterns": {
            "cert_id": r"\b(?:Certificate|Cert\.?|Credential|ID|No)[\s\.:No\-]*([A-Z0-9\-\/]{4,24})\b",
            "date": r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
            "year": r"\b(19\d{2}|20[0-3]\d)\b",
            "name": r"\b(?:awarded to|presented to|certified that|Mr\.?|Ms\.?|Mrs\.?)\s+([A-Z][A-Za-z .]{2,50})\b",
            "course": r"\b(?:course|training|program|workshop)[\s:]*([A-Za-z][A-Za-z &\-]{2,60})\b",
        },
        "required": [],
    },
    "generic": {
        "label": "Generic Document",
        "keywords": ["date", "name", "address", "signature", "authorized",
                     "seal", "certificate", "registration", "number", "details",
                     "issued", "valid", "office", "department"],
        "patterns": {
            "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "phone": r"\b[6-9]\d{9}\b",
            "date": r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
            "pincode": r"\b[1-9]\d{5}\b",
        },
        "required": [],
    },
}

# Types with a dedicated verifier adapter (everything else → GenericVerifier).
SPECIFIC_TYPES = ["aadhaar", "pan", "passport", "driving_licence",
                  "marksheet", "certificate"]

# Structured ID formats whose carriers legitimately contain low-variety runs
# (e.g. zero-padded licence serials). The fabrication screen's statistical
# tell (entropy) is suppressed for numbers matching these; crisp tells
# (all-same/sequential/repeated blocks) still fire.
KNOWN_STRUCTURED_FORMATS = {
    "driving_licence": [r"[A-Z]{2}\d{2}\s?\d{4}\s?\d{7}"],
}
