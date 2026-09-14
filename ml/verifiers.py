"""Modular document verifiers — one adapter per document type.

Common interface:  verifier.verify(text) -> {
    "score": 0..1, "status": PASS|REVIEW|FAIL, "checks": [...], "fields": {...} }

Structure matching itself lives in verify_document.syntax_checks (its
long-standing, battle-tested location — imported lazily to avoid a module
cycle); adapters add the interface, status mapping, field extraction for
cross-field validation, and per-type extras. New document types only need
a template in doc_templates.py + a subclass here.
"""
import re

from doc_templates import TEMPLATES


def _structure(text, doc_type):
    from verify_document import syntax_checks  # lazy: avoids import cycle
    return syntax_checks(text, doc_type)


class BaseVerifier:
    doc_type = "generic"

    def verify(self, text):
        checks, score = _structure(text, self.doc_type)
        fields = self.extract_fields(text)
        tpl = TEMPLATES[self.doc_type]
        required_failed = any(
            c["check"] == f"syntax:{r}" and c["status"] == "fail"
            for r in tpl.get("required", []) for c in checks)
        if required_failed or score < 0.4:
            status = "FAIL"
        elif score < 0.7:
            status = "REVIEW"
        else:
            status = "PASS"
        out = {"score": score, "status": status,
               "checks": checks, "fields": fields}
        extra = self.extra_checks(text, fields)
        if extra:
            out["extra"] = extra
        return out

    def extract_fields(self, text):
        return {}

    def extra_checks(self, text, fields):
        return None


class AadhaarVerifier(BaseVerifier):
    doc_type = "aadhaar"

    def extract_fields(self, text):
        m = re.search(TEMPLATES["aadhaar"]["patterns"]["aadhaar_number"], text)
        d = re.search(TEMPLATES["aadhaar"]["patterns"]["dob"], text,
                      re.IGNORECASE)
        g = re.search(TEMPLATES["aadhaar"]["patterns"]["gender"], text)
        return {"aadhaar_number": re.sub(r"\D", "", m.group(0)) if m else None,
                "dob": d.group(1) if d else None,
                "gender": g.group(0) if g else None}

    def extra_checks(self, text, fields):
        from verify_document import verhoeff_check  # lazy: avoids cycle
        notes = []
        if fields.get("aadhaar_number"):
            n = fields["aadhaar_number"]
            notes.append({"check": "structure:checksum",
                          "status": "pass" if verhoeff_check(n) else "warn",
                          "message": "Verhoeff checksum validates the number format"
                                     if verhoeff_check(n) else
                                     "Checksum needed single-digit OCR tolerance (see syntax check)"})
        return notes or None


class PANVerifier(BaseVerifier):
    doc_type = "pan"

    def extract_fields(self, text):
        m = re.search(TEMPLATES["pan"]["patterns"]["pan_number"], text.upper())
        d = re.search(TEMPLATES["pan"]["patterns"]["dob"], text)
        n = re.search(TEMPLATES["pan"]["patterns"]["name"], text)
        out = {"pan_number": m.group(0) if m else None,
               "dob": d.group(1) if d else None,
               "name": n.group(1).strip() if n else None}
        # VALID FORMAT != GENUINE — stated on every PAN result.
        out["_disclaimer"] = ("Valid PAN format does not mean the PAN is "
                              "genuine; no official registry check performed.")
        return out


class PassportVerifier(BaseVerifier):
    doc_type = "passport"

    def extract_fields(self, text):
        m = re.search(TEMPLATES["passport"]["patterns"]["passport_no"],
                      text.upper())
        d = re.search(TEMPLATES["passport"]["patterns"]["dob"], text)
        nat = re.search(TEMPLATES["passport"]["patterns"]["nationality"], text)
        return {"passport_no": m.group(0) if m else None,
                "dob": d.group(1) if d else None,
                "nationality": nat.group(0) if nat else None}


class DrivingLicenceVerifier(BaseVerifier):
    doc_type = "driving_licence"

    def extract_fields(self, text):
        tpl = TEMPLATES["driving_licence"]["patterns"]
        m = re.search(tpl["dl_no"], text.upper())
        d = re.search(tpl["dob"], text, re.IGNORECASE)
        v = re.search(tpl["validity"], text, re.IGNORECASE)
        return {"dl_no": " ".join(m.group(0).split()) if m else None,
                "dob": d.group(1) if d else None,
                "validity": v.group(1) if v else None}

    def extra_checks(self, text, fields):
        notes = []
        v = fields.get("validity")
        d = fields.get("dob")
        if v and d:
            try:
                from datetime import datetime as _dt
                vd = _dt.strptime(v, "%d-%m-%Y")
                dd = _dt.strptime(d, "%d-%m-%Y")
                notes.append({"check": "structure:validity_logic",
                              "status": "pass" if vd > dd else "fail",
                              "message": "Licence validity after date of birth"
                                         if vd > dd else
                                         "Validity date precedes date of birth — impossible"})
            except ValueError:
                pass
        return notes or None


class MarksheetVerifier(BaseVerifier):
    doc_type = "marksheet"

    def extract_fields(self, text):
        tpl = TEMPLATES["marksheet"]["patterns"]
        r = re.search(tpl["roll_no"], text, re.IGNORECASE | re.DOTALL)
        y = re.search(tpl["year"], text)
        return {"roll_no": " ".join(r.group(0).split()) if r else None,
                "year": y.group(0) if y else None}


class CertificateVerifier(BaseVerifier):
    doc_type = "certificate"

    def extract_fields(self, text):
        tpl = TEMPLATES["certificate"]["patterns"]
        c = re.search(tpl["cert_id"], text, re.IGNORECASE)
        d = re.search(tpl["date"], text)
        n = re.search(tpl["name"], text)
        return {"cert_id": c.group(1).strip() if c else None,
                "date": d.group(1) if d else None,
                "recipient": n.group(1).strip() if n else None}


class GenericVerifier(BaseVerifier):
    doc_type = "generic"


_VERIFIERS = {"aadhaar": AadhaarVerifier, "pan": PANVerifier,
              "passport": PassportVerifier,
              "driving_licence": DrivingLicenceVerifier,
              "marksheet": MarksheetVerifier,
              "certificate": CertificateVerifier, "generic": GenericVerifier}


def get_verifier(doc_type):
    """Return the adapter for doc_type (GenericVerifier fallback)."""
    return _VERIFIERS.get(doc_type, GenericVerifier)()
