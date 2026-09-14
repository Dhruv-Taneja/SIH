"""EvidenceAggregator + DecisionEngine.

Replaces the old score = 0.35·OCR + 0.35·syntax + 0.30·semantic with
per-document-type weighted evidence plus hard rules:

  NEEDS_REVIEW — "cannot verify": poor image, uncertain classification,
      nothing to cross-check, tamper screen inconclusive.
  SUSPICIOUS   — contradictions: conflicting fields, invalid MRZ, strong
      tamper flags on a GOOD image. Never from low OCR alone.
  VERIFIED     — strong positive evidence AND no critical contradiction.

"Cannot verify" is NOT "fake". Thresholds/weights live in
config/documentVerification.json (documented, calibratable).
"""
import os
import json

STATUS_SCORE = {"PASS": 1.0, "REVIEW": 0.5, "FAIL": 0.0, "N/A": None,
                "NOT_AVAILABLE": None, "NOT_CHECKED": None}

# Finding severities (§1): INFO < WARNING < SUSPICIOUS < CRITICAL.
# (PASS/INFO are not findings.) A CRITICAL can never be averaged away —
# the decision matrix (§10) short-circuits on it before any score is read.
SEVERITY_RANK = {"INFO": 0, "PASS": 0, "WARNING": 1, "SUSPICIOUS": 2,
                 "CRITICAL": 3}

_CONFIG = None


def load_config():
    global _CONFIG
    if _CONFIG is None:
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "config", "documentVerification.json")
        with open(here) as f:
            _CONFIG = json.load(f)
    return _CONFIG


def aggregate(doc_type, evidence):
    """Weighted mean over AVAILABLE evidence. Returns (score, breakdown)."""
    cfg = load_config()
    weights = cfg["evidence_weights"].get(doc_type,
                                          cfg["evidence_weights"]["generic"])
    total = wsum = 0.0
    breakdown = {}
    for name, weight in weights.items():
        ev = evidence.get(name, {})
        st = ev.get("status", "NOT_AVAILABLE")
        val = STATUS_SCORE.get(st)
        if val is None:
            breakdown[name] = {"status": st, "excluded": True}
            continue
        # Fold raw sub-scores in where present (e.g. OCR confidence).
        raw = ev.get("score")
        if isinstance(raw, (int, float)):
            val = round(0.5 * val + 0.5 * max(0.0, min(1.0, raw)), 3)
        breakdown[name] = {"status": st, "value": val, "weight": weight}
        total += val * weight
        wsum += weight
    return (round(total / wsum, 3) if wsum else 0.0), breakdown


def _mode_cfg():
    cfg = load_config()
    mode = cfg.get("verification_mode", "strict")
    modes = cfg.get("strictness", {})
    return mode, modes.get(mode, next(iter(modes.values())))


def risk_level(decision, findings):
    """LOW / MEDIUM / HIGH — what the decision means for a reviewer."""
    sev = {f["severity"] for f in findings}
    if decision == "SUSPICIOUS":
        return "HIGH"
    if "SUSPICIOUS" in sev or "CRITICAL" in sev:
        return "MEDIUM"
    if decision == "NEEDS_REVIEW":
        return "MEDIUM"
    return "LOW"


def confidence_label(agg_score, findings):
    """Calibrated words, not fake percentages (§12)."""
    sev = {f["severity"] for f in findings}
    if "CRITICAL" in sev or "SUSPICIOUS" in sev:
        return "Strong evidence of a problem" if "CRITICAL" in sev \
            else "Moderate evidence of a problem"
    if agg_score >= 0.75:
        return "Strong supporting evidence"
    if agg_score >= 0.55:
        return "Moderate supporting evidence"
    if agg_score >= 0.35:
        return "Weak evidence"
    return "Insufficient evidence"


def build_findings(doc_type, evidence, classification, image_quality,
                   ocr_conf, struct_checks):
    """Translate evidence into severity-graded findings (§2, §4).

    Every finding: {check, category, severity, confidence, message, evidence}.
    Quality-aware by design: the same anomaly is CRITICAL on a clean readable
    image but only WARNING on noise (missing text ≠ fraud).
    """
    findings = []
    iq = image_quality.get("status")
    iq_good = iq in ("PASS", "N/A")
    ocr_ok = evidence.get("ocr", {}).get("status") == "PASS"
    cross = evidence.get("cross_field", {})
    tamper = evidence.get("tamper", {})

    def add(check, category, severity, confidence, message, ev=None):
        findings.append({"check": check, "category": category,
                         "severity": severity, "confidence": confidence,
                         "message": message, "evidence": ev or {}})

    # --- B: independent representations of the same field conflict ---
    for c in cross.get("conflicts", []):
        field = c.get("field", "")
        if field == "mrz_check_digits":
            add("cross_field_mrz", "cross_field", "CRITICAL", 0.95,
                "MRZ check digits do not validate — misread or altered MRZ", c)
        elif field == "marks_arithmetic":
            add("cross_field_arithmetic", "cross_field", "SUSPICIOUS", 0.85,
                "Marksheet figures disagree: " + c.get("value_2", ""), c)
        elif iq_good and ocr_ok:
            add("cross_field_" + field, "cross_field", "CRITICAL", 0.9,
                f"Independent representations of '{field}' contradict each other", c)
        else:
            add("cross_field_" + field, "cross_field", "SUSPICIOUS", 0.6,
                f"Possible '{field}' conflict, but image/OCR too poor to call it fraud", c)

    # --- Number fabrication: blatant patterns (0000 1111 2222) ---
    # Crisp structural tells (repeated/block/groups/sequential) are
    # essentially impossible as OCR accidents, so they stand even when the
    # photo is merely mediocre-but-readable. Only the statistical tell
    # (entropy) demands a clean read.
    fab = evidence.get("fabrication", {})
    for h in fab.get("hits", []):
        structural = h.get("kind", "entropy") != "entropy"
        if iq_good or (structural and ocr_ok and iq in ("REVIEW",)):
            add("fabricated_number", "fabrication", "CRITICAL", 0.9,
                f"ID number looks fabricated: {h['reason']}", h)
        else:
            add("fabricated_number", "fabrication", "WARNING", 0.6,
                f"ID number looks fabricated ({h['reason']}) — but read is poor, verify visually", h)
    # --- E/F: required structure absent despite a readable image ---
    # Guard: with very little extracted text, missing fields are EXPECTED
    # (small/rotated photo), not suspicious — downgrade to WARNING.
    words = 999
    for c in struct_checks or []:
        if c.get("check") == "syntax:text_volume" and isinstance(c.get("value"), int):
            words = c["value"]
    if evidence.get("structure", {}).get("status") == "FAIL":
        if iq_good and ocr_ok and words >= 15:
            add("structure_absent", "structure", "SUSPICIOUS", 0.8,
                "Required document structure is substantially absent despite a clear, readable image")
        else:
            add("structure_absent", "structure", "WARNING", 0.6,
                "Required structure missing — may just be unreadable; retake photo")

    # --- H: confidently classified as X, strong structure of Y ---
    if classification.get("type_hint"):
        add("type_mismatch", "classification", "SUSPICIOUS", 0.7,
            classification["type_hint"], {"hint": classification["type_hint"]})
    if not classification.get("confident", True):
        add("type_uncertain", "classification", "WARNING", 0.6,
            "Document type could not be established confidently")

    # --- G: tamper regions (overlap of independent signals escalates) ---
    for r in tamper.get("regions", []):
        sigs = r.get("signals", [])
        if len(sigs) >= 2:
            add("tamper_region", "tamper", "CRITICAL", 0.75,
                f"Suspicious region with overlapping signals: {', '.join(sigs)}", r)
        else:
            add("tamper_region", "tamper", "WARNING", 0.5,
                f"Region needs a look: {', '.join(sigs)}", r)
    for f in tamper.get("flags", []):
        add("tamper_flag", "tamper", "WARNING", 0.5, f)

    # --- OCR readability: WARNING at most, never fraud (§11) ---
    if evidence.get("ocr", {}).get("status") == "FAIL":
        add("ocr_unreadable", "ocr", "WARNING", 0.8,
            "Text unreadable — readability failure, not fraud evidence")
    elif evidence.get("ocr", {}).get("status") == "REVIEW":
        add("ocr_partial", "ocr", "WARNING", 0.6, "Text partially readable")

    # --- INFO: limitations that must be visible, not hidden ---
    add("crypto_unavailable", "cryptographic", "INFO", 1.0,
        "Cryptographic authenticity not established — no official validator configured")
    if doc_type == "pan":
        add("pan_format_disclaimer", "structure", "INFO", 1.0,
            "Valid PAN format does not mean the PAN is genuine; no registry check performed")
    return findings


def decide(doc_type, evidence, agg_score, classification, image_quality,
           struct_checks=None, tamper_regions_extra=None):
    """Strict decision matrix (§10). Rules beat scores; scores never hide
    a CRITICAL. Returns (decision, reasons, findings, risk, confidence)."""
    cfg = load_config()["decision"]
    mode, mcfg = _mode_cfg()
    verified_min = mcfg.get("verified_min", cfg["verified_min"])
    warn_needed = mcfg.get("warnings_with_one_high_for_suspicious", 1)
    need_img_pass = mcfg.get("verified_require_image_pass", True)

    findings = build_findings(doc_type, evidence, classification,
                              image_quality,
                              evidence.get("ocr", {}).get("score", 0),
                              struct_checks or [])
    crit = [f for f in findings if f["severity"] == "CRITICAL"]
    highs = [f for f in findings if f["severity"] == "SUSPICIOUS"]
    warns = [f for f in findings if f["severity"] == "WARNING"]

    iq_raw = image_quality.get("status")
    iq = "PASS" if iq_raw in ("PASS", "N/A") else iq_raw
    ocr_ok = evidence.get("ocr", {}).get("status") == "PASS"
    reasons = []

    # Q1 first: unreadable image ⇒ evidence itself is unreliable —
    # cannot verify, never suspicious (§3, §11).
    if iq_raw == "FAIL":
        reasons.append("Insufficient image quality for reliable verification — retake the photo.")
        return ("NEEDS_REVIEW", reasons, findings,
                risk_level("NEEDS_REVIEW", findings),
                confidence_label(agg_score, findings))

    # Hard fails next: a CRITICAL finding (fabricated number, conflicting
    # independent representations, overlapping tamper signals) outranks
    # everything — including an uncertain classification. A fabricated
    # number is fabricated no matter what type the document is.
    if crit:
        cats = sorted({f["category"] for f in crit})
        reasons.append(f"CRITICAL finding(s) in: {', '.join(cats)} — " +
                       crit[0]["message"])
        return ("SUSPICIOUS", reasons, findings, "HIGH",
                confidence_label(agg_score, findings))

    if not classification.get("confident", True) and classification.get("auto", False):
        reasons.append("Document type could not be established confidently — manual review needed.")
        return ("NEEDS_REVIEW", reasons, findings,
                risk_level("NEEDS_REVIEW", findings),
                confidence_label(agg_score, findings))

    # 2+ independent HIGH findings → SUSPICIOUS.
    high_cats = {f["category"] for f in highs}
    if len(highs) >= 2 and len(high_cats) >= 2:
        reasons.append("Multiple independent high-severity findings: " +
                       ", ".join(sorted(high_cats)) + ".")
        return ("SUSPICIOUS", reasons, findings, "HIGH",
                confidence_label(agg_score, findings))
    # 1 HIGH + WARNINGs → SUSPICIOUS (strict needs fewer warnings).
    if len(highs) >= 1 and len(warns) >= warn_needed:
        reasons.append("One high-severity finding plus supporting warnings: " +
                       highs[0]["message"])
        return ("SUSPICIOUS", reasons, findings, "HIGH",
                confidence_label(agg_score, findings))

    # VERIFIED: strong evidence, no significant anomaly, quality bar by mode.
    img_ok = (iq == "PASS") if need_img_pass else (iq in ("PASS", "REVIEW"))
    if agg_score >= verified_min and ocr_ok and img_ok and not highs:
        reasons.append("Available verification checks passed with no significant anomaly.")
        reasons.append("Note: cryptographic authenticity not established (no official validator configured).")
        if mode == "strict":
            reasons.append("Strict mode: every required check and cross-field screen held.")
        return ("VERIFIED", reasons, findings, "LOW",
                confidence_label(agg_score, findings))

    if agg_score >= cfg["review_min"]:
        reasons.append("Evidence is mixed or incomplete — manual review needed; not evidence of fakery.")
        return ("NEEDS_REVIEW", reasons, findings,
                risk_level("NEEDS_REVIEW", findings),
                confidence_label(agg_score, findings))
    if iq == "PASS" and ocr_ok:
        reasons.append("Readable document fails most checks — likely wrong document type or altered content.")
        return ("SUSPICIOUS", reasons, findings, "HIGH",
                confidence_label(agg_score, findings))
    reasons.append("Evidence insufficient for any conclusion.")
    return ("NEEDS_REVIEW", reasons, findings,
            risk_level("NEEDS_REVIEW", findings),
            confidence_label(agg_score, findings))
