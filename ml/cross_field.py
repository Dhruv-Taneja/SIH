"""CrossFieldValidator — the same fact from two places must agree.

Compares information extracted from different parts/sources of a document:
  - repeated DOB / ID-number occurrences (OCR vs OCR)
  - OCR vs MRZ for passports (ICAO 9303 check digits — real math, no API)
  - marksheet arithmetic: total == sum(subjects), percentage consistency,
    obtained <= maximum

Returns {"status": PASS|FAIL|NOT_CHECKED, "conflicts": [...], "details": {...}}.
NOT_CHECKED is neutral (insufficient data), never evidence of anything.
"""
import re

GENERIC_DATE = re.compile(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})")


# ---------------------------------------------------------------- MRZ (ICAO 9303)

def mrz_check_digit(s):
    """ICAO check digit: weights 7-3-1, A-Z=10-35, '<'=0."""
    weights = (7, 3, 1)
    total = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            v = int(ch)
        elif "A" <= ch <= "Z":
            v = ord(ch) - 55
        elif ch == "<":
            v = 0
        else:
            return -1
        total += v * weights[i % 3]
    return total % 10


def find_mrz_lines(text):
    """Return TD3 MRZ lines (2×44) if present, else []. Strict: only
    exact-length lines qualify, because parsing needs exact positions."""
    cands = re.findall(r"[A-Z0-9<]{40,48}", text.replace(" ", ""))
    lines = [c for c in cands if len(c) == 44 and c.count("<") >= 5]
    out = []
    for i in range(len(lines) - 1):
        if lines[i][0] in "PVI" and lines[i + 1][0].isalnum():
            out.append((lines[i], lines[i + 1]))
    return out


def detect_mrz(text):
    """Loose MRZ presence test for CLASSIFICATION recall.

    Rotated/small MRZ often OCRs with dropped characters, so exact-44
    matching misses it. A long [A-Z0-9<] run containing '<<' filler is an
    overwhelmingly strong passport signal that normal text never produces.
    Returns the matched blob or ''.
    """
    compact = (text or "").upper().replace(" ", "")
    best = ""
    for m in re.findall(r"[A-Z0-9<]{30,50}", compact):
        if "<<" in m and sum(c == "<" for c in m) >= 4 and len(m) > len(best):
            best = m
    return best


def parse_mrz_td3(line1, line2):
    """Parse + validate TD3 pair. Returns dict with validity flags."""
    def ok(field, given):
        return mrz_check_digit(field) == given

    try:
        num, num_cd = line2[0:9], line2[9]
        dob, dob_cd = line2[13:19], line2[19]
        exp, exp_cd = line2[21:27], line2[27]
        per, per_cd = line2[28:42], line2[42]
        comp_cd = line2[43]
        checks = {
            "number": ok(num, int(num_cd)) if num_cd.isdigit() else False,
            "dob": ok(dob, int(dob_cd)) if dob_cd.isdigit() else False,
            "expiry": ok(exp, int(exp_cd)) if exp_cd.isdigit() else False,
            "personal": ok(per, int(per_cd)) if per_cd.isdigit() else False,
            "composite": ok(line2[0:10] + line2[13:20] + line2[21:43],
                             int(comp_cd)) if comp_cd.isdigit() else False,
        }
        name_raw = line1[5:44].replace("<", " ").strip()
        return {"format": "TD3", "valid": all(checks.values()),
                "checks": checks,
                "document_number": num.replace("<", ""),
                "nationality": line2[10:13],
                "dob_mrz": dob, "expiry_mrz": exp, "name_mrz": name_raw,
                "line1": line1, "line2": line2}
    except Exception as e:
        return {"format": "TD3", "valid": False, "error": str(e)}


# Each document type compares only its OWN number kind: a licence carries
# an alphanumeric DL number, never a numeric-only Aadhaar — hunting Aadhaar
# numbers on a DL (or vice versa) manufactures false conflicts. Generic
# (unknown type) still watches all kinds, labelled as-is.
KINDS_BY_TYPE = {
    "aadhaar": ["aadhaar"],
    "pan": ["pan"],
    "passport": ["passport"],
    "driving_licence": ["dl"],
    "marksheet": [],
    "certificate": [],
    "generic": ["aadhaar", "pan", "passport", "dl"],
}


# ------------------------------------------------------- field extraction

def _dob_context_dates(text):
    """Dates appearing near a DOB/birth label."""
    out = []
    for m in GENERIC_DATE.finditer(text):
        ctx = text[max(0, m.start() - 25):m.end()].lower()
        if "dob" in ctx or "birth" in ctx or "born" in ctx:
            out.append(m.group(0))
    return out


def _all_id_like(text, kind):
    # Collapse digit spacing first ("2345 6789 0123" → one run) so that a
    # 16-digit VID can't yield phantom overlapping 12-digit "Aadhaars".
    flat = re.sub(r"(?<=\d) (?=\d)", "", text)
    if kind == "aadhaar":
        return re.findall(r"\b[2-9]\d{11}\b", flat)
    if kind == "pan":
        return re.findall(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text.upper())
    if kind == "passport":
        return re.findall(r"\b[A-Z]{1,2}\d{6,7}\b", text.upper())
    if kind == "dl":
        return [" ".join(m.group(0).split()).upper() for m in
                re.finditer(r"\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{7}\b", text.upper())]
    return []


def _norm_date(d):
    p = re.split(r"[\/\-\.]", d)
    try:
        dd, mm, yy = int(p[0]), int(p[1]), int(p[2])
        if yy < 100:
            yy += 2000 if yy < 50 else 1900
        return f"{dd:02d}/{mm:02d}/{yy}"
    except (ValueError, IndexError):
        return d


# ------------------------------------------------------- marksheet arithmetic

def marksheet_arithmetic(text, tolerance_marks=2, tolerance_pct=1.5):
    """obtained<=max, total==sum(subjects), percentage consistency.

    Returns (rules, ok_count, total_count). Absent data → rule 'not checked'.
    """
    rules = []
    from doc_templates import TEMPLATES as _T
    pairs = []
    for m in re.finditer(_T["marksheet"]["patterns"]["marks"], text,
                         re.IGNORECASE):
        a, b = re.split(r"\s?\/\s?", m.group(1))
        a, b = int(a), int(b)
        # Denominators are per-subject maxima (≤1000); larger values are
        # years/dates caught by the pattern (e.g. "08/2008" in a DOB).
        if b <= 1000:
            pairs.append((a, b))
    totals = [int(x) for x in
              re.findall(r"\bTOTAL[\s:]*(\d{2,4})\b", text, re.IGNORECASE)]
    pcts = [float(x) for x in
            re.findall(r"\b(\d{1,3}(?:\.\d{1,2})?)\s?%", text)]
    words_total = re.findall(
        r"\bNINETY|EIGHTY|SEVENTY|SIXTY|FIFTY\b", text, re.IGNORECASE)

    if not pairs and not totals:
        rules.append({"rule": "has_figures", "status": "not_checked",
                      "message": "No tabular marks/total figures to audit"})
        return rules, 0, 0

    checked = passed = 0
    over = [(a, b) for a, b in pairs if a > b]
    if pairs:
        checked += 1
        if not over:
            passed += 1
            rules.append({"rule": "obtained_le_max", "status": "pass",
                          "message": f"All {len(pairs)} subject marks within maximum"})
        else:
            rules.append({"rule": "obtained_le_max", "status": "fail",
                          "message": f"Obtained > maximum in {len(over)} entries: {over[:3]}"})
    if totals and pairs:
        checked += 1
        s = sum(a for a, _ in pairs)
        if any(abs(s - t) <= tolerance_marks for t in totals):
            passed += 1
            rules.append({"rule": "total_eq_sum", "status": "pass",
                          "message": f"Total consistent with subject sum ({s})"})
        else:
            rules.append({"rule": "total_eq_sum", "status": "fail",
                          "message": f"Subject sum {s} matches no stated total {totals}"})
    if pcts and (pairs or totals):
        checked += 1
        base = (sum(a for a, _ in pairs), sum(b for _, b in pairs)) if pairs \
            else (totals[0], None)
        if base[1]:
            calc = round(100.0 * base[0] / base[1], 1)
            if any(abs(calc - p) <= tolerance_pct for p in pcts):
                passed += 1
                rules.append({"rule": "percentage_ok", "status": "pass",
                              "message": f"Stated percentage matches computed {calc}%"})
            else:
                rules.append({"rule": "percentage_ok", "status": "fail",
                              "message": f"Computed {calc}% matches no stated {pcts[:3]}"})
        elif words_total:
            passed += 1
            rules.append({"rule": "percentage_ok", "status": "pass",
                          "message": "Percentage stated in words; figures incomplete — accepted"})
    return rules, passed, checked


# ------------------------------------------------------- main validator

def validate(text, doc_type, ocr_fields=None, mrz_lines=None):
    """Cross-check repeated/parallel facts. Returns evidence dict.

    mrz_lines: strict TD3 pairs recovered from ANY OCR orientation
    candidate (the coherence vote may prefer a read that dropped them).
    """
    conflicts = []
    details = {}
    ocr_fields = ocr_fields or {}

    # 1. DOB label-context occurrences must agree with each other.
    dobs = {_norm_date(d) for d in _dob_context_dates(text)}
    details["dob_occurrences"] = sorted(dobs)
    if len(dobs) > 1:
        conflicts.append({"field": "date_of_birth", "source_1": "OCR",
                          "value_1": sorted(dobs)[0], "source_2": "OCR",
                          "value_2": sorted(dobs)[1],
                          "note": "Two different DOBs printed on one document"})

    # 2. Repeated ID numbers must agree — only the kind this document
    # actually carries (a DL repeats DL numbers, never Aadhaar numbers).
    for kind in KINDS_BY_TYPE.get(doc_type, []):
        vals = set(_all_id_like(text, kind))
        if vals:
            details[f"{kind}_occurrences"] = sorted(vals)
        if len(vals) > 1:
            conflicts.append({"field": kind + "_number", "source_1": "OCR",
                              "value_1": sorted(vals)[0], "source_2": "OCR",
                              "value_2": sorted(vals)[1],
                              "note": "Same document shows two different numbers"})

    # 3. Passport: OCR vs MRZ (parsed + check-digit validated).
    mrz_info = None
    if doc_type == "passport":
        pairs = mrz_lines or find_mrz_lines(text.upper().replace(" ", ""))
        if pairs:
            mrz_info = parse_mrz_td3(*pairs[0])
            details["mrz"] = {k: mrz_info[k] for k in
                              ("valid", "checks", "document_number",
                               "nationality", "dob_mrz", "expiry_mrz")
                              if k in mrz_info}
            if not mrz_info.get("valid"):
                conflicts.append({"field": "mrz_check_digits",
                                  "source_1": "MRZ", "value_1": "check digits",
                                  "source_2": "MRZ", "value_2": "do not validate",
                                  "note": "MRZ check digits fail — misread or altered MRZ"})
            else:
                ocr_no = (ocr_fields.get("passport_no") or "").upper()
                if ocr_no and ocr_no != mrz_info["document_number"]:
                    conflicts.append({"field": "passport_number",
                                      "source_1": "OCR", "value_1": ocr_no,
                                      "source_2": "MRZ",
                                      "value_2": mrz_info["document_number"],
                                      "note": "OCR number disagrees with MRZ"})
        else:
            details["mrz"] = {"present": False,
                              "note": "No MRZ detected (photo may crop it)"}

    # 4. Marksheet arithmetic.
    if doc_type == "marksheet":
        rules, ok_n, tot_n = marksheet_arithmetic(text)
        details["arithmetic"] = rules
        for r in rules:
            if r["status"] == "fail":
                conflicts.append({"field": "marks_arithmetic",
                                  "source_1": "table", "value_1": r["rule"],
                                  "source_2": "computed", "value_2": r["message"],
                                  "note": "Figures printed on the marksheet disagree"})

    if conflicts:
        return {"status": "FAIL", "conflicts": conflicts, "details": details}
    if details.get("dob_occurrences") or details.get("mrz") \
            or details.get("arithmetic") \
            or any(k.endswith("_occurrences") and details.get(k)
                   for k in details):
        return {"status": "PASS", "conflicts": [], "details": details,
                "note": "Repeated facts agree; no conflicts"}
    return {"status": "NOT_CHECKED", "conflicts": [], "details": details,
            "note": "Too little repeated data to cross-check — neutral"}
