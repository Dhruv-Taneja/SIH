"""Fabricated-number screen — catches obviously made-up ID numbers.

Genuine document numbers are effectively random. Fabricated ones almost
always show low-entropy tells: repeated digits (0000…), repeated groups
(0000 1111 2222), sequential runs (123456), or tiny digit variety.

This runs on EVERY document type over any long digit run — so a fake
Aadhaar misclassified as Generic is still caught.

Severity is quality-aware: blatant patterns on a clean read → CRITICAL;
on a poor read (digits themselves may be misread) → WARNING only.
"""
import math
import re


def _entropy(s):
    if not s:
        return 0.0
    return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s))
                for c in set(s))


def _has_sequential_run(s, length=6):
    d = [int(c) for c in s]
    for i in range(len(d) - length + 1):
        w = d[i:i + length]
        if all(b - a == 1 for a, b in zip(w, w[1:])):
            return True, "".join(map(str, w)), "ascending"
        if all(a - b == 1 for a, b in zip(w, w[1:])):
            return True, "".join(map(str, w)), "descending"
    return False, None, None


def plausibility(digits):
    """Check one digit string. Returns (ok, reason, kind).

    kind is one of repeated/block/groups/sequential/entropy — the crisp,
    structural tells (everything but entropy) are essentially impossible
    as OCR accidents, so they stay significant even on mediocre photos.
    """
    n = len(digits)
    if n < 6:
        return True, None, None
    # 1. One digit repeated (000000…, 9999…).
    if len(set(digits)) == 1:
        return False, f"all {n} digits are '{digits[0]}' — never genuine", "repeated"
    # 1b. Long run of one digit inside the number (…7777777…).
    m = re.search(r"(\d)\1{5,}", digits)
    if m:
        return False, f"run of {len(m.group(0))} identical '{m.group(1)}' digits — fabricated pattern", "repeated"
    # 2. Same block repeated (121212, 11111111, 12341234).
    for size in (2, 3, 4):
        if n >= size * 3 and n % size == 0:
            blocks = [digits[i:i + size] for i in range(0, n, size)]
            if len(set(blocks)) == 1:
                return False, f"block '{blocks[0]}' repeated {len(blocks)}× — fabricated pattern", "block"
    # 3. Repeated single-digit groups (0000 1111 2222, 9999 9999 9999).
    if n >= 8 and n % 4 == 0:
        groups = [digits[i:i + 4] for i in range(0, n, 4)]
        single = [g for g in groups if len(set(g)) == 1]
        if len(single) >= max(2, len(groups) / 2):
            return False, f"repeated single-digit groups ({' '.join(groups)}) — fabricated pattern", "groups"
    # 4. Long sequential run (…123456…, …654321…).
    if n >= 6:
        hit, run, direction = _has_sequential_run(digits)
        if hit:
            return False, f"sequential {direction} run '{run}' — never genuine", "sequential"
    # 5. Tiny digit variety over a long number.
    if n >= 10 and _entropy(digits) < 2.0:
        return False, f"only {len(set(digits))} distinct digits in {n} (entropy {_entropy(digits):.2f}) — fabricated pattern", "entropy"
    return True, None, None


def scan_text(text):
    """Find long digit runs (spaces collapsed) and flag fabricated ones."""
    flat = re.sub(r"(?<=\d)[ \u00a0](?=\d)", "", text or "")
    hits = []
    for m in re.finditer(r"\d{8,16}", flat):
        ok, reason, kind = plausibility(m.group(0))
        if not ok:
            hits.append({"number": m.group(0), "reason": reason, "kind": kind})
    # de-dup while keeping order
    seen, uniq = set(), []
    for h in hits:
        if h["number"] not in seen:
            seen.add(h["number"])
            uniq.append(h)
    return uniq


def mask_id(value, keep_ends=2):
    """Mask an identifier for display: 911407563214 → 91••••••••14."""
    s = str(value)
    if len(s) <= keep_ends * 2 or not any(c.isdigit() for c in s):
        return s
    return s[:keep_ends] + "•" * (len(s) - keep_ends * 2) + s[-keep_ends:]
