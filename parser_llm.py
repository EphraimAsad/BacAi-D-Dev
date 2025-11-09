# -------------------------------------------------
# Helpers for name normalization
# -------------------------------------------------
def _base_name(field: str) -> str:
    """Normalize a DB field to its base analyte name (e.g. 'Rhamnose Fermentation' -> 'rhamnose')."""
    return (
        field.lower()
        .replace(" fermentation", "")
        .replace(" utilization", "")
        .replace(" test", "")
        .strip()
    )

def _tokenize_analyte_list(s: str) -> list[str]:
    """
    Split 'glucose and sucrose, not lactose or rhamnose' into tokens.
    Accepts words, numbers, %, and hyphens (e.g., '6.5%').
    """
    # Replace separators with commas, then split
    s = re.sub(r"\s*(?:,|and|or|\&)\s*", ",", s.strip(), flags=re.I)
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    return tokens


# -------------------------------------------------
# Deterministic pattern extractor (fermentations, shorthand, NaCl)
# -------------------------------------------------
def extract_fermentations_regex(text: str, fermentation_fields: list[str]) -> dict:
    """
    Capture:
      - 'ferments X, Y and Z'  -> Positive for those
      - 'does not ferment X or Y' / 'non-fermenter for X' -> Negative
      - shorthand: 'lactose -', 'rhamnose +' (with optional 'fermentation')
      - 'ONPG positive/negative' -> ONPG Test
      - 'grows in 6.5% NaCl' / 'no growth in ... NaCl' -> NaCl Tolerance
    Maps only to fields that exist in the current DB.
    """
    out: dict[str, str] = {}
    t = text.lower()

    # Build quick lookup: base analyte -> full field name(s)
    base_to_field = {}
    for f in fermentation_fields:
        base_to_field.setdefault(_base_name(f), set()).add(f)

    # 1) Positive list: 'ferments X, Y and Z' / 'utilizes X ...'
    for m in re.finditer(r"(?:ferments|utilizes)\s+([a-z0-9\.\-%\s,/&]+)", t):
        analytes = _tokenize_analyte_list(m.group(1))
        for a in analytes:
            b = a.replace("(", "").replace(")", "").strip().lower()
            if b in base_to_field:
                for field in base_to_field[b]:
                    out[field] = "Positive"

    # 2) Negative list: 'does not ferment X or Y' / 'cannot ferment X' / 'non-fermenter for X'
    neg_patterns = [
        r"(?:does\s+not\s+(?:ferment|utilize)|cannot\s+(?:ferment|utilize)|unable\s+to\s+(?:ferment|utilize))\s+([a-z0-9\.\-%\s,/&]+)",
        r"non[-\s]?fermenter\s+(?:for|of)?\s+([a-z0-9\.\-%\s,/&]+)",
    ]
    for pat in neg_patterns:
        for m in re.finditer(pat, t):
            analytes = _tokenize_analyte_list(m.group(1))
            for a in analytes:
                b = a.replace("(", "").replace(")", "").strip().lower()
                if b in base_to_field:
                    for field in base_to_field[b]:
                        out[field] = "Negative"

    # 3) Shorthand per-analyte signs: 'lactose -', 'rhamnose +', optional 'fermentation'
    #    Capture word followed by optional 'fermentation' then a sign.
    for m in re.finditer(r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b", t):
        a, sign = m.group(1).lower(), m.group(2)
        if a in base_to_field:
            for field in base_to_field[a]:
                out[field] = "Positive" if sign == "+" else "Negative"

    # 4) ONPG explicit (often not modelled as 'Fermentation')
    #    Try to map to a field whose base is 'onpg' even if it's named 'ONPG Test'
    onpg_val = None
    if re.search(r"\bonpg\s*(?:test)?\s*positive\b", t):
        onpg_val = "Positive"
    elif re.search(r"\bonpg\s*(?:test)?\s*negative\b", t):
        onpg_val = "Negative"
    elif re.search(r"\bonpg\s*\+\b", t):
        onpg_val = "Positive"
    elif re.search(r"\bonpg\s*\-\b", t):
        onpg_val = "Negative"
    if onpg_val:
        # Try to find a matching field in fermentation_fields first, else let engine use its own 'ONPG Test' column if it exists
        if "onpg" in base_to_field:
            for field in base_to_field["onpg"]:
                out[field] = onpg_val
        else:
            out["ONPG Test"] = onpg_val  # safe fallback; ignored if DB lacks this column

    # 5) NaCl tolerance phrasing
    if re.search(r"\bgrows\s+in\s+[0-9\.]+\s*%?\s*na\s*cl\b", t):
        out["NaCl Tolerance"] = "Positive"
    if re.search(r"\bno\s+growth\s+in\s+[0-9\.]+\s*%?\s*na\s*cl\b", t):
        out["NaCl Tolerance"] = "Negative"

    return out
