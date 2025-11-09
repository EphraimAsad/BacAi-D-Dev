# parser_llm.py — Schema-clamped ultra build
# - Unicode/whitespace normalization
# - Robust fermentation parsing (incl. "but not … or/nor …" + fallback sweep)
# - ONPG / NaCl tolerant / growth temp / Gram
# - Media detection (XLD, MacConkey, Blood, etc.) with diagnostic exclusions
# - Colony morphology phrase capture
# - Oxygen requirement labels (Facultative Anaerobe, Obligate Aerobe/Anaerobe, Microaerophilic, Capnophilic, Intracellular)
# - Decarboxylases / dihydrolase
# - Haemolysis bridge: Type→Haemolysis (Alpha/Beta=Positive, Gamma/None=Negative)
# - **NEW**: Clamp all values to your exact Excel spellings (with synonyms)
# - **NEW**: Alias fixes (e.g., "Glucose Fermantation" → "Glucose Fermentation")

import os, json, re
from typing import Dict, List, Tuple, Set
from parser_basic import parse_input_free_text as fallback_parser


# ──────────────────────────────────────────────────────────────────────────────
# Allowed values & canonicalization (from your sheet)
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_VALUES: Dict[str, Set[str]] = {
    "Gram Stain": {"Positive", "Negative", "Variable"},
    "Shape": {"Cocci", "Rods", "Bacilli", "Spiral", "Short Rods"},
    "Catalase": {"Positive", "Negative", "Variable"},
    "Oxidase": {"Positive", "Negative", "Variable"},
    "Colony Morphology": set(),  # free text (you’ll provide later)
    "Haemolysis": {"Positive", "Negative", "Variable"},
    "Haemolysis Type": {"None", "Beta", "Gamma", "Alpha"},
    "Indole": {"Positive", "Negative", "Variable"},
    "Growth Temperature": set(),  # numeric user value (°C) — compare against DB "low//high"
    "Media Grown On": set(),      # use whitelist below, but allow free text fallback
    "Motility": {"Positive", "Negative", "Variable"},
    "Capsule": {"Positive", "Negative", "Variable"},
    "Spore Formation": {"Positive", "Negative", "Variable"},
    "Oxygen Requirement": {"Intracellular", "Aerobic", "Anaerobic", "Facultative Anaerobe", "Microaerophilic", "Capnophilic"},
    "Methyl Red": {"Positive", "Negative", "Variable"},
    "VP": {"Positive", "Negative", "Variable"},
    "Citrate": {"Positive", "Negative", "Variable"},
    "Urease": {"Positive", "Negative", "Variable"},
    "H2S": {"Positive", "Negative", "Variable"},
    "Lactose Fermentation": {"Positive", "Negative", "Variable"},
    "Glucose Fermentation": {"Positive", "Negative", "Variable"},
    "Sucrose Fermentation": {"Positive", "Negative", "Variable"},
    "Nitrate Reduction": {"Positive", "Negative", "Variable"},
    "Lysine Decarboxylase": {"Positive", "Negative", "Variable"},
    "Ornitihine Decarboxylase": {"Positive", "Negative", "Variable"},
    "Arginine dihydrolase": {"Positive", "Negative", "Variable"},
    "Gelatin Hydrolysis": {"Positive", "Negative", "Variable"},
    "Esculin Hydrolysis": {"Positive", "Negative", "Variable"},
    "Dnase": {"Positive", "Negative", "Variable"},
    "ONPG": {"Positive", "Negative", "Variable"},
    "NaCl Tolerant (>=6%)": {"Positive", "Negative", "Variable"},
    "Lipase Test": {"Positive", "Negative", "Variable"},
    "Xylose Fermentation": {"Positive", "Negative", "Variable"},
    "Rhamnose Fermentation": {"Positive", "Negative", "Variable"},
    "Mannitol Fermentation": {"Positive", "Negative", "Variable"},
    "Sorbitol Fermentation": {"Positive", "Negative", "Variable"},
    "Maltose Fermentation": {"Positive", "Negative", "Variable"},
    "Arabinose Fermentation": {"Positive", "Negative", "Variable"},
    "Raffinose Fermentation": {"Positive", "Negative", "Variable"},
    "Inositol Fermentation": {"Positive", "Negative", "Variable"},
    "Trehalose Fermentation": {"Positive", "Negative", "Variable"},
    "Coagulase": {"Positive", "Negative", "Variable"},
}

# Media whitelist (from your list). We still accept other "___ Agar" phrases,
# but we’ll title-case and de-duplicate them; diagnostic media are excluded elsewhere.
MEDIA_WHITELIST = {
    "MacConkey Agar", "Nutrient Agar", "ALOA", "Palcam", "Preston", "Columbia", "BP",
    "Mannitol Salt Agar", "MRS", "Anaerobic Media", "XLD Agar", "TBG", "TCBS", "VID",
    "EMB Agar", "CCI", "Salt Nutrient Agar", "Thayer Martin Agar", "Tryptic Soy Agar",
    "Chocolate Agar", "Bacteroides Bile Esculin Agar", "KVLB Agar", "Charcoal Blood Agar",
    "Anaerobic Blood Agar", "Yeast Extract Mannitol Agar", "Burks Medium", "Peptone Water",
    "Sabouraud Dextrose Agar", "Yeast Extract Peptone Dextrose", "Malt Extract Agar",
    "Middlebrook Agar", "Inorganic Mineral Nitrate Media", "Inorganic Mineral Ammonia Media",
    "Iron Media", "Sulfur Media", "Organic Media", "Yeast Extract Agar", "Cellulose Agar",
    "Baciillus Media", "Pyridoxal", "Lcysteine", "Ferrous Sulfate Media",
    "Hayflicks Agar", "Cell Culture", "Intracellular", "Brain Heart Infusion Agar",
    "Human Fibroblast Cell Culture", "BCYE Agar"
}

# synonym tables to clamp free-text to your exact spellings
VALUE_SYNONYMS: Dict[str, Dict[str, str]] = {
    # Gram Stain
    "Gram Stain": {
        "gram positive": "Positive", "gram-positive": "Positive", "g+": "Positive",
        "gram negative": "Negative", "gram-negative": "Negative", "g-": "Negative",
        "variable": "Variable"
    },
    # Shape
    "Shape": {
        "rod": "Rods", "rods": "Rods", "bacillus": "Bacilli", "bacilli": "Bacilli",
        "coccus": "Cocci", "cocci": "Cocci", "spiral": "Spiral", "short rods": "Short Rods"
    },
    # Generic test polarity
    "*POLARITY*": {
        "+": "Positive", "positive": "Positive", "pos": "Positive",
        "-": "Negative", "negative": "Negative", "neg": "Negative",
        "weakly positive": "Variable", "variable": "Variable", "weak": "Variable"
    },
    # Oxygen requirement
    "Oxygen Requirement": {
        "facultative": "Facultative Anaerobe", "facultative anaerobe": "Facultative Anaerobe",
        "facultative aerobe": "Facultative Anaerobe",  # clamp to your set
        "aerobe": "Aerobic", "aerobic": "Aerobic", "obligate aerobe": "Aerobic",
        "anaerobe": "Anaerobic", "anaerobic": "Anaerobic", "obligate anaerobe": "Anaerobic",
        "microaerophile": "Microaerophilic", "microaerophilic": "Microaerophilic",
        "capnophile": "Capnophilic", "capnophilic": "Capnophilic",
        "intracellular": "Intracellular"
    },
    # Haemolysis Type
    "Haemolysis Type": {
        "beta": "Beta", "β": "Beta", "alpha": "Alpha", "α": "Alpha",
        "gamma": "Gamma", "γ": "Gamma", "none": "None"
    },
}

# fields that use POLARITY synonyms
POLARITY_FIELDS = {
    "Catalase","Oxidase","Haemolysis","Indole","Motility","Capsule","Spore Formation",
    "Methyl Red","VP","Citrate","Urease","H2S","Lactose Fermentation","Glucose Fermentation",
    "Sucrose Fermentation","Nitrate Reduction","Lysine Decarboxylase","Ornitihine Decarboxylase",
    "Arginine dihydrolase","Gelatin Hydrolysis","Esculin Hydrolysis","Dnase","ONPG",
    "NaCl Tolerant (>=6%)","Lipase Test","Xylose Fermentation","Rhamnose Fermentation",
    "Mannitol Fermentation","Sorbitol Fermentation","Maltose Fermentation","Arabinose Fermentation",
    "Raffinose Fermentation","Inositol Fermentation","Trehalose Fermentation","Coagulase"
}


# ──────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ──────────────────────────────────────────────────────────────────────────────
def normalize_columns(db_fields: List[str]) -> List[str]:
    """Exact DB fields with original casing, excluding 'Genus'."""
    return [f for f in db_fields if f and f.strip().lower() != "genus"]


def summarize_field_categories(db_fields: List[str]) -> Dict[str, List[str]]:
    """Light categorization (for prompt context only)."""
    cats = {"Morphology": [], "Enzyme": [], "Fermentation": [], "Other": []}
    for f in normalize_columns(db_fields):
        n = f.strip()
        l = n.lower()
        if any(k in l for k in ["gram", "shape", "morphology", "motility", "capsule", "spore", "oxygen requirement", "media grown"]):
            cats["Morphology"].append(n)
        elif any(k in l for k in [
            "oxidase","catalase","urease","coagulase","lipase","indole",
            "citrate","vp","methyl red","gelatin","dnase","nitrate","h2s","esculin"
        ]):
            cats["Enzyme"].append(n)
        elif "fermentation" in l or "utilization" in l:
            cats["Fermentation"].append(n)
        else:
            cats["Other"].append(n)
    return cats


# ──────────────────────────────────────────────────────────────────────────────
# Aliases → exact sheet column names
# ──────────────────────────────────────────────────────────────────────────────
def build_alias_map(db_fields: List[str]) -> Dict[str, str]:
    """Common phrases/abbreviations to your exact Excel columns (keys lowercase)."""
    exact = {f.lower(): f for f in normalize_columns(db_fields)}
    alias: Dict[str, str] = {}

    def add(a: str, target: str):
        t = target.lower()
        if t in exact:
            alias[a.lower()] = exact[t]

    # Canonical/typo-fix names
    add("mr", "Methyl Red"); add("methyl red", "Methyl Red")
    add("vp", "VP"); add("voges proskauer", "VP")
    add("h2s", "H2S"); add("dnase", "Dnase")
    add("gelatin", "Gelatin Hydrolysis"); add("gelatin liquefaction", "Gelatin Hydrolysis")
    add("lipase", "Lipase Test"); add("lipase test", "Lipase Test")
    add("onpg", "ONPG"); add("onpg test", "ONPG")
    add("esculin hydrolysis", "Esculin Hydrolysis")
    add("nacl tolerance", "NaCl Tolerant (>=6%)"); add("nacl tolerant", "NaCl Tolerant (>=6%)"); add("nacl", "NaCl Tolerant (>=6%)")
    add("nitrate", "Nitrate Reduction"); add("nitrate reduction", "Nitrate Reduction")
    add("lysine decarboxylase", "Lysine Decarboxylase")
    add("ornithine decarboxylase", "Ornitihine Decarboxylase"); add("ornitihine decarboxylase", "Ornitihine Decarboxylase")
    add("arginine dihydrolase", "Arginine dihydrolase")
    add("coagulase", "Coagulase"); add("citrate", "Citrate")
    add("urease", "Urease"); add("indole", "Indole")
    add("oxidase", "Oxidase"); add("catalase", "Catalase")
    add("motility", "Motility"); add("capsule", "Capsule")
    add("spore formation", "Spore Formation")
    add("haemolysis", "Haemolysis"); add("haemolysis type", "Haemolysis Type")
    add("growth temperature", "Growth Temperature")
    add("media grown on", "Media Grown On")
    add("oxygen requirement", "Oxygen Requirement")
    add("gram stain", "Gram Stain"); add("shape", "Shape")
    # important typo field in your sheet headers or user text:
    add("glucose fermantation", "Glucose Fermentation")  # typo → canonical

    # Fermentation bases (e.g., "rhamnose" → "Rhamnose Fermentation")
    for f in normalize_columns(db_fields):
        if f.lower().endswith(" fermentation"):
            base = f[:-12].strip().lower()
            alias[base] = f

    # Media aliases (lowercase keys → canonical exact)
    for m in MEDIA_WHITELIST:
        alias[m.lower()] = "Media Grown On"

    return alias


# ──────────────────────────────────────────────────────────────────────────────
# Text normalization utilities
# ──────────────────────────────────────────────────────────────────────────────
_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def normalize_text(raw: str) -> str:
    t = raw or ""
    t = t.replace("°", " °")
    t = t.translate(_SUBSCRIPT_DIGITS)
    t = (t.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
           .replace("\u2013", "-").replace("\u2014", "-"))
    t = re.sub(r"hemolys", "haemolys", t, flags=re.I)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _normalize_token(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"[.,;:!?\u2013\u2014\-]+$", "", s)
    return s.strip()

def _tokenize_list(s: str) -> List[str]:
    s = re.sub(r"\s*(?:,|and|or|&|nor)\s*", ",", s.strip(), flags=re.I)
    items = [t.strip() for t in s.split(",") if t.strip()]
    return [re.sub(r"[.,;:\s]+$", "", i) for i in items]


# ──────────────────────────────────────────────────────────────────────────────
# Safe setter & canonicalization helpers
# ──────────────────────────────────────────────────────────────────────────────
def _set_field_safe(out: Dict[str, str], key: str, val: str):
    cur = out.get(key)
    if cur is None:
        out[key] = val; return
    if cur == "Negative" and val == "Positive":
        return
    out[key] = val

def _canon_value(field: str, value: str) -> str:
    """Map synonyms/short forms to your exact spellings, then clamp to ALLOWED_VALUES if defined."""
    v = (value or "").strip()
    if not v:
        return v

    # polarity fields use "*POLARITY*" synonyms first
    if field in POLARITY_FIELDS:
        low = v.lower()
        if low in VALUE_SYNONYMS.get("*POLARITY*", {}):
            v = VALUE_SYNONYMS["*POLARITY*"][low]
        else:
            # "+/-" inside string
            if re.fullmatch(r"\+|positive|pos", low): v = "Positive"
            elif re.fullmatch(r"\-|negative|neg", low): v = "Negative"
            elif "weak" in low or "variable" in low: v = "Variable"

    # field-specific synonyms
    low = v.lower()
    if field in VALUE_SYNONYMS:
        v = VALUE_SYNONYMS[field].get(low, v)

    # Clamp to allowed if set
    allowed = ALLOWED_VALUES.get(field)
    if allowed:
        # accept only if exact; else try title-case variants
        if v not in allowed:
            # try title-case and sentence-case
            tv = v.title()
            if tv in allowed:
                v = tv
            else:
                # last resort: polarity guess for tests
                if field in POLARITY_FIELDS:
                    # default to "Positive"/"Negative"/"Variable"? Leave original if unknown; engine will treat "Unknown" if omitted
                    pass
    return v


# ──────────────────────────────────────────────────────────────────────────────
# Regex enrichment: fermentations + ONPG + NaCl
# ──────────────────────────────────────────────────────────────────────────────
def extract_fermentations_regex(text: str, db_fields: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    t = normalize_text(text)
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)

    ferm_fields = [f for f in fields if f.lower().endswith(" fermentation")]
    base_to_field = {f[:-12].strip().lower(): f for f in ferm_fields}

    def set_field_by_base(base: str, val: str):
        b = _normalize_token(base)
        if b in base_to_field:
            _set_field_safe(out, base_to_field[b], _canon_value(base_to_field[b], val))
        elif b in alias and alias[b] in fields:
            _set_field_safe(out, alias[b], _canon_value(alias[b], val))

    # POSITIVE lists ("ferments"/"utilizes")
    for m in re.finditer(r"(?:ferments|utilizes)\s+([a-z0-9\.\-%\s,/&]+)", t, flags=re.I):
        span = re.split(r"(?i)\bbut\s+not\b", m.group(1))[0]
        for a in _tokenize_list(span):
            set_field_by_base(a, "Positive")

    # NEGATIVE lists (explicit)
    neg_pats = [
        r"(?:does\s+not|doesn't)\s+(?:ferment|utilize)\s+([a-z0-9\.\-%\s,/&]+)",
        r"cannot\s+(?:ferment|utilize)\s+([a-z0-9\.\-%\s,/&]+)",
        r"unable\s+to\s+(?:ferment|utilize)\s+([a-z0-9\.\-%\s,/&]+)",
        r"non[-\s]?fermenter\s+(?:for|of)?\s+([a-z0-9\.\-%\s,/&]+)",
    ]
    for pat in neg_pats:
        for m in re.finditer(pat, t, flags=re.I):
            for a in _tokenize_list(m.group(1)):
                set_field_by_base(a, "Negative")

    # NEGATIVE after "but not X, Y or Z / nor Z"
    for m in re.finditer(r"(?:ferments|utilizes)[^.]*?\bbut\s+not\s+([\w\s,;.&-]+)", t, flags=re.I):
        segment = m.group(1)
        segment = re.sub(r"\bor\b", ",", segment, flags=re.I)
        segment = re.sub(r"\bnor\b", ",", segment, flags=re.I)
        for a in _tokenize_list(segment):
            a = re.sub(r"[.,;:\s]+$", "", a)
            set_field_by_base(a, "Negative")
        seg_l = " " + segment.lower() + " "
        for base in base_to_field.keys():
            if re.search(rf"\b{re.escape(base)}\b", seg_l, flags=re.I):
                set_field_by_base(base, "Negative")

    # Shorthand "lactose -" / "rhamnose +"
    for m in re.finditer(r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b", t, flags=re.I):
        a, sign = m.group(1), m.group(2)
        set_field_by_base(a, "Positive" if sign == "+" else "Negative")

    # ONPG
    if re.search(r"\bonpg\s*(?:test)?\s*(?:is\s+)?(\+|positive)\b", t, flags=re.I):
        if "onpg" in alias and alias["onpg"] in fields:
            _set_field_safe(out, alias["onpg"], "Positive")
    elif re.search(r"\bonpg\s*(?:test)?\s*(?:is\s+)?(\-|negative)\b", t, flags=re.I):
        if "onpg" in alias and alias["onpg"] in fields:
            _set_field_safe(out, alias["onpg"], "Negative")

    # NaCl tolerant
    if re.search(r"\b(tolerant|grows|growth)\s+(?:in|up\s+to|to|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t, flags=re.I):
        if "nacl tolerant" in alias and alias["nacl tolerant"] in fields:
            _set_field_safe(out, alias["nacl tolerant"], "Positive")
    if re.search(r"\bno\s+growth\s+(?:in|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t, flags=re.I):
        if "nacl tolerant" in alias and alias["nacl tolerant"] in fields:
            _set_field_safe(out, alias["nacl tolerant"], "Negative")
    if re.search(r"\bnacl\s+tolerant\b", t, flags=re.I):
        if "nacl tolerant" in alias and alias["nacl tolerant"] in fields:
            _set_field_safe(out, alias["nacl tolerant"], "Positive")

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Regex enrichment: morphology, enzyme/other tests, capsule, haemolysis, oxygen, growth temp,
# media detection (with diagnostic exclusion), colony morphology
# ──────────────────────────────────────────────────────────────────────────────
def extract_biochem_regex(text: str, db_fields: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    t = normalize_text(text)
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)

    def set_field(key_like: str, val: str):
        target = alias.get(key_like.lower(), key_like)
        if target in fields:
            _set_field_safe(out, target, _canon_value(target, val))

    # Gram
    gram_pos = re.search(r"\bgram[-\s]?positive\b", t, flags=re.I)
    gram_neg = re.search(r"\bgram[-\s]?negative\b", t, flags=re.I)
    if gram_pos and not gram_neg:
        set_field("gram stain", "Positive")
    elif gram_neg and not gram_pos:
        set_field("gram stain", "Negative")

    # Generic biochemical (+/-/weak)
    generic_targets = [
        "catalase","oxidase","coagulase","urease","lipase","indole",
        "citrate","vp","methyl red","gelatin","dnase","nitrate reduction","nitrate","h2s","esculin hydrolysis"
    ]
    for test in generic_targets:
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:\+|positive|detected)\b", t, flags=re.I):
            set_field(test, "Positive")
        elif re.search(rf"\b{test}\s*(?:test)?\s*(?:\-|negative|not\s+detected)\b", t, flags=re.I):
            set_field(test, "Negative")
        elif re.search(rf"\b{test}\s*(?:test)?\s*weak(?:ly)?\s*positive\b", t, flags=re.I):
            set_field(test, "Variable")

    # Nitrate phrase variants
    if re.search(r"\breduces\s+nitrate\b", t, flags=re.I):
        set_field("nitrate", "Positive")
    if re.search(r"\bdoes\s+not\s+reduce\s+nitrate\b", t, flags=re.I):
        set_field("nitrate", "Negative")

    # H2S precedence
    if re.search(r"\bproduces\s+h\s*2\s*s\s+negative\b", t, flags=re.I):
        set_field("h2s", "Negative")
    if re.search(r"\bh\s*2\s*s\s+(?:\+|positive|detected)\b", t, flags=re.I):
        set_field("h2s", "Positive")
    if re.search(r"\bh\s*2\s*s\s+(?:\-|negative|not\s+detected)\b", t, flags=re.I):
        set_field("h2s", "Negative")
    if re.search(r"\bproduces\s+h\s*2\s*s\b", t, flags=re.I):
        set_field("h2s", "Positive")

    # Gelatin liquefaction / Esculin phrasing
    if re.search(r"\bgelatin\s+liquefaction\s+(?:\+|positive)\b", t, flags=re.I):
        set_field("gelatin", "Positive")
    if re.search(r"\besculin\s+hydrolysis\s+(?:\+|positive)\b", t, flags=re.I) or \
       re.search(r"\bpositive\s+esculin\s+hydrolysis\b", t, flags=re.I):
        set_field("esculin hydrolysis", "Positive")

    # Capsule
    if re.search(r"\b(capsulated|encapsulated)\b", t, flags=re.I):
        set_field("capsule", "Positive")
    if re.search(r"\bnon[-\s]?capsulated\b", t, flags=re.I):
        set_field("capsule", "Negative")

    # Haemolysis type
    if re.search(r"\b(beta|β)[-\s]?haem", t, flags=re.I):
        set_field("haemolysis type", "Beta")
    elif re.search(r"\b(alpha|α)[-\s]?haem", t, flags=re.I):
        set_field("haemolysis type", "Alpha")
    elif re.search(r"\b(gamma|γ)[-\s]?haem", t, flags=re.I) or re.search(r"\bno\s+haemolysis\b", t, flags=re.I):
        set_field("haemolysis type", "Gamma")

    # Oxygen requirement (expanded)
    if re.search(r"\bfacultative\b", t, flags=re.I):
        set_field("oxygen requirement", "Facultative Anaerobe")
    elif re.search(r"\baerobic\b", t, flags=re.I):
        set_field("oxygen requirement", "Aerobic")
    elif re.search(r"\banaerobic\b", t, flags=re.I):
        set_field("oxygen requirement", "Anaerobic")
    elif re.search(r"\bmicroaerophil(ic|e)\b", t, flags=re.I):
        set_field("oxygen requirement", "Microaerophilic")
    elif re.search(r"\bcapnophil(ic|e)\b", t, flags=re.I):
        set_field("oxygen requirement", "Capnophilic")
    elif re.search(r"\bintracellular\b", t, flags=re.I):
        set_field("oxygen requirement", "Intracellular")

    # Decarboxylases/dihydrolase
    decarbox_patterns = [
        ("lysine decarboxylase", r"\blysine\s+decarboxylase\s+(?:test\s+)?(\+|positive)\b", "Positive"),
        ("lysine decarboxylase", r"\blysine\s+decarboxylase\s+(?:test\s+)?(\-|negative)\b", "Negative"),
        ("ornithine decarboxylase", r"\bornithine\s+decarboxylase\s+(?:test\s+)?(\+|positive)\b", "Positive"),
        ("ornithine decarboxylase", r"\bornithine\s+decarboxylase\s+(?:test\s+)?(\-|negative)\b", "Negative"),
        ("ornitihine decarboxylase", r"\bornitihine\s+decarboxylase\s+(?:test\s+)?(\+|positive)\b", "Positive"),
        ("ornitihine decarboxylase", r"\bornitihine\s+decarboxylase\s+(?:test\s+)?(\-|negative)\b", "Negative"),
        ("arginine dihydrolase", r"\barginine\s+dihydrolase\s+(?:test\s+)?(\+|positive)\b", "Positive"),
        ("arginine dihydrolase", r"\barginine\s+dihydrolase\s+(?:test\s+)?(\-|negative)\b", "Negative"),
    ]
    for key, pat, val in decarbox_patterns:
        if re.search(pat, t, flags=re.I):
            set_field(key, val)

    # Growth Temperature: only when "grows at"
    for m in re.finditer(r"(?<!no\s)grows\s+(?:well\s+)?at\s+([0-9]{2,3})\s*°?\s*c", t, flags=re.I):
        temp_num = m.group(1)
        set_field("growth temperature", temp_num)

    # Media detection with diagnostic exclusion
    diagnostic_exclude = ["triple sugar iron", "tsi", "sim", "kligler", "msrv"]
    media_hits = re.findall(r"\b([a-z0-9\-\+]+)\s+agar\b", t, flags=re.I)
    collected_media: List[str] = []
    for mname in media_hits:
        lowname = mname.lower()
        if any(ex in lowname for ex in diagnostic_exclude):
            continue
        name = (mname.strip().upper())
        if name in {"XLD", "MACCONKEY", "BLOOD"}:
            pretty = "XLD Agar" if name == "XLD" else \
                     "MacConkey Agar" if name == "MACCONKEY" else "Blood Agar"
        else:
            pretty = mname.strip().title() + " Agar"
        # Clamp to whitelist spelling if close
        if pretty in MEDIA_WHITELIST:
            canon = pretty
        else:
            canon = pretty  # allow others; still set as title-cased
        if canon not in collected_media:
            collected_media.append(canon)
    if collected_media:
        _set_field_safe(out, "Media Grown On", "; ".join(collected_media))

    # Colony morphology phrase
    col_match = re.search(r"colon(?:y|ies)\s+(?:are|appear)\s+([^.]+?)(?:\s+on|\.)", t, flags=re.I)
    if col_match:
        desc = col_match.group(1).strip()
        desc = re.sub(r"\b(with|show|showing|appearing|that|and\s+show|and\s+with)\b", "", desc, flags=re.I)
        desc = re.sub(r"\s+", " ", desc).strip()
        _set_field_safe(out, "Colony Morphology", desc.title())

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders (LLM optional — regex handles most)
# ──────────────────────────────────────────────────────────────────────────────
def summarize_field_categories(db_fields: List[str]) -> Dict[str, List[str]]:
    # (redeclared because we need it for prompts; keep identical)
    cats = {"Morphology": [], "Enzyme": [], "Fermentation": [], "Other": []}
    for f in normalize_columns(db_fields):
        n = f.strip()
        l = n.lower()
        if any(k in l for k in ["gram", "shape", "morphology", "motility", "capsule", "spore", "oxygen requirement", "media grown"]):
            cats["Morphology"].append(n)
        elif any(k in l for k in [
            "oxidase","catalase","urease","coagulase","lipase","indole",
            "citrate","vp","methyl red","gelatin","dnase","nitrate","h2s","esculin"
        ]):
            cats["Enzyme"].append(n)
        elif "fermentation" in l or "utilization" in l:
            cats["Fermentation"].append(n)
        else:
            cats["Other"].append(n)
    return cats

def build_prompt(user_text: str, cats: Dict[str, List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    morph = ", ".join(cats["Morphology"][:10])
    enz = ", ".join(cats["Enzyme"][:10])
    ferm = ", ".join(cats["Fermentation"][:10])
    other = ", ".join(cats["Other"][:10])
    system = (
        "You parse microbiology observations into structured results. "
        "Focus on morphology, enzyme, and growth traits. Fermentations are handled by rules. "
        "Return JSON; unmentioned fields='Unknown'.\n"
        f"Morphology: {morph}\nEnzyme: {enz}\nFermentation: {ferm}\nOther: {other}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Previous facts:\n{prior}\nObservation:\n{user_text}"},
    ]

def build_prompt_text(user_text: str, cats: Dict[str, List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    return (
        "Extract morphology, enzyme and growth results from this description. "
        "Leave fermentation mapping to regex rules. "
        "Return JSON; unmentioned fields='Unknown'.\n\n"
        f"Previous facts:\n{prior}\nObservation:\n{user_text}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Normalization to your schema (drop unknowns, bridge haemolysis with logic)
# ──────────────────────────────────────────────────────────────────────────────
def normalize_to_schema(parsed: Dict[str, str], db_fields: List[str]) -> Dict[str, str]:
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)
    out: Dict[str, str] = {}

    # Map keys to exact columns and clamp values
    for k, v in (parsed or {}).items():
        kk = k.strip()
        key_l = kk.lower()
        target = None

        if kk in fields:
            target = kk
        elif key_l in alias:
            target = alias[key_l]

        if target in fields:
            cv = _canon_value(target, v)
            if cv not in ("", None, "Unknown"):
                out[target] = cv

    # Haemolysis Type → Haemolysis with sign logic
    ht = alias.get("haemolysis type")
    h = alias.get("haemolysis")
    if ht in out and h in fields:
        tval = out.get(ht, "")
        if tval in {"Alpha", "Beta"}:
            out[h] = "Positive"
        elif tval in {"Gamma", "None"}:
            out[h] = "Negative"

    # Clamp Media Grown On to your spellings when possible (keep others title-cased)
    if "Media Grown On" in out and out["Media Grown On"]:
        parts = [p.strip() for p in out["Media Grown On"].split(";") if p.strip()]
        fixed = []
        for p in parts:
            # normalize spacing/case
            pt = p.strip()
            # if whitelisted, keep canonical
            if pt in MEDIA_WHITELIST:
                fixed.append(pt)
            else:
                # try case-insensitive match to whitelist
                match = next((m for m in MEDIA_WHITELIST if m.lower() == pt.lower()), None)
                fixed.append(match if match else pt)
        out["Media Grown On"] = "; ".join(dict.fromkeys(fixed))  # dedupe preserve order

    return out


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def parse_input_free_text(
    user_text: str,
    prior_facts: Dict | None = None,
    db_fields: List[str] | None = None,
) -> Dict:
    if not user_text.strip():
        return {}

    db_fields = db_fields or []
    cats = summarize_field_categories(db_fields)

    # Step 1: (optional) LLM parse — regex will enrich heavily either way
    try:
        model_choice = os.getenv("BACTAI_MODEL", "local").lower()
        if model_choice == "gpt":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            messages = build_prompt(user_text, cats, prior_facts)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            llm_parsed = json.loads(resp.choices[0].message.content)
        else:
            import ollama
            prompt = build_prompt_text(user_text, cats, prior_facts)
            out = ollama.chat(
                model=os.getenv("LOCAL_MODEL", "llama3"),
                messages=[{"role": "user", "content": prompt}],
            )
            m = re.search(r"\{.*\}", out["message"]["content"], re.S)
            llm_parsed = json.loads(m.group(0)) if m else {}
    except Exception as e:
        print("⚠️ LLM parser failed — fallback:", e)
        llm_parsed = fallback_parser(user_text, prior_facts)

    # Step 2: Regex enrichment to exact columns
    regex_ferm = extract_fermentations_regex(user_text, db_fields)
    regex_bio = extract_biochem_regex(user_text, db_fields)

    merged = {}
    merged.update(llm_parsed or {})
    merged.update(regex_ferm)
    merged.update(regex_bio)

    # Step 3: Normalize to your schema & clamp values
    normalized = normalize_to_schema(merged, db_fields)
    return normalized
