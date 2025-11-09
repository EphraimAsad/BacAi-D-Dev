# parser_llm.py — Schema-clamped ultra build (TSI-only exclusion)
# - Unicode/whitespace normalization
# - Robust fermentation parsing (incl. "but not … or/nor …" + fallback sweep)
# - ONPG / NaCl tolerant / growth temp / Gram
# - Media detection with **TSI-only exclusion**
# - Colony morphology phrase capture
# - Oxygen requirement labels (Facultative Anaerobe, Aerobic, Anaerobic, Microaerophilic, Capnophilic, Intracellular)
# - Decarboxylases / dihydrolase
# - Haemolysis bridge: Type→Haemolysis (Alpha/Beta=Positive, Gamma/None=Negative)  ← change here if you prefer
# - Clamp all values to your exact Excel spellings (with synonyms)
# - Alias fixes (e.g., "Glucose Fermantation" → "Glucose Fermentation")

import os, json, re
from typing import Dict, List, Set
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
    "Media Grown On": set(),      # parsed + clamped when possible
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

# Media whitelist (from your list). We accept any "... Agar" names,
# but keep these spellings when matched.
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

# synonyms to clamp free-text to your exact spellings
VALUE_SYNONYMS: Dict[str, Dict[str, str]] = {
    "Gram Stain": {
        "gram positive": "Positive", "gram-positive": "Positive", "g+": "Positive",
        "gram negative": "Negative", "gram-negative": "Negative", "g-": "Negative",
        "variable": "Variable"
    },
    "Shape": {
        "rod": "Rods", "rods": "Rods", "bacillus": "Bacilli", "bacilli": "Bacilli",
        "coccus": "Cocci", "cocci": "Cocci", "spiral": "Spiral", "short rods": "Short Rods"
    },
    "Oxygen Requirement": {
        "facultative": "Facultative Anaerobe", "facultative anaerobe": "Facultative Anaerobe",
        "facultative aerobe": "Facultative Anaerobe",
        "aerobe": "Aerobic", "aerobic": "Aerobic",
        "anaerobe": "Anaerobic", "anaerobic": "Anaerobic",
        "microaerophile": "Microaerophilic", "microaerophilic": "Microaerophilic",
        "capnophile": "Capnophilic", "capnophilic": "Capnophilic",
        "intracellular": "Intracellular"
    },
    "Haemolysis Type": {
        "beta": "Beta", "β": "Beta", "alpha": "Alpha", "α": "Alpha",
        "gamma": "Gamma", "γ": "Gamma", "none": "None"
    },
    "*POLARITY*": {
        "+": "Positive", "positive": "Positive", "pos": "Positive",
        "-": "Negative", "negative": "Negative", "neg": "Negative",
        "weakly positive": "Variable", "variable": "Variable", "weak": "Variable"
    },
}

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
    return [f for f in db_fields if f and f.strip().lower() != "genus"]

def _summarize_field_categories(db_fields: List[str]) -> Dict[str, List[str]]:
    cats = {"Morphology": [], "Enzyme": [], "Fermentation": [], "Other": []}
    for f in normalize_columns(db_fields):
        n = f.strip(); l = n.lower()
        if any(k in l for k in ["gram", "shape", "morphology", "motility", "capsule", "spore", "oxygen requirement", "media grown"]):
            cats["Morphology"].append(n)
        elif any(k in l for k in ["oxidase","catalase","urease","coagulase","lipase","indole","citrate","vp","methyl red","gelatin","dnase","nitrate","h2s","esculin"]):
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
    exact = {f.lower(): f for f in normalize_columns(db_fields)}
    alias: Dict[str, str] = {}

    def add(a: str, target: str):
        t = target.lower()
        if t in exact:
            alias[a.lower()] = exact[t]

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
    add("glucose fermantation", "Glucose Fermentation")  # typo → canonical

    # Fermentation bases
    for f in normalize_columns(db_fields):
        if f.lower().endswith(" fermentation"):
            base = f[:-12].strip().lower()
            alias[base] = f

    # Media whitelist names map to "Media Grown On" key (value will be the joined list)
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
    v = (value or "").strip()
    if not v:
        return v

    # polarity fields
    if field in POLARITY_FIELDS:
        low = v.lower()
        if low in VALUE_SYNONYMS.get("*POLARITY*", {}):
            v = VALUE_SYNONYMS["*POLARITY*"][low]
        else:
            if re.fullmatch(r"\+|positive|pos", low): v = "Positive"
            elif re.fullmatch(r"\-|negative|neg", low): v = "Negative"
            elif "weak" in low or "variable" in low: v = "Variable"

    # field-specific synonyms
    low = v.lower()
    if field in VALUE_SYNONYMS:
        v = VALUE_SYNONYMS[field].get(low, v)

    # clamp to allowed if defined
    allowed = ALLOWED_VALUES.get(field)
    if allowed and v not in allowed:
        tv = v.title()
        if tv in allowed:
            v = tv
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

    # POSITIVE lists
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

    # NEGATIVE after "but not …"
    for m in re.finditer(r"(?:ferments|utilizes)[^.]*?\bbut\s+not\s+([\w\s,;.&-]+)", t, flags=re.I):
        seg = m.group(1)
        seg = re.sub(r"\bor\b", ",", seg, flags=re.I)
        seg = re.sub(r"\bnor\b", ",", seg, flags=re.I)
        for a in _tokenize_list(seg):
            a = re.sub(r"[.,;:\s]+$", "", a)
            set_field_by_base(a, "Negative")
        seg_l = " " + seg.lower() + " "
        for base in base_to_field.keys():
            if re.search(rf"\b{re.escape(base)}\b", seg_l, flags=re.I):
                set_field_by_base(base, "Negative")

    # shorthand "+/-"
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
# Regex enrichment: morphology / enzymes / haemolysis / oxygen / growth / media / colony
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
    if re.search(r"\bgram[-\s]?positive\b", t, flags=re.I) and not re.search(r"\bgram[-\s]?negative\b", t, flags=re.I):
        set_field("gram stain", "Positive")
    elif re.search(r"\bgram[-\s]?negative\b", t, flags=re.I) and not re.search(r"\bgram[-\s]?positive\b", t, flags=re.I):
        set_field("gram stain", "Negative")

    # Generic tests
    for test in ["catalase","oxidase","coagulase","urease","lipase","indole","citrate","vp","methyl red","gelatin","dnase","nitrate reduction","nitrate","h2s","esculin hydrolysis"]:
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:\+|positive|detected)\b", t, flags=re.I):
            set_field(test, "Positive")
        elif re.search(rf"\b{test}\s*(?:test)?\s*(?:\-|negative|not\s+detected)\b", t, flags=re.I):
            set_field(test, "Negative")
        elif re.search(rf"\b{test}\s*(?:test)?\s*weak(?:ly)?\s*positive\b", t, flags=re.I):
            set_field(test, "Variable")

    # Nitrate phrasing
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

    # Capsule
    if re.search(r"\b(capsulated|encapsulated)\b", t, flags=re.I):
        set_field("capsule", "Positive")
    if re.search(r"\bnon[-\s]?capsulated\b", t, flags=re.I):
        set_field("capsule", "Negative")

    # Haemolysis type (and "no haemolysis" → Gamma)
    if re.search(r"\b(beta|β)[-\s]?haem", t, flags=re.I):
        set_field("haemolysis type", "Beta")
    elif re.search(r"\b(alpha|α)[-\s]?haem", t, flags=re.I):
        set_field("haemolysis type", "Alpha")
    elif re.search(r"\b(gamma|γ)[-\s]?haem", t, flags=re.I) or re.search(r"\bno\s+haemolysis\b", t, flags=re.I):
        set_field("haemolysis type", "Gamma")

    # Oxygen requirement
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

    # Growth Temperature: only "grows at"
    for m in re.finditer(r"(?<!no\s)grows\s+(?:well\s+)?at\s+([0-9]{2,3})\s*°?\s*c", t, flags=re.I):
        set_field("growth temperature", m.group(1))

    # Media detection — **TSI-only exclusion**
    diagnostic_exclude = ["triple sugar iron", "tsi"]
    media_hits = re.findall(r"\b([a-z0-9\-\+]+)\s+agar\b", t, flags=re.I)
    collected_media: List[str] = []
    for mname in media_hits:
        lowname = mname.lower()
        if any(ex in lowname for ex in diagnostic_exclude):
            continue
        name = mname.strip().upper()
        if name in {"XLD", "MACCONKEY", "BLOOD"}:
            pretty = "XLD Agar" if name == "XLD" else "MacConkey Agar" if name == "MACCONKEY" else "Blood Agar"
        else:
            pretty = mname.strip().title() + " Agar"
        # keep whitelist spellings when applicable
        canon = next((w for w in MEDIA_WHITELIST if w.lower() == pretty.lower()), pretty)
        if canon not in collected_media:
            collected_media.append(canon)
    if collected_media:
        _set_field_safe(out, "Media Grown On", "; ".join(collected_media))

    # Colony morphology phrase
    col_match = re.search(r"colon(?:y|ies)\s+(?:are|appear)\s+([^.]+?)(?:\s+on|\.)", t, flags=re.I)
    if col_match:
        desc = re.sub(r"\b(with|show|showing|appearing|that|and\s+show|and\s+with)\b", "", col_match.group(1).strip(), flags=re.I)
        desc = re.sub(r"\s+", " ", desc).strip()
        _set_field_safe(out, "Colony Morphology", desc.title())

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders (LLM optional)
# ──────────────────────────────────────────────────────────────────────────────
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
# Normalization & Haemolysis bridge
# ──────────────────────────────────────────────────────────────────────────────
def normalize_to_schema(parsed: Dict[str, str], db_fields: List[str]) -> Dict[str, str]:
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)
    out: Dict[str, str] = {}

    for k, v in (parsed or {}).items():
        kk = k.strip(); key_l = kk.lower(); target = None
        if kk in fields:
            target = kk
        elif key_l in alias:
            target = alias[key_l]
        if target in fields:
            cv = _canon_value(target, v)
            if cv not in ("", None, "Unknown"):
                out[target] = cv

    # Bridge Haemolysis Type → Haemolysis (adjust here if you prefer a different mapping)
    ht = alias.get("haemolysis type"); h = alias.get("haemolysis")
    if ht in out and h in fields:
        tval = out.get(ht, "")
        if tval in {"Alpha", "Beta"}:
            out[h] = "Positive"
        elif tval in {"Gamma", "None"}:
            out[h] = "Negative"

    # Clamp media spellings when possible
    if "Media Grown On" in out and out["Media Grown On"]:
        parts = [p.strip() for p in out["Media Grown On"].split(";") if p.strip()]
        fixed = []
        for p in parts:
            match = next((m for m in MEDIA_WHITELIST if m.lower() == p.lower()), p)
            fixed.append(match)
        # de-dupe preserving order
        seen = set(); ordered = []
        for x in fixed:
            if x not in seen:
                ordered.append(x); seen.add(x)
        out["Media Grown On"] = "; ".join(ordered)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def parse_input_free_text(user_text: str, prior_facts: Dict | None = None, db_fields: List[str] | None = None) -> Dict:
    if not user_text.strip():
        return {}

    db_fields = db_fields or []
    cats = _summarize_field_categories(db_fields)

    # Optional LLM parse (regex will enrich regardless)
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
            out = ollama.chat(model=os.getenv("LOCAL_MODEL", "llama3"),
                              messages=[{"role": "user", "content": prompt}])
            m = re.search(r"\{.*\}", out["message"]["content"], re.S)
            llm_parsed = json.loads(m.group(0)) if m else {}
    except Exception as e:
        print("⚠️ LLM parser failed — fallback:", e)
        llm_parsed = fallback_parser(user_text, prior_facts)

    # Regex enrichment
    regex_ferm = extract_fermentations_regex(user_text, db_fields)
    regex_bio = extract_biochem_regex(user_text, db_fields)

    # Merge → normalize
    merged = {}
    merged.update(llm_parsed or {})
    merged.update(regex_ferm)
    merged.update(regex_bio)

    return normalize_to_schema(merged, db_fields)
