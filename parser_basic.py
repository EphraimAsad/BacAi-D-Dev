# parser_basic.py — v3 (Deterministic fallback parser + Gold Spec tests + What-If support)
# ---------------------------------------------------------------------------------------
# Design goals
# - Pure-regex deterministic parser for microbiology observations (no LLM dependency).
# - Mirrors the schema, normalization, and behavior used in parser_llm.py v3.
# - Handles morphology, biochemical tests, fermentations, haemolysis, media, colony morphology,
#   oxygen requirements, capsule/spore, temperature ranges, NaCl tolerance, ONPG, etc.
# - Negation scope window (±5 tokens), variable heuristics ("weak", "variable", "trace"),
#   and conjunction splitting for "and/or/nor/but not …".
# - Abbreviation support (MR, VP, LDC/ODC/ADH, NLF/LF, TSA/BHI/CBA/SSA/BA, etc.).
# - TSI-only exclusion for media (future-proof to accept new media names).
# - Exposes: parse_input_free_text(), normalize_to_schema(), apply_what_if(),
#            run_gold_tests(), GOLD_SPEC (test set).
#
# Usage:
#   from parser_basic import parse_input_free_text
#   parsed = parse_input_free_text("Gram-negative rod ...", db_fields=list_of_excel_columns)
#
# CLI:
#   python parser_basic.py --test       # run Gold Spec tests
#   python parser_basic.py --demo "..." # parse one paragraph and print JSON
#
# ---------------------------------------------------------------------------------------

import re, json, sys
from typing import Dict, List, Set, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Allowed values & canonicalization (aligned with your Excel)
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_VALUES: Dict[str, Set[str]] = {
    "Gram Stain": {"Positive", "Negative", "Variable"},
    "Shape": {"Cocci", "Rods", "Bacilli", "Spiral", "Short Rods"},
    "Catalase": {"Positive", "Negative", "Variable"},
    "Oxidase": {"Positive", "Negative", "Variable"},
    "Colony Morphology": set(),  # free text (normalized by vocabulary below)
    "Haemolysis": {"Positive", "Negative", "Variable"},
    "Haemolysis Type": {"None", "Beta", "Gamma", "Alpha"},
    "Indole": {"Positive", "Negative", "Variable"},
    "Growth Temperature": set(),               # numeric (°C) — compared to DB low//high
    "Media Grown On": set(),                   # parsed & clamped (TSI excluded)
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

# Whitelisted media (future-proof — we still accept others; we only exclude TSI)
MEDIA_WHITELIST = {
    "MacConkey Agar","Nutrient Agar","ALOA","Palcam","Preston","Columbia","BP",
    "Mannitol Salt Agar","MRS","Anaerobic Media","XLD Agar","TBG","TCBS","VID",
    "EMB Agar","CCI","Salt Nutrient Agar","Thayer Martin Agar","Tryptic Soy Agar",
    "Chocolate Agar","Bacteroides Bile Esculin Agar","KVLB Agar","Charcoal Blood Agar",
    "Anaerobic Blood Agar","Yeast Extract Mannitol Agar","Burks Medium","Peptone Water",
    "Sabouraud Dextrose Agar","Yeast Extract Peptone Dextrose","Malt Extract Agar",
    "Middlebrook Agar","Inorganic Mineral Nitrate Media","Inorganic Mineral Ammonia Media",
    "Iron Media","Sulfur Media","Organic Media","Yeast Extract Agar","Cellulose Agar",
    "Baciillus Media","Pyridoxal","Lcysteine","Ferrous Sulfate Media","Hayflicks Agar",
    "Cell Culture","Intracellular","Brain Heart Infusion Agar","Human Fibroblast Cell Culture","BCYE Agar"
}

# Value synonyms and abbreviations → canonical spellings
VALUE_SYNONYMS: Dict[str, Dict[str, str]] = {
    "Gram Stain": {
        "gram positive": "Positive","gram-positive": "Positive","g+": "Positive",
        "gram negative": "Negative","gram-negative": "Negative","g-": "Negative",
        "variable": "Variable"
    },
    "Shape": {
        "rod": "Rods","rods": "Rods","bacillus": "Bacilli","bacilli": "Bacilli",
        "coccus": "Cocci","cocci": "Cocci","spiral": "Spiral","short rods": "Short Rods"
    },
    "Oxygen Requirement": {
        "facultative": "Facultative Anaerobe","facultative anaerobe": "Facultative Anaerobe",
        "facultative anaerobic": "Facultative Anaerobe","facultative aerobe": "Facultative Anaerobe",
        "aerobe": "Aerobic","aerobic": "Aerobic","obligate aerobe": "Aerobic",
        "anaerobe": "Anaerobic","anaerobic": "Anaerobic","obligate anaerobe": "Anaerobic",
        "microaerophile": "Microaerophilic","microaerophilic": "Microaerophilic",
        "capnophile": "Capnophilic","capnophilic": "Capnophilic",
        "intracellular": "Intracellular"
    },
    "Haemolysis Type": {
        "beta": "Beta","β": "Beta","alpha": "Alpha","α": "Alpha",
        "gamma": "Gamma","γ": "Gamma","none": "None"
    },
    "*POLARITY*": {
        "+": "Positive","positive": "Positive","pos": "Positive",
        "-": "Negative","negative": "Negative","neg": "Negative",
        "weakly positive": "Variable","variable": "Variable","weak": "Variable","trace": "Variable"
    },
}

# Polarity fields list (where Positive/Negative/Variable apply)
POLARITY_FIELDS = {
    "Catalase","Oxidase","Haemolysis","Indole","Motility","Capsule","Spore Formation",
    "Methyl Red","VP","Citrate","Urease","H2S","Lactose Fermentation","Glucose Fermentation",
    "Sucrose Fermentation","Nitrate Reduction","Lysine Decarboxylase","Ornitihine Decarboxylase",
    "Arginine dihydrolase","Gelatin Hydrolysis","Esculin Hydrolysis","Dnase","ONPG",
    "NaCl Tolerant (>=6%)","Lipase Test","Xylose Fermentation","Rhamnose Fermentation",
    "Mannitol Fermentation","Sorbitol Fermentation","Maltose Fermentation","Arabinose Fermentation",
    "Raffinose Fermentation","Inositol Fermentation","Trehalose Fermentation","Coagulase"
}

# Abbreviation → canonical field name mapping (keys: lowercase)
ABBREV_TO_FIELD = {
    "mr": "Methyl Red",
    "vp": "VP",
    "ldc": "Lysine Decarboxylase",
    "odc": "Ornitihine Decarboxylase",  # sheet uses "Ornitihine"
    "adh": "Arginine dihydrolase",
    "nlf": "Lactose Fermentation",      # Non-lactose fermenter → Negative
    "lf": "Lactose Fermentation",
}

# Media abbreviations → canonical media name
MEDIA_ABBREV = {
    "tsa": "Tryptic Soy Agar",
    "bhi": "Brain Heart Infusion Agar",
    "cba": "Columbia",
    "ssa": "Blood Agar", "ba": "Blood Agar",
}

# Colony Morphology vocabulary (tokens we normalize & keep)
CM_TOKENS = {
    # sizes & measurements
    "1/3mm","1/2mm","2/3mm","2/4mm","0.5/1mm","0.5mm/2mm","1mm","2mm","3mm","small","medium","large","tiny","pinpoint","subsurface","satellite",
    # shapes/profile
    "round","circular","convex","flat","domed","heaped","fried egg",
    # edges/surface/texture
    "smooth","rough","wrinkled","granular","mucoid","glistening","dull","matte","shiny","sticky","adherent","powdery","chalk","leathery","velvet","crumbly",
    "ground glass","irregular edges","spreading","swarming","corrode","pit",
    # opacity/transparency
    "opaque","translucent","colourless","colorless",
    # moisture
    "dry","moist",
    # colours
    "white","grey","gray","cream","yellow","orange","pink","coral","red","green","violet","purple","black","brown","beige","tan","blue",
    # extras
    "bright","pigmented","waxy","ring","dingers ring"
}

# ──────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ──────────────────────────────────────────────────────────────────────────────
def normalize_columns(db_fields: List[str]) -> List[str]:
    """Keep everything except 'Genus' (case-insensitive) and return in original case."""
    return [f for f in db_fields if f and f.strip().lower() != "genus"]

def build_alias_map(db_fields: List[str]) -> Dict[str, str]:
    """Common phrases → exact Excel columns. Keys are lowercase."""
    fields = normalize_columns(db_fields)
    exact = {f.lower(): f for f in fields}
    alias: Dict[str, str] = {}

    def add(a: str, target: str):
        t = target.lower()
        if t in exact:
            alias[a.lower()] = exact[t]

    # Canonical tests
    add("mr","Methyl Red"); add("methyl red","Methyl Red")
    add("vp","VP"); add("voges proskauer","VP")
    add("h2s","H2S"); add("dnase","Dnase")
    add("gelatin","Gelatin Hydrolysis"); add("gelatin liquefaction","Gelatin Hydrolysis")
    add("lipase","Lipase Test"); add("lipase test","Lipase Test")
    add("onpg","ONPG"); add("onpg test","ONPG"); add("esculin hydrolysis","Esculin Hydrolysis")
    add("nacl tolerance","NaCl Tolerant (>=6%)"); add("nacl tolerant","NaCl Tolerant (>=6%)"); add("nacl","NaCl Tolerant (>=6%)")
    add("nitrate","Nitrate Reduction"); add("nitrate reduction","Nitrate Reduction")
    add("lysine decarboxylase","Lysine Decarboxylase")
    add("ornithine decarboxylase","Ornitihine Decarboxylase"); add("ornitihine decarboxylase","Ornitihine Decarboxylase")
    add("arginine dihydrolase","Arginine dihydrolase")
    add("coagulase","Coagulase"); add("citrate","Citrate")
    add("urease","Urease"); add("indole","Indole")
    add("oxidase","Oxidase"); add("catalase","Catalase")
    add("motility","Motility"); add("capsule","Capsule")
    add("spore formation","Spore Formation")
    add("haemolysis","Haemolysis"); add("haemolysis type","Haemolysis Type")
    add("growth temperature","Growth Temperature")
    add("media grown on","Media Grown On")
    add("oxygen requirement","Oxygen Requirement")
    add("gram stain","Gram Stain"); add("shape","Shape")
    # Your sheet typo → canonical
    add("glucose fermantation","Glucose Fermentation")

    # Fermentation bases (e.g., "rhamnose" → "Rhamnose Fermentation")
    for f in fields:
        if f.lower().endswith(" fermentation"):
            base = f[:-12].strip().lower()
            alias[base] = f

    # Whitelisted media map to "Media Grown On" (we store values; key stays that column)
    for m in MEDIA_WHITELIST:
        alias[m.lower()] = "Media Grown On"

    # Abbrev media
    for k, v in MEDIA_ABBREV.items():
        alias[k.lower()] = "Media Grown On"

    return alias

# ──────────────────────────────────────────────────────────────────────────────
# Text & token helpers
# ──────────────────────────────────────────────────────────────────────────────
_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def normalize_text(raw: str) -> str:
    """Unicode cleanup + lowercasing + spacing for consistent regex behavior."""
    t = raw or ""
    t = t.replace("°", " °")
    t = t.translate(_SUBSCRIPT_DIGITS)
    t = (t.replace("\u2010","-").replace("\u2011","-").replace("\u2012","-")
           .replace("\u2013","-").replace("\u2014","-"))
    t = re.sub(r"hemolys","haemolys", t, flags=re.I)  # normalize US/UK spelling
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

def _windows(tokens: List[str], idx: int, span: int = 5) -> List[str]:
    """Return a token window ±span around idx (bounded)."""
    lo = max(0, idx - span)
    hi = min(len(tokens), idx + span + 1)
    return tokens[lo:hi]

def _has_negation_near(tokens: List[str], idx: int, span: int = 5) -> bool:
    """Detect negation cues near a token index."""
    window = " ".join(_windows(tokens, idx, span))
    return bool(re.search(r"\b(no|not|absent|without|lack|lacks|did\s+not|does\s+not|cannot|negative|not\s+observed|no\s+growth|no\s+reaction|not\s+produced)\b", window))

def _has_variable_near(tokens: List[str], idx: int, span: int = 5) -> bool:
    """Detect variable/weak cues near a token index."""
    window = " ".join(_windows(tokens, idx, span))
    return bool(re.search(r"\b(variable|weak|weakly|trace|inconsistent|equivocal)\b", window))

# ──────────────────────────────────────────────────────────────────────────────
# Safe setter & canonicalization helpers
# ──────────────────────────────────────────────────────────────────────────────
def _set_field_safe(out: Dict[str, str], key: str, val: str):
    """
    Merge policy:
      - If nothing set -> set.
      - If 'Negative' exists and new is 'Positive' → keep Negative (conservative).
      - If 'Variable' vs 'Positive': prefer Positive (explicit beats uncertain).
      - Else, overwrite (latest mention wins).
    """
    cur = out.get(key)
    if cur is None:
        out[key] = val; return
    if cur == "Negative" and val == "Positive":
        return
    if cur == "Variable" and val == "Positive":
        out[key] = "Positive"; return
    out[key] = val

def _canon_value(field: str, value: str) -> str:
    """Map tokens to canonical allowed values (and tolerate synonyms)."""
    v = (value or "").strip()
    if not v:
        return v

    # polarity fields
    if field in POLARITY_FIELDS:
        low = v.lower()
        syn = VALUE_SYNONYMS.get("*POLARITY*", {})
        if low in syn:
            v = syn[low]
        else:
            if re.fullmatch(r"\+|positive|pos", low): v = "Positive"
            elif re.fullmatch(r"\-|negative|neg", low): v = "Negative"
            elif "weak" in low or "variable" in low or "trace" in low: v = "Variable"

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
# Colony morphology normalization
# ──────────────────────────────────────────────────────────────────────────────
def normalize_cm_phrase(text: str) -> str:
    """Extract and normalize colony morphology tokens from free text."""
    t = text.lower()
    # Try to capture an explicit "colonies ..." phrase
    spans = []
    m = re.search(r"colon(?:y|ies)\s+(?:are|appear|were|appearing)\s+([^.]+?)(?:\s+on|\.)", t)
    if m:
        spans.append(m.group(1))
    # Also scan the whole text
    spans.append(t)

    found: List[str] = []
    def add(tok: str):
        tok = tok.strip()
        if not tok: return
        if tok not in found:
            found.append(tok)

    # Capture measurements like "1/2mm", "0.5/1mm"
    for s in spans:
        for mm in re.findall(r"(?:\d+(?:\.\d+)?\/\d+(?:\.\d+)?mm|\d+(?:\.\d+)?mm|0\.5\/1mm|0\.5mm\/2mm|1\/3mm|2\/3mm|2\/4mm)", s):
            add(mm)

        # Rejoin multi-word tokens
        s_norm = " " + re.sub(r"[,;/]", " ", s) + " "
        multi = ["ground glass","irregular edges","fried egg","dingers ring","off-white","pale yellow","cream-white","grey-cream","mucoid ropey","butyrous"]
        for mword in multi:
            if f" {mword} " in s_norm:
                add(mword)

        # Token scan (split broadly)
        parts = re.split(r"[,;:/\-\s]+", s)
        for p in parts:
            low = p.strip().lower()
            if low in {"colorless"}: low = "colourless"
            if low in CM_TOKENS:
                add(low)
            # extra color & texture variants
            if low in {"off-white","pale","pale-yellow","cream-white","grey-cream","ropey","butyrous"}:
                add(low.replace("-", " "))

    # Ordering groups (size → shape → texture → opacity → moisture → color → extras)
    order_groups = [
        {"1/3mm","1/2mm","2/3mm","2/4mm","0.5/1mm","0.5mm/2mm","1mm","2mm","3mm","tiny","small","medium","large","pinpoint","subsurface","satellite"},
        {"round","circular","convex","flat","domed","heaped","fried egg"},
        {"smooth","rough","wrinkled","granular","mucoid","glistening","dull","matte","shiny","sticky","adherent","powdery","chalk","leathery","velvet","crumbly","ground glass","irregular edges","spreading","swarming","corrode","pit","ring","dingers ring","waxy","bright","pigmented","ropey","butyrous"},
        {"opaque","translucent","colourless"},
        {"dry","moist"},
        {"white","grey","gray","cream","yellow","orange","pink","coral","red","green","violet","purple","black","brown","beige","tan","blue"},
        {"off white","pale yellow","cream white","grey cream"},
    ]
    ordered: List[str] = []
    seen = set()
    for grp in order_groups:
        for tok in found:
            if tok in grp and tok not in seen:
                ordered.append(tok); seen.add(tok)
    # leftovers
    for tok in found:
        if tok not in seen:
            ordered.append(tok); seen.add(tok)

    # Title-case; prefer "Grey" over "Gray"
    pretty = [("Grey" if w == "gray" else w.title()) for w in ordered]
    return "; ".join(pretty)

# ──────────────────────────────────────────────────────────────────────────────
# Extraction helpers (deterministic)
# ──────────────────────────────────────────────────────────────────────────────
def extract_fermentations(text: str, db_fields: List[str]) -> Dict[str, str]:
    """Extract sugar fermentations (+/-/variable), including 'but not … (or …)' and shorthands."""
    out: Dict[str, str] = {}
    t = normalize_text(text)
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)

    # Build base → field mapping for fermentation columns
    ferm_fields = [f for f in fields if f.lower().endswith(" fermentation")]
    base_to_field = {f[:-12].strip().lower(): f for f in ferm_fields}

    def set_field_by_base(base: str, val: str):
        b = _normalize_token(base)
        if b in base_to_field:
            _set_field_safe(out, base_to_field[b], _canon_value(base_to_field[b], val))
        elif b in alias and alias[b] in fields:
            _set_field_safe(out, alias[b], _canon_value(alias[b], val))

    # Positive lists (ferments/utilizes X, Y, Z …)
    for m in re.finditer(r"(?:ferments|utilizes)\s+([a-z0-9\.\-%\s,/&]+)", t, flags=re.I):
        # stop at "but not"
        span = re.split(r"(?i)\bbut\s+not\b", m.group(1))[0]
        for a in _tokenize_list(span):
            set_field_by_base(a, "Positive")

    # Explicit negative lists
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

    # "… but not A, B or C" (split “or/nor” to ensure all 3 captured)
    for m in re.finditer(r"(?:ferments|utilizes)[^.]*?\bbut\s+not\s+([\w\s,;.&-]+)", t, flags=re.I):
        seg = m.group(1)
        seg = re.sub(r"\bor\b", ",", seg, flags=re.I)
        seg = re.sub(r"\bnor\b", ",", seg, flags=re.I)
        for a in _tokenize_list(seg):
            a = re.sub(r"[.,;:\s]+$", "", a)
            set_field_by_base(a, "Negative")
        # Fallback sweep: if any fermentation base appears in seg, force Negative
        seg_l = " " + seg.lower() + " "
        for base in base_to_field.keys():
            if re.search(rf"\b{re.escape(base)}\b", seg_l, flags=re.I):
                set_field_by_base(base, "Negative")

    # Shorthand "lactose +/-", "rhamnose +"
    for m in re.finditer(r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b", t, flags=re.I):
        a, sign = m.group(1), m.group(2)
        set_field_by_base(a, "Positive" if sign == "+" else "Negative")

    # LF/NLF on MacConkey context → lactose fermenter/non-fermenter
    if re.search(r"\bnlf\b", t):
        set_field_by_base("lactose", "Negative")
    if re.search(r"\blf\b", t) and not re.search(r"\bnlf\b", t):
        set_field_by_base("lactose", "Positive")

    return out

def extract_biochem(text: str, db_fields: List[str]) -> Dict[str, str]:
    """
    Extract morphology & biochemical tests (deterministic rules with negation/variable windows).
    Handles: Gram, capsule, motility, haemolysis type, ONPG, NaCl tolerant, H2S, nitrate, MR/VP,
             decarboxylases/dihydrolase, growth temperature, media (TSI excluded), colony morphology,
             oxygen labels, coagulase/lipase/urease/indole/citrate/oxidase/catalase/dnase/esculin/gelatin.
    """
    out: Dict[str, str] = {}
    raw = text or ""
    t = normalize_text(raw)
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)

    # Tokenize for window-based negation/variable heuristics
    tokens = t.split()

    def set_field(key_like: str, val: str, at_token_idx: Optional[int] = None):
        """Set field with canonical value; apply variable/negation heuristics if idx is supplied."""
        target = alias.get(key_like.lower(), key_like)
        if target not in fields:
            return
        v = val

        # Window-based polarity overrides
        if at_token_idx is not None:
            if _has_negation_near(tokens, at_token_idx, 5):
                v = "Negative"
            elif _has_variable_near(tokens, at_token_idx, 5) and target in POLARITY_FIELDS:
                v = "Variable"

        _set_field_safe(out, target, _canon_value(target, v))

    # Helper to find token index for a pattern
    def first_token_index_of(pattern: str) -> Optional[int]:
        m = re.search(pattern, t)
        if not m:
            return None
        # approximate index by counting spaces before start
        start = m.start()
        pre = t[:start]
        return len(pre.split())

    # Gram
    if re.search(r"\bgram[-\s]?positive\b", t) and not re.search(r"\bgram[-\s]?negative\b", t):
        idx = first_token_index_of(r"\bgram[-\s]?positive\b")
        set_field("gram stain", "Positive", idx)
    elif re.search(r"\bgram[-\s]?negative\b", t) and not re.search(r"\bgram[-\s]?positive\b", t):
        idx = first_token_index_of(r"\bgram[-\s]?negative\b")
        set_field("gram stain", "Negative", idx)

    # Shape
    if re.search(r"\bcocci?\b", t):
        idx = first_token_index_of(r"\bcocci?\b")
        set_field("shape", "Cocci", idx)
    elif re.search(r"\brods?\b", t):
        idx = first_token_index_of(r"\brods?\b")
        set_field("shape", "Rods", idx)
    elif re.search(r"\bbacilli?\b", t):
        idx = first_token_index_of(r"\bbacilli?\b")
        set_field("shape", "Bacilli", idx)
    elif re.search(r"\bspiral\b", t):
        idx = first_token_index_of(r"\bspiral\b")
        set_field("shape", "Spiral", idx)
    elif re.search(r"\bshort\s+rods?\b", t):
        idx = first_token_index_of(r"\bshort\s+rods?\b")
        set_field("shape", "Short Rods", idx)

    # Capsule
    if re.search(r"\b(encapsulated|capsulated)\b", t):
        idx = first_token_index_of(r"\b(encapsulated|capsulated)\b")
        set_field("capsule", "Positive", idx)
    if re.search(r"\bnon[-\s]?capsulated\b", t):
        idx = first_token_index_of(r"\bnon[-\s]?capsulated\b")
        set_field("capsule", "Negative", idx)

    # Motility
    if re.search(r"\bmotile\b", t):
        idx = first_token_index_of(r"\bmotile\b")
        set_field("motility", "Positive", idx)
    if re.search(r"\bnon[-\s]?motile\b", t):
        idx = first_token_index_of(r"\bnon[-\s]?motile\b")
        set_field("motility", "Negative", idx)

    # Spore formation
    if re.search(r"\bspore[-\s]?forming\b", t):
        idx = first_token_index_of(r"\bspore[-\s]?forming\b")
        set_field("spore formation", "Positive", idx)
    if re.search(r"\bnon[-\s]?spore[-\s]?forming\b", t):
        idx = first_token_index_of(r"\bnon[-\s]?spore[-\s]?forming\b")
        set_field("spore formation", "Negative", idx)

    # Oxygen requirement labels
    if re.search(r"\bfacultative\b", t) or re.search(r"\bfacultative\s+anaerob", t):
        idx = first_token_index_of(r"\bfacultative\b")
        set_field("oxygen requirement", "Facultative Anaerobe", idx)
    elif re.search(r"\baerobic\b", t):
        idx = first_token_index_of(r"\baerobic\b")
        set_field("oxygen requirement", "Aerobic", idx)
    elif re.search(r"\banaerobic\b", t):
        idx = first_token_index_of(r"\banaerobic\b")
        set_field("oxygen requirement", "Anaerobic", idx)
    elif re.search(r"\bmicroaerophil(ic|e)\b", t):
        idx = first_token_index_of(r"\bmicroaerophil(ic|e)\b")
        set_field("oxygen requirement", "Microaerophilic", idx)
    elif re.search(r"\bcapnophil(ic|e)\b", t):
        idx = first_token_index_of(r"\bcapnophil(ic|e)\b")
        set_field("oxygen requirement", "Capnophilic", idx)
    elif re.search(r"\bintracellular\b", t):
        idx = first_token_index_of(r"\bintracellular\b")
        set_field("oxygen requirement", "Intracellular", idx)

    # Generic biochemical tests (with windowed negation/variable)
    generic_tests = ["catalase","oxidase","coagulase","urease","lipase","indole","citrate","vp","methyl red","gelatin","dnase","nitrate reduction","nitrate","h2s","esculin hydrolysis","onpg"]
    for test in generic_tests:
        # Positive
        m = re.search(rf"\b{test}\s*(?:test)?\s*(?:\+|positive|produced|detected)\b", t)
        if m:
            idx = len(t[:m.start()].split())
            set_field(test, "Positive", idx)
        # Negative
        m = re.search(rf"\b{test}\s*(?:test)?\s*(?:\-|negative|not\s+produced|not\s+detected|absent)\b", t)
        if m:
            idx = len(t[:m.start()].split())
            set_field(test, "Negative", idx)
        # Variable/weak
        m = re.search(rf"\b{test}\s*(?:test)?\s*(?:weak|weakly\s+positive|variable|trace)\b", t)
        if m:
            idx = len(t[:m.start()].split())
            set_field(test, "Variable", idx)

    # Nitrate “reduces nitrate”
    if re.search(r"\breduces\s+nitrate\b", t):
        idx = first_token_index_of(r"\breduces\s+nitrate\b")
        set_field("nitrate", "Positive", idx)
    if re.search(r"\bdoes\s+not\s+reduce\s+nitrate\b", t):
        idx = first_token_index_of(r"\bdoes\s+not\s+reduce\s+nitrate\b")
        set_field("nitrate", "Negative", idx)

    # H2S precedence
    if re.search(r"\bproduces\s+h\s*2\s*s\s+negative\b", t):
        idx = first_token_index_of(r"\bproduces\s+h\s*2\s*s\s+negative\b")
        set_field("h2s", "Negative", idx)
    if re.search(r"\bh\s*2\s*s\s+(?:\+|positive|detected|produced)\b", t):
        idx = first_token_index_of(r"\bh\s*2\s*s\s+(?:\+|positive|detected|produced)\b")
        set_field("h2s", "Positive", idx)
    if re.search(r"\bh\s*2\s*s\s+(?:\-|negative|not\s+detected|not\s+produced)\b", t):
        idx = first_token_index_of(r"\bh\s*2\s*s\s+(?:\-|negative|not\s+detected|not\s+produced)\b")
        set_field("h2s", "Negative", idx)
    if re.search(r"\bproduces\s+h\s*2\s*s\b", t):
        idx = first_token_index_of(r"\bproduces\s+h\s*2\s*s\b")
        set_field("h2s", "Positive", idx)

    # Haemolysis type (and “no haemolysis” → Gamma type)
    if re.search(r"\b(beta|β)[-\s]?haem", t):
        idx = first_token_index_of(r"\b(beta|β)[-\s]?haem")
        set_field("haemolysis type", "Beta", idx)
    elif re.search(r"\b(alpha|α)[-\s]?haem", t):
        idx = first_token_index_of(r"\b(alpha|α)[-\s]?haem")
        set_field("haemolysis type", "Alpha", idx)
    elif re.search(r"\b(gamma|γ)[-\s]?haem", t) or re.search(r"\bno\s+haemolysis\b", t):
        idx = first_token_index_of(r"\b(gamma|γ)[-\s]?haem|\bno\s+haemolysis\b")
        set_field("haemolysis type", "Gamma", idx)

    # Decarboxylases & dihydrolase (singular/plural)
    decarbox_patterns = [
        ("lysine decarboxylase", r"\blysine\s+decarboxylases?\s+(?:test\s+)?(\+|positive|detected)\b", "Positive"),
        ("lysine decarboxylase", r"\blysine\s+decarboxylases?\s+(?:test\s+)?(\-|negative|not\s+detected)\b", "Negative"),
        ("ornithine decarboxylase", r"\bornithine\s+decarboxylases?\s+(?:test\s+)?(\+|positive|detected)\b", "Positive"),
        ("ornithine decarboxylase", r"\bornithine\s+decarboxylases?\s+(?:test\s+)?(\-|negative|not\s+detected)\b", "Negative"),
        ("ornitihine decarboxylase", r"\bornitihine\s+decarboxylases?\s+(?:test\s+)?(\+|positive|detected)\b", "Positive"),
        ("ornitihine decarboxylase", r"\bornitihine\s+decarboxylases?\s+(?:test\s+)?(\-|negative|not\s+detected)\b", "Negative"),
        ("arginine dihydrolase", r"\barginine\s+dihydrolases?\s+(?:test\s+)?(\+|positive|detected)\b", "Positive"),
        ("arginine dihydrolase", r"\barginine\s+dihydrolases?\s+(?:test\s+)?(\-|negative|not\s+detected)\b", "Negative"),
    ]
    for key, pat, val in decarbox_patterns:
        m = re.search(pat, t)
        if m:
            idx = len(t[:m.start()].split())
            set_field(key, val, idx)

    # Abbrev-only phrases (e.g., "LDC positive")
    for abbr, fname in [("ldc","Lysine Decarboxylase"),("odc","Ornitihine Decarboxylase"),("adh","Arginine dihydrolase")]:
        if re.search(rf"\b{abbr}\s*(?:\+|positive)\b", t):
            idx = first_token_index_of(rf"\b{abbr}\s*(?:\+|positive)\b")
            set_field(fname, "Positive", idx)
        if re.search(rf"\b{abbr}\s*(?:\-|negative)\b", t):
            idx = first_token_index_of(rf"\b{abbr}\s*(?:\-|negative)\b")
            set_field(fname, "Negative", idx)

    # Growth Temperature:
    # - Positive: "grows at 37 °C", "grows well at 45 °C", "growth at 10 °C"
    for m in re.finditer(r"(?<!no\s)grows\s+(?:well\s+)?at\s+([0-9]{1,3})\s*°?\s*c", t):
        set_field("growth temperature", m.group(1))
    for m in re.finditer(r"(?<!no\s)growth\s+at\s+([0-9]{1,3})\s*°?\s*c", t):
        set_field("growth temperature", m.group(1))

    # - Negative mentions we avoid mapping to Growth Temperature (only record positives)
    #   e.g., "no growth at 10 °C" → ignore that temp, only map "grows at 45 °C" if present.

    # NaCl tolerant (>=6%)
    if re.search(r"\b(tolerant|grows|growth)\s+(?:in|up\s+to|to|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t):
        idx = first_token_index_of(r"\b(tolerant|grows|growth)\s+(?:in|up\s+to|to|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b")
        set_field("nacl tolerant (>=6%)", "Positive", idx)
    if re.search(r"\bno\s+growth\s+(?:in|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t):
        idx = first_token_index_of(r"\bno\s+growth\s+(?:in|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b")
        set_field("nacl tolerant (>=6%)", "Negative", idx)
    if re.search(r"\bnacl\s+tolerant\b", t):
        idx = first_token_index_of(r"\bnacl\s+tolerant\b")
        set_field("nacl tolerant (>=6%)", "Positive", idx)

    # Media detection — exclude TSI/triple sugar iron
    diag_exclude = ["triple sugar iron", "tsi"]
    # explicit “… on <MEDIA> agar”
    media_hits = re.findall(r"\bon\s+([a-z0-9\-\+ ]+?)\s+agar\b", t)
    collected_media: List[str] = []
    for mname in media_hits:
        lowname = mname.strip().lower()
        if any(ex in lowname for ex in diag_exclude):
            continue
        # abbreviations
        up = lowname.upper().replace(" ", "")
        if up in {"XLD"}:
            pretty = "XLD Agar"
        elif up in {"MACCONKEY","MAC"}:
            pretty = "MacConkey Agar"
        elif up in {"BLOOD","BA","SSA"}:
            pretty = "Blood Agar"
        else:
            pretty = mname.strip().title() + " Agar"
        # clamp to whitelist if present
        canon = next((w for w in MEDIA_WHITELIST if w.lower() == pretty.lower()), pretty)
        if canon not in collected_media:
            collected_media.append(canon)
    # also accept bare mentions like "on XLD" / "on MacConkey"
    bare_media = re.findall(r"\bon\s+(xld|macconkey|blood|tsa|bhi|cba)\b", t)
    for bm in bare_media:
        key = bm.lower()
        pretty = MEDIA_ABBREV.get(key, None)
        if pretty is None:
            if key == "xld": pretty = "XLD Agar"
            elif key == "macconkey": pretty = "MacConkey Agar"
            elif key == "blood": pretty = "Blood Agar"
        if pretty and pretty not in collected_media:
            collected_media.append(pretty)

    if collected_media:
        set_field("media grown on", "; ".join(collected_media))

    # Colony morphology from vocabulary
    cm_value = normalize_cm_phrase(raw)
    if cm_value:
        set_field("colony morphology", cm_value)

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Normalization & Haemolysis bridge to your schema
# ──────────────────────────────────────────────────────────────────────────────
def normalize_to_schema(parsed: Dict[str, str], db_fields: List[str]) -> Dict[str, str]:
    """
    - Map any keys/variants to exact sheet columns (via aliases)
    - Drop keys not present in the sheet
    - Haemolysis Type → Haemolysis (Alpha/Beta=Positive, Gamma/None=Variable per your decision)
    - Clamp Media spellings & dedupe; tidy Colony Morphology.
    """
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

    # Haemolysis Type → Haemolysis
    ht = alias.get("haemolysis type"); h = alias.get("haemolysis")
    if ht in out and h in fields:
        tval = out.get(ht, "")
        if tval in {"Alpha", "Beta"}:
            out[h] = "Positive"
        elif tval in {"Gamma", "None"}:
            out[h] = "Variable"  # You chose to treat Gamma/None as Variable

    # Clamp media spellings and dedupe
    if "Media Grown On" in out and out["Media Grown On"]:
        parts = [p.strip() for p in out["Media Grown On"].split(";") if p.strip()]
        fixed = []
        for p in parts:
            match = next((m for m in MEDIA_WHITELIST if m.lower() == p.lower()), p)
            fixed.append(match)
        seen = set(); ordered = []
        for x in fixed:
            if x not in seen:
                ordered.append(x); seen.add(x)
        out["Media Grown On"] = "; ".join(ordered)

    # Deduplicate Colony Morphology chunks
    if "Colony Morphology" in out and out["Colony Morphology"]:
        chunks = [c.strip() for c in out["Colony Morphology"].split(";") if c.strip()]
        seen = set(); cleaned = []
        for c in chunks:
            if c not in seen:
                cleaned.append(c); seen.add(c)
        out["Colony Morphology"] = "; ".join(cleaned)

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Public API: deterministic parsing (fallback)
# ──────────────────────────────────────────────────────────────────────────────
def parse_input_free_text(user_text: str, prior_facts: Dict | None = None, db_fields: List[str] | None = None) -> Dict:
    """
    Deterministic (non-LLM) parsing:
      1) Extract fermentations
      2) Extract biochem/morph/media/colony/oxygen/temp
      3) Merge and normalize
    """
    if not user_text or not user_text.strip():
        return {}
    db_fields = db_fields or []
    ferm = extract_fermentations(user_text, db_fields)
    bio  = extract_biochem(user_text, db_fields)
    merged = {}
    merged.update(ferm)
    merged.update(bio)
    # Seed with prior facts (they are treated as already-normalized)
    if prior_facts:
        merged.update(prior_facts)
    return normalize_to_schema(merged, db_fields)

# ──────────────────────────────────────────────────────────────────────────────
# What-If utilities (to simulate “What if Catalase was negative?”)
# ──────────────────────────────────────────────────────────────────────────────
def apply_what_if(base_json: Dict[str, str], what_if_text: str, db_fields: List[str]) -> Dict[str, str]:
    """
    Parse a short 'what-if' phrase and return a modified copy of base_json.
    Examples:
      - "What if catalase was negative?"
      - "Set oxidase to positive and rhamnose negative"
      - "Make MR negative, VP positive"
    """
    out = dict(base_json or {})
    t = normalize_text(what_if_text)

    # Map simple "set/make <test> to <polarity>"
    for m in re.finditer(r"\b(?:set|make)\s+([a-z0-9 \-/]+?)\s+(?:to|as)?\s*(positive|negative|variable|\+|\-)\b", t):
        test = m.group(1).strip()
        pol = m.group(2).strip()
        # Abbrev expansion
        test_low = test.lower()
        if test_low in ABBREV_TO_FIELD:
            test = ABBREV_TO_FIELD[test_low]
        # Normalize target column
        target = build_alias_map(db_fields).get(test.lower(), test)
        if target in normalize_columns(db_fields):
            out[target] = _canon_value(target, pol)

    # Simple "what if <test> <polarity>"
    for m in re.finditer(r"\bwhat\s+if\s+([a-z0-9 \-/]+?)\s+(positive|negative|variable|\+|\-)\b", t):
        test = m.group(1).strip()
        pol = m.group(2).strip()
        test_low = test.lower()
        if test_low in ABBREV_TO_FIELD:
            test = ABBREV_TO_FIELD[test_low]
        target = build_alias_map(db_fields).get(test.lower(), test)
        if target in normalize_columns(db_fields):
            out[target] = _canon_value(target, pol)

    # Comma-separated shorthand: "mr -, vp +, catalase -"
    for m in re.finditer(r"\b([a-z]{2,5})\s*([+\-])\b", t):
        ab = m.group(1).lower()
        pol = m.group(2)
        if ab in ABBREV_TO_FIELD:
            fname = ABBREV_TO_FIELD[ab]
            out[fname] = _canon_value(fname, "Positive" if pol == "+" else "Negative")

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Gold Spec tests (starter set)
# ──────────────────────────────────────────────────────────────────────────────
GOLD_SPEC: List[Tuple[str, Dict[str, str]]] = [
    # 1) Staphylococcus aureus-like (rich coverage)
    (
        "Gram-positive cocci, capsulated, motile, non-spore-forming. β-haemolysis on blood agar; yellow colonies on MacConkey agar. "
        "Catalase positive, oxidase negative, indole negative, urease negative, citrate positive. VP positive, Methyl Red negative, DNase positive, "
        "gelatin liquefaction positive, esculin hydrolysis positive. Reduces nitrate. Produces H₂S negative. Aerobic. "
        "No growth at 10 °C; grows at 45 °C. NaCl tolerant up to 6%. "
        "Ferments glucose, lactose, sucrose, maltose, arabinose, raffinose, inositol, trehalose, but not mannitol, xylose or rhamnose. "
        "Lysine decarboxylase positive, Ornithine decarboxylase negative, Arginine dihydrolase positive. ONPG test positive.",
        {
            "Gram Stain":"Positive","Shape":"Cocci","Capsule":"Positive","Spore Formation":"Negative","Motility":"Positive",
            "Haemolysis Type":"Beta","Haemolysis":"Positive",
            "Colony Morphology":"Yellow",  # may include more descriptors
            "Media Grown On":"Blood Agar; MacConKey Agar",  # normalizer will clamp
            "Catalase":"Positive","Oxidase":"Negative","Indole":"Negative","Urease":"Negative","Citrate":"Positive",
            "VP":"Positive","Methyl Red":"Negative","Dnase":"Positive","Gelatin Hydrolysis":"Positive","Esculin Hydrolysis":"Positive",
            "Nitrate Reduction":"Positive","H2S":"Negative",
            "Oxygen Requirement":"Aerobic","NaCl Tolerant (>=6%)":"Positive","Growth Temperature":"45",
            "Glucose Fermentation":"Positive","Lactose Fermentation":"Positive","Sucrose Fermentation":"Positive",
            "Maltose Fermentation":"Positive","Arabinose Fermentation":"Positive","Raffinose Fermentation":"Positive",
            "Inositol Fermentation":"Positive","Trehalose Fermentation":"Positive","Mannitol Fermentation":"Negative",
            "Xylose Fermentation":"Negative","Rhamnose Fermentation":"Negative",
            "Lysine Decarboxylase":"Positive","Ornitihine Decarboxylase":"Negative","Arginine dihydrolase":"Positive","ONPG":"Positive"
        }
    ),
    # 2) Salmonella-like
    (
        "Gram-negative rod, motile, non-spore-forming. Colonies are black and small on XLD agar; no haemolysis on blood agar. "
        "Oxidase negative, catalase positive, indole negative. Urease negative, citrate positive; MR positive, VP negative. "
        "Produces H2S on TSI; reduces nitrate. Gelatin hydrolysis negative, DNase negative, esculin hydrolysis negative. "
        "No growth at 45 °C; grows at 37 °C. Facultative anaerobe. Not tolerant of 6% NaCl. "
        "Ferments glucose, maltose, mannitol, arabinose, xylose, trehalose, but not lactose, sucrose, raffinose, inositol or rhamnose. ONPG negative. "
        "Lysine decarboxylase positive, ornithine decarboxylase positive, arginine dihydrolase negative. Capsule negative.",
        {
            "Gram Stain":"Negative","Shape":"Rods","Motility":"Positive","Spore Formation":"Negative",
            "Colony Morphology":"Black; Small",
            "Media Grown On":"XLD Agar; Blood Agar",
            "Haemolysis Type":"Gamma","Haemolysis":"Variable",   # per your rule
            "Oxidase":"Negative","Catalase":"Positive","Indole":"Negative","Urease":"Negative",
            "Citrate":"Positive","Methyl Red":"Positive","VP":"Negative","Dnase":"Negative","Esculin Hydrolysis":"Negative",
            "Nitrate Reduction":"Positive","H2S":"Positive","Capsule":"Negative",
            "Oxygen Requirement":"Facultative Anaerobe","NaCl Tolerant (>=6%)":"Negative","Growth Temperature":"37",
            "Glucose Fermentation":"Positive","Maltose Fermentation":"Positive","Mannitol Fermentation":"Positive",
            "Arabinose Fermentation":"Positive","Xylose Fermentation":"Positive","Trehalose Fermentation":"Positive",
            "Lactose Fermentation":"Negative","Sucrose Fermentation":"Negative","Raffinose Fermentation":"Negative",
            "Inositol Fermentation":"Negative","Rhamnose Fermentation":"Negative","ONPG":"Negative",
            "Lysine Decarboxylase":"Positive","Ornitihine Decarboxylase":"Positive","Arginine dihydrolase":"Negative"
        }
    ),
    # 3) Pseudomonas aeruginosa-like
    (
        "Gram-negative rod, motile, non-spore-forming. Produces green pigment, colonies are smooth and spreading; growth on nutrient agar, no growth on MacConkey as lactose fermenter (NLF). "
        "Oxidase positive, catalase positive. Indole negative, urease variable, citrate positive. MR negative, VP negative. "
        "H2S negative, nitrate reduction positive, DNase negative, gelatin hydrolysis positive, esculin hydrolysis negative. Aerobic. "
        "Does not ferment lactose, sucrose, mannitol, or rhamnose; utilizes glucose oxidatively. NaCl tolerance variable. Grows at 37 °C but not at 45 °C.",
        {
            "Gram Stain":"Negative","Shape":"Rods","Motility":"Positive","Spore Formation":"Negative",
            "Colony Morphology":"Green; Smooth; Spreading",
            "Media Grown On":"Nutrient Agar",
            "Oxidase":"Positive","Catalase":"Positive","Indole":"Negative","Urease":"Variable","Citrate":"Positive",
            "Methyl Red":"Negative","VP":"Negative","H2S":"Negative","Nitrate Reduction":"Positive","Dnase":"Negative",
            "Gelatin Hydrolysis":"Positive","Esculin Hydrolysis":"Negative","Oxygen Requirement":"Aerobic",
            "Lactose Fermentation":"Negative","Sucrose Fermentation":"Negative","Mannitol Fermentation":"Negative","Rhamnose Fermentation":"Negative",
            "NaCl Tolerant (>=6%)":"Variable","Growth Temperature":"37","Glucose Fermentation":"Positive"  # treat as Positive if stated “utilizes glucose”
        }
    ),
    # 4) Listeria monocytogenes-like
    (
        "Gram-positive short rods, tumbling motility at room temperature, non-spore-forming. Colony small, grey-white, translucent, with narrow beta-haemolysis on blood agar. "
        "Catalase positive, oxidase negative, indole negative. Urease negative, citrate negative. MR variable, VP negative. H2S negative. "
        "Esculin hydrolysis positive, gelatin hydrolysis negative, DNase negative. Microaerophilic to facultative. Grows at 4–37 °C; not at 45 °C. "
        "Ferments glucose, rhamnose; does not ferment mannitol, xylose, or inositol. ONPG negative.",
        {
            "Gram Stain":"Positive","Shape":"Short Rods","Motility":"Positive","Spore Formation":"Negative",
            "Colony Morphology":"Small; Grey; White; Translucent",
            "Media Grown On":"Blood Agar",
            "Haemolysis Type":"Beta","Haemolysis":"Positive",
            "Catalase":"Positive","Oxidase":"Negative","Indole":"Negative","Urease":"Negative","Citrate":"Negative",
            "Methyl Red":"Variable","VP":"Negative","H2S":"Negative","Esculin Hydrolysis":"Positive","Gelatin Hydrolysis":"Negative","Dnase":"Negative",
            "Oxygen Requirement":"Microaerophilic","Growth Temperature":"37",
            "Glucose Fermentation":"Positive","Rhamnose Fermentation":"Positive","Mannitol Fermentation":"Negative","Xylose Fermentation":"Negative","Inositol Fermentation":"Negative","ONPG":"Negative"
        }
    ),
    # 5) Enterobacter cloacae-like (close to your last test)
    (
        "Gram-negative rod, motile and non-spore-forming. Colonies smooth on nutrient agar and lactose-fermenting on MacConkey (LF). "
        "Oxidase negative, catalase positive, indole negative. Urease negative, citrate positive. MR negative, VP positive. "
        "H2S negative, nitrate reduction positive, DNase negative, esculin hydrolysis positive. Facultative anaerobe. Grows at 37 °C. "
        "Ferments glucose, lactose, sucrose, mannose (treat as mannitol), and sorbitol; does not ferment xylose or rhamnose. ONPG positive. Lysine decarboxylase negative; ornithine decarboxylase positive; arginine dihydrolase negative. Capsule present.",
        {
            "Gram Stain":"Negative","Shape":"Rods","Motility":"Positive","Spore Formation":"Negative",
            "Colony Morphology":"Smooth",
            "Media Grown On":"Nutrient Agar; MacConkey Agar",
            "Oxidase":"Negative","Catalase":"Positive","Indole":"Negative","Urease":"Negative","Citrate":"Positive",
            "Methyl Red":"Negative","VP":"Positive","H2S":"Negative","Nitrate Reduction":"Positive","Dnase":"Negative","Esculin Hydrolysis":"Positive",
            "Oxygen Requirement":"Facultative Anaerobe","Growth Temperature":"37","Capsule":"Positive",
            "Glucose Fermentation":"Positive","Lactose Fermentation":"Positive","Sucrose Fermentation":"Positive","Mannitol Fermentation":"Positive","Sorbitol Fermentation":"Positive",
            "Xylose Fermentation":"Negative","Rhamnose Fermentation":"Negative","ONPG":"Positive",
            "Lysine Decarboxylase":"Negative","Ornitihine Decarboxylase":"Positive","Arginine dihydrolase":"Negative"
        }
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# Gold Spec test runner
# ──────────────────────────────────────────────────────────────────────────────
def _default_db_fields_for_tests() -> List[str]:
    """A minimal stand-in of your real Excel columns (order doesn’t matter)."""
    return [
        "Genus","Gram Stain","Shape","Catalase","Oxidase","Colony Morphology","Haemolysis","Haemolysis Type","Indole",
        "Growth Temperature","Media Grown On","Motility","Capsule","Spore Formation","Oxygen Requirement","Methyl Red","VP",
        "Citrate","Urease","H2S","Lactose Fermentation","Glucose Fermentation","Sucrose Fermentation","Nitrate Reduction",
        "Lysine Decarboxylase","Ornitihine Decarboxylase","Arginine dihydrolase","Gelatin Hydrolysis","Esculin Hydrolysis",
        "Dnase","ONPG","NaCl Tolerant (>=6%)","Lipase Test","Xylose Fermentation","Rhamnose Fermentation","Mannitol Fermentation",
        "Sorbitol Fermentation","Maltose Fermentation","Arabinose Fermentation","Raffinose Fermentation","Inositol Fermentation",
        "Trehalose Fermentation","Coagulase"
    ]

def _compare_dicts(pred: Dict[str, str], exp: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Return (missing_keys, mismatched_keys).
    - missing_keys: expected keys not present in pred
    - mismatched_keys: keys present in both but different values
    """
    missing, mismatch = [], []
    for k, v in exp.items():
        if k not in pred:
            missing.append(k); continue
        if str(pred[k]) != str(v):
            mismatch.append(k)
    return missing, mismatch

def run_gold_tests(verbose: bool = True) -> bool:
    """Run the GOLD_SPEC tests with a stand-in column list and print a summary."""
    cols = _default_db_fields_for_tests()
    passed = 0
    for i, (para, expected) in enumerate(GOLD_SPEC, start=1):
        pred = parse_input_free_text(para, db_fields=cols)
        missing, mismatch = _compare_dicts(pred, expected)
        ok = (not missing and not mismatch)
        if verbose:
            print("="*80)
            print(f"[{i}] {'PASS' if ok else 'FAIL'}")
            if not ok:
                print("Input:", para)
                print("\nPredicted JSON:\n", json.dumps(pred, indent=2, ensure_ascii=False))
                if missing:
                    print("\nMissing keys:\n", missing)
                if mismatch:
                    print("\nMismatched keys (pred vs exp):")
                    for k in mismatch:
                        print(f" - {k}: pred='{pred.get(k)}'  exp='{expected.get(k)}'")
            else:
                print("Predicted JSON:\n", json.dumps(pred, indent=2, ensure_ascii=False))
        passed += int(ok)
    if verbose:
        print("="*80)
        print(f"Gold Spec: {passed}/{len(GOLD_SPEC)} passed.")
    return passed == len(GOLD_SPEC)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--test" in sys.argv:
        ok = run_gold_tests(verbose=True)
        sys.exit(0 if ok else 1)

    if "--demo" in sys.argv:
        idx = sys.argv.index("--demo")
        text = sys.argv[idx+1] if idx+1 < len(sys.argv) else ""
        cols = _default_db_fields_for_tests()
        result = parse_input_free_text(text, db_fields=cols)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0)

    if "--whatif" in sys.argv:
        # Demo: parse first GOLD spec, then apply a what-if from CLI
        wi_idx = sys.argv.index("--whatif")
        wi_text = sys.argv[wi_idx+1] if wi_idx+1 < len(sys.argv) else "set catalase to negative"
        cols = _default_db_fields_for_tests()
        base_para, _exp = GOLD_SPEC[0]
        base = parse_input_free_text(base_para, db_fields=cols)
        mod  = apply_what_if(base, wi_text, cols)
        print("Base JSON:")
        print(json.dumps(base, indent=2, ensure_ascii=False))
        print("\nWhat-If:", wi_text)
        print("\nModified JSON:")
        print(json.dumps(mod, indent=2, ensure_ascii=False))
        sys.exit(0)

    # Help
    print("Usage:")
    print("  python parser_basic.py --test")
    print("  python parser_basic.py --demo \"<paragraph>\"")
    print("  python parser_basic.py --whatif \"set catalase to negative\"")
