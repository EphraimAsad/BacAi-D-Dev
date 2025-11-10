# parser_basic.py — v4 (Pure-regex self-learning fallback)
# ---------------------------------------------------------------------------------------
# What this file does
# - Deterministic (no-LLM) parsing of free-text microbiology observations into your schema
# - Same field/normalization vocabulary you use elsewhere
# - Writes structured feedback on mismatches to parser_feedback.json
# - Analyzes feedback to detect recurring mistakes (≥3 occurrences) and stores
#   learned heuristics in parser_memory.json
# - Optional auto-patcher can inject new regex lines into this very file
#
# Public API
#   parse_input_free_text(text, prior_facts=None, db_fields=None) -> Dict[str,str]
#   apply_what_if(base_json, what_if_text, db_fields) -> Dict[str,str]
#   run_gold_tests(verbose=True) -> bool
#   append_feedback_case(name, text, expected, got)  # structured logging
#   analyze_feedback_and_learn(feedback_path="parser_feedback.json",
#                              memory_path="parser_memory.json")
#   auto_update_parser_regex(memory_path="parser_memory.json",
#                            parser_file="parser_basic.py")
#
# CLI:
#   python parser_basic.py --test
#   python parser_basic.py --demo "..."
#   python parser_basic.py --whatif "set catalase to negative"
#
# NOTE: This file is **pure regex**. Learning is also regex-based (no LLM).
# ---------------------------------------------------------------------------------------

import os
import re
import json
import sys
import difflib
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Allowed values & canonicalization (aligned with your Excel)
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_VALUES: Dict[str, Set[str]] = {
    "Gram Stain": {"Positive", "Negative", "Variable"},
    "Shape": {"Cocci", "Rods", "Bacilli", "Spiral", "Short Rods"},
    "Catalase": {"Positive", "Negative", "Variable"},
    "Oxidase": {"Positive", "Negative", "Variable"},
    "Colony Morphology": set(),  # free text
    "Haemolysis": {"Positive", "Negative", "Variable"},
    "Haemolysis Type": {"None", "Beta", "Gamma", "Alpha"},
    "Indole": {"Positive", "Negative", "Variable"},
    "Growth Temperature": set(),
    "Media Grown On": set(),
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

# Whitelisted media (TSI explicitly excluded later)
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
    "Cell Culture","Intracellular","Brain Heart Infusion Agar","Human Fibroblast Cell Culture","BCYE Agar",
    "Columbia Blood Agar","Blood Agar"
}

# Synonyms / abrevs
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
        "weakly positive": "Variable","variable": "Variable","weak": "Variable","trace": "Variable","slight": "Variable"
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

ABBREV_TO_FIELD = {
    "mr": "Methyl Red",
    "vp": "VP",
    "ldc": "Lysine Decarboxylase",
    "odc": "Ornitihine Decarboxylase",
    "adh": "Arginine dihydrolase",
    "nlf": "Lactose Fermentation",
    "lf": "Lactose Fermentation",
}

MEDIA_ABBREV = {
    "tsa": "Tryptic Soy Agar",
    "bhi": "Brain Heart Infusion Agar",
    "cba": "Columbia Blood Agar",
    "ssa": "Blood Agar",
    "ba": "Blood Agar",
}

CM_TOKENS = {
    # sizes & measurements
    "1/3mm","1/2mm","2/3mm","2/4mm","0.5/1mm","0.5mm/2mm","1mm","2mm","3mm","small","medium","large","tiny","pinpoint","subsurface","satellite",
    # shapes/profile
    "round","circular","convex","flat","domed","heaped","fried egg",
    # edges/surface/texture
    "smooth","rough","wrinkled","granular","mucoid","glistening","dull","matte","shiny","sticky","adherent","powdery","chalk","leathery","velvet","crumbly",
    "ground glass","irregular edges","spreading","swarming","corrode","pit","ropey","butyrous","waxy","ring","dingers ring","bright","pigmented",
    # opacity/transparency
    "opaque","translucent","colourless","colorless",
    # moisture
    "dry","moist",
    # colours
    "white","grey","gray","cream","off-white","yellow","pale yellow","orange","pink","coral","red","green","violet","purple","black","brown","beige","tan","blue",
}

# ──────────────────────────────────────────────────────────────────────────────
# Learning storage files (shared with parser_llm.py)
# ──────────────────────────────────────────────────────────────────────────────
FEEDBACK_PATH = "parser_feedback.json"
MEMORY_PATH = "parser_memory.json"

def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: str, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Could not write {path}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Pattern lists eligible for auto-learning injection
#   The auto-updater appends new r"...regex..." items into these lists.
#   The core extractor still uses its built-in robust patterns; these lists are
#   *additional* matchers learned from feedback.
# ──────────────────────────────────────────────────────────────────────────────
CATALASE_PATTERNS: List[str] = []
OXIDASE_PATTERNS: List[str] = []
INDOLE_PATTERNS: List[str] = []
VP_PATTERNS: List[str] = []
MR_PATTERNS: List[str] = []
UREASE_PATTERNS: List[str] = []
CITRATE_PATTERNS: List[str] = []
H2S_PATTERNS: List[str] = []
COAGULASE_PATTERNS: List[str] = []
LIPASE_PATTERNS: List[str] = []
ESCULIN_PATTERNS: List[str] = []
DNASE_PATTERNS: List[str] = []
GELATIN_PATTERNS: List[str] = []
NITRATE_PATTERNS: List[str] = []
DECARBOXYLASE_PATTERNS: List[str] = []
FERMENTATION_PATTERNS: List[str] = []  # learned patterns that include sugar name + polarity words

# ──────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ──────────────────────────────────────────────────────────────────────────────
_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def normalize_columns(db_fields: List[str]) -> List[str]:
    return [f for f in (db_fields or []) if f and f.strip().lower() != "genus"]

def build_alias_map(db_fields: List[str]) -> Dict[str, str]:
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
    add("glucose fermantation","Glucose Fermentation")  # sheet typo

    # Auto base→fermentation column
    for f in fields:
        if f.lower().endswith(" fermentation"):
            base = f[:-12].strip().lower()
            alias[base] = f

    # Media names → "Media Grown On"
    for m in MEDIA_WHITELIST:
        alias[m.lower()] = "Media Grown On"
    for k, v in MEDIA_ABBREV.items():
        alias[k.lower()] = "Media Grown On"

    return alias

# ──────────────────────────────────────────────────────────────────────────────
# Text & token helpers
# ──────────────────────────────────────────────────────────────────────────────
def normalize_text(raw: str) -> str:
    t = raw or ""
    t = t.replace("°", " °")
    t = t.translate(_SUBSCRIPT_DIGITS)
    t = (t.replace("\u2010","-").replace("\u2011","-").replace("\u2012","-")
           .replace("\u2013","-").replace("\u2014","-").replace("–","-").replace("—","-"))
    t = re.sub(r"hemolys", "haemolys", t, flags=re.I)
    t = re.sub(r"gray", "grey", t, flags=re.I)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _normalize_token(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"[.,;:!?\-]+$", "", s)
    return s.strip()

def _tokenize_list(s: str) -> List[str]:
    s = re.sub(r"\s*(?:,|and|or|&|nor)\s*", ",", s.strip(), flags=re.I)
    return [re.sub(r"[.,;:\s]+$", "", t.strip()) for t in s.split(",") if t.strip()]

def _windows(tokens: List[str], idx: int, span: int = 5) -> List[str]:
    lo = max(0, idx - span)
    hi = min(len(tokens), idx + span + 1)
    return tokens[lo:hi]

def _has_negation_near(tokens: List[str], idx: int, span: int = 5) -> bool:
    window = " ".join(_windows(tokens, idx, span))
    return bool(re.search(r"\b(no|not|absent|without|lack|lacks|did\s+not|does\s+not|cannot|negative|not\s+observed|no\s+growth|no\s+reaction|not\s+produced|non[-\s]?fermenter)\b", window))

def _has_variable_near(tokens: List[str], idx: int, span: int = 5) -> bool:
    window = " ".join(_windows(tokens, idx, span))
    return bool(re.search(r"\b(variable|weak|weakly|trace|inconsistent|equivocal|slight)\b", window))

# ──────────────────────────────────────────────────────────────────────────────
# Safe setter & canonicalization helpers
# ──────────────────────────────────────────────────────────────────────────────
def _set_field_safe(out: Dict[str, str], key: str, val: str):
    cur = out.get(key)
    if cur is None:
        out[key] = val; return
    if cur == "Negative" and val == "Positive":
        return
    if cur == "Variable" and val == "Positive":
        out[key] = "Positive"; return
    out[key] = val

def _canon_value(field: str, value: str) -> str:
    v = (value or "").strip()
    if not v:
        return v

    if field in POLARITY_FIELDS:
        low = v.lower()
        syn = VALUE_SYNONYMS.get("*POLARITY*", {})
        if low in syn:
            v = syn[low]
        else:
            if re.fullmatch(r"\+|positive|pos", low): v = "Positive"
            elif re.fullmatch(r"\-|negative|neg", low): v = "Negative"
            elif any(x in low for x in ["weak","variable","trace","slight"]): v = "Variable"

    low = v.lower()
    if field in VALUE_SYNONYMS:
        v = VALUE_SYNONYMS[field].get(low, v)

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
    t = text.lower()
    spans = []
    m = re.search(r"colon(?:y|ies)\s+(?:are|appear|were|appearing|appeared)\s+([^.]+?)(?:\s+on|\.)", t)
    if m:
        spans.append(m.group(1))
    spans.append(t)

    found: List[str] = []
    def add(tok: str):
        tok = tok.strip()
        if tok and tok not in found:
            found.append(tok)

    # measurements + multiwords + tokens
    for s in spans:
        for mm in re.findall(r"(?:\d+(?:\.\d+)?\/\d+(?:\.\d+)?mm|\d+(?:\.\d+)?mm|0\.5\/1mm|0\.5mm\/2mm|1\/3mm|2\/3mm|2\/4mm)", s):
            add(mm)

        s_norm = " " + re.sub(r"[,;/]", " ", s) + " "
        for mw in ["ground glass","irregular edges","fried egg","dingers ring","off-white","pale yellow","cream-white","grey-cream","mucoid ropey","butyrous"]:
            if f" {mw} " in s_norm:
                add(mw)

        parts = re.split(r"[,;:/\-\s]+", s)
        for p in parts:
            low = p.strip().lower()
            if low == "colorless": low = "colourless"
            if low in CM_TOKENS:
                add(low)
            if low in {"off-white","pale","pale-yellow","cream-white","grey-cream","ropey","butyrous"}:
                add(low.replace("-", " "))

    # order & pretty
    groups = [
        {"1/3mm","1/2mm","2/3mm","2/4mm","0.5/1mm","0.5mm/2mm","1mm","2mm","3mm","tiny","small","medium","large","pinpoint","subsurface","satellite"},
        {"round","circular","convex","flat","domed","heaped","fried egg"},
        {"smooth","rough","wrinkled","granular","mucoid","glistening","dull","matte","shiny","sticky","adherent","powdery","chalk","leathery","velvet","crumbly",
         "ground glass","irregular edges","spreading","swarming","corrode","pit","ring","dingers ring","ropey","butyrous","waxy","bright","pigmented"},
        {"opaque","translucent","colourless"},
        {"dry","moist"},
        {"white","grey","gray","cream","off-white","yellow","pale yellow","orange","pink","coral","red","green","violet","purple","black","brown","beige","tan","blue"},
    ]
    ordered, seen = [], set()
    for grp in groups:
        for tok in found:
            if tok in grp and tok not in seen:
                ordered.append(tok); seen.add(tok)
    for tok in found:
        if tok not in seen:
            ordered.append(tok); seen.add(tok)

    pretty = []
    for w in ordered:
        if w == "gray": w = "grey"
        if re.search(r"\d", w) or w.isupper():
            pretty.append(w)
        else:
            if w == "pale yellow": pretty.append("Yellow (Pale)")
            elif w == "off-white": pretty.append("Off-White")
            elif w == "cream-white": pretty.append("Cream; White")
            else: pretty.append(w.title())

    flat = []
    for item in pretty:
        if item == "Cream; White":
            flat.extend(["Cream","White"])
        else:
            flat.append(item)
    return "; ".join(flat)

# ──────────────────────────────────────────────────────────────────────────────
# Extraction: fermentations (core + learned patterns)
# ──────────────────────────────────────────────────────────────────────────────
def extract_fermentations(text: str, db_fields: List[str]) -> Dict[str, str]:
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

    # positive lists
    for m in re.finditer(r"(?:ferments?|utilizes?|produces?\s+acid\s+from)\s+([a-z0-9\.\-%\s,/&]+)", t, flags=re.I):
        span = re.split(r"(?i)\bbut\s+not\b", m.group(1))[0]
        for a in _tokenize_list(span):
            set_field_by_base(a, "Positive")

    # explicit negatives
    for pat in [
        r"(?:does\s+not|doesn't)\s+(?:ferment|utilize)\s+([a-z0-9\.\-%\s,/&]+)",
        r"cannot\s+(?:ferment|utilize)\s+([a-z0-9\.\-%\s,/&]+)",
        r"unable\s+to\s+(?:ferment|utilize)\s+([a-z0-9\.\-%\s,/&]+)",
        r"non[-\s]?fermenter\s+(?:for|of)?\s+([a-z0-9\.\-%\s,/&]+)",
    ]:
        for m in re.finditer(pat, t, flags=re.I):
            for a in _tokenize_list(m.group(1)):
                set_field_by_base(a, "Negative")

    # “but not …”
    for m in re.finditer(r"(?:ferments?|utilizes?)[^.]*?\bbut\s+not\s+([\w\s,;.&-]+)", t, flags=re.I):
        seg = m.group(1)
        seg = re.sub(r"\bor\b", ",", seg, flags=re.I)
        seg = re.sub(r"\bnor\b", ",", seg, flags=re.I)
        for a in _tokenize_list(seg):
            set_field_by_base(a, "Negative")

    # shorthand +/- (e.g., "lactose +")
    for m in re.finditer(r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b", t, flags=re.I):
        a, sign = m.group(1), m.group(2)
        set_field_by_base(a, "Positive" if sign == "+" else "Negative")

    # variable phrasing: “raffinose variable”
    for base in list(base_to_field.keys()):
        if re.search(rf"\b{re.escape(base)}\b\s+(?:variable|inconsistent|weak|trace|slight|irregular)", t, flags=re.I):
            set_field_by_base(base, "Variable")

    # LF / NLF short forms
    if re.search(r"\bnlf\b", t):
        set_field_by_base("lactose", "Negative")
    if re.search(r"\blf\b", t) and not re.search(r"\bnlf\b", t):
        set_field_by_base("lactose", "Positive")

    # 🔎 Learned fermentation patterns (from FERMENTATION_PATTERNS)
    # These patterns typically include the sugar base in the text; we find which one.
    for pat in FERMENTATION_PATTERNS:
        m = re.search(pat, t, flags=re.I)
        if not m:
            continue
        matched = m.group(0).lower()
        # infer sugar base mentioned
        sugar = next((b for b in base_to_field.keys() if re.search(rf"\b{re.escape(b)}\b", matched, flags=re.I)), None)
        if not sugar:
            continue
        # infer polarity from the matched text
        if re.search(r"\+|positive|detected", matched):
            set_field_by_base(sugar, "Positive")
        elif re.search(r"\-|negative|not\s+detected|not\s+ferment|non[-\s]?ferment", matched):
            set_field_by_base(sugar, "Negative")
        elif re.search(r"variable|weak|trace|slight|inconsistent|equivocal", matched):
            set_field_by_base(sugar, "Variable")

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Extraction: biochem/morph/media/oxygen/temp (core + learned patterns)
# ──────────────────────────────────────────────────────────────────────────────
def extract_biochem(text: str, db_fields: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    raw = text or ""
    t = normalize_text(raw)
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)
    tokens = t.split()

    def set_field(key_like: str, val: str, at_token_idx: Optional[int] = None):
        target = alias.get(key_like.lower(), key_like)
        if target not in fields:
            return
        v = val
        if at_token_idx is not None:
            if _has_negation_near(tokens, at_token_idx, 5):
                v = "Negative"
            elif _has_variable_near(tokens, at_token_idx, 5) and target in POLARITY_FIELDS:
                v = "Variable"
        _set_field_safe(out, target, _canon_value(target, v))

    def first_token_index_of(pattern: str) -> Optional[int]:
        m = re.search(pattern, t)
        if not m:
            return None
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
        idx = first_token_index_of(r"\bcocci?\b"); set_field("shape", "Cocci", idx)
    if re.search(r"\bshort\s+rods?\b", t):
        idx = first_token_index_of(r"\bshort\s+rods?\b"); set_field("shape", "Short Rods", idx)
    elif re.search(r"\bbacilli?\b", t):
        idx = first_token_index_of(r"\bbacilli?\b"); set_field("shape", "Bacilli", idx)
    elif re.search(r"\brods?\b", t):
        idx = first_token_index_of(r"\brods?\b"); set_field("shape", "Rods", idx)
    if re.search(r"\bspiral\b", t):
        idx = first_token_index_of(r"\bspiral\b"); set_field("shape", "Spiral", idx)

    # Motility
    if re.search(r"\bnon[-\s]?motile\b", t):
        idx = first_token_index_of(r"\bnon[-\s]?motile\b"); set_field("motility", "Negative", idx)
    elif re.search(r"\bmotile\b", t):
        idx = first_token_index_of(r"\bmotile\b"); set_field("motility", "Positive", idx)

    # Capsule
    if re.search(r"\b(encapsulated|capsulated)\b", t):
        idx = first_token_index_of(r"\b(encapsulated|capsulated)\b"); set_field("capsule", "Positive", idx)
    if re.search(r"\bnon[-\s]?capsulated\b|\bcapsule\s+absent\b", t):
        idx = first_token_index_of(r"\bnon[-\s]?capsulated\b|\bcapsule\s+absent\b"); set_field("capsule", "Negative", idx)
    if re.search(r"\bcapsule\s+(?:variable|inconsistent|weak)\b", t):
        idx = first_token_index_of(r"\bcapsule\s+(?:variable|inconsistent|weak)\b"); set_field("capsule", "Variable", idx)

    # Spore formation
    if re.search(r"\bnon[-\s]?spore[-\s]?forming\b|\bno\s+spores?\b", t):
        idx = first_token_index_of(r"\bnon[-\s]?spore[-\s]?forming\b|\bno\s+spores?\b"); set_field("spore formation", "Negative", idx)
    if re.search(r"\bspore[-\s]?forming\b|\bspores?\s+present\b", t):
        idx = first_token_index_of(r"\bspore[-\s]?forming\b|\bspores?\s+present\b"); set_field("spore formation", "Positive", idx)

    # Oxygen
    if re.search(r"\bintracellular\b", t): set_field("oxygen requirement", "Intracellular")
    elif re.search(r"\bcapnophil(ic|e)\b", t): set_field("oxygen requirement", "Capnophilic")
    elif re.search(r"\bmicroaerophil(ic|e)\b", t): set_field("oxygen requirement", "Microaerophilic")
    elif re.search(r"\bfacultative\b", t) or re.search(r"\bfacultative\s+anaerob", t): set_field("oxygen requirement", "Facultative Anaerobe")
    elif re.search(r"\baerobic\b", t): set_field("oxygen requirement", "Aerobic")
    elif re.search(r"\banaerobic\b", t): set_field("oxygen requirement", "Anaerobic")

    # Generic biochemical tests (core patterns)
    generic = ["catalase","oxidase","coagulase","urease","lipase","indole",
               "citrate","vp","methyl red","gelatin","dnase","nitrate reduction","nitrate","h2s","esculin hydrolysis","onpg"]
    for test in generic:
        m = re.search(rf"\b{test}\s*(?:test)?\s*(?:\+|positive|produced|detected)\b", t)
        if m:
            idx = len(t[:m.start()].split()); set_field(test, "Positive", idx)
        m = re.search(rf"\b{test}\s*(?:test)?\s*(?:\-|negative|not\s+produced|not\s+detected|absent)\b", t)
        if m:
            idx = len(t[:m.start()].split()); set_field(test, "Negative", idx)
        m = re.search(rf"\b{test}\s*(?:test)?\s*(?:weak|weakly\s+positive|variable|trace|slight)\b", t)
        if m:
            idx = len(t[:m.start()].split()); set_field(test, "Variable", idx)

    # Nitrate verbs
    if re.search(r"\breduces\s+nitrate\b", t): set_field("nitrate", "Positive")
    if re.search(r"\bdoes\s+not\s+reduce\s+nitrate\b", t): set_field("nitrate", "Negative")

    # H2S precedence
    if re.search(r"\bh\s*2\s*s\s+(?:\+|positive|detected|produced)\b", t): set_field("h2s", "Positive")
    if re.search(r"\bh\s*2\s*s\s+(?:\-|negative|not\s+detected|not\s+produced)\b", t): set_field("h2s", "Negative")

    # Haemolysis Type
    if re.search(r"\b(beta|β)[-\s]?haem", t): set_field("haemolysis type", "Beta")
    elif re.search(r"\b(alpha|α)[-\s]?haem", t): set_field("haemolysis type", "Alpha")
    elif re.search(r"\b(gamma|γ)[-\s]?haem\b", t) or re.search(r"\bno\s+haemolysis\b|\bhaemolysis\s+not\s+observed\b", t):
        set_field("haemolysis type", "Gamma")

    # Abbrev-only for decarboxylases (LDC/ODC/ADH)
    for abbr, fname in [("ldc","Lysine Decarboxylase"),("odc","Ornitihine Decarboxylase"),("adh","Arginine dihydrolase")]:
        if re.search(rf"\b{abbr}\s*(?:\+|positive)\b", t): set_field(fname, "Positive")
        if re.search(rf"\b{abbr}\s*(?:\-|negative)\b", t): set_field(fname, "Negative")

    # Decarboxylases spelled out (singular/plural)
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

    # Growth temperature (record positives; ignore “no growth at …”)
    for m in re.finditer(r"(?<!no\s)grows\s+(?:well\s+)?at\s+([0-9]{1,3})\s*°?\s*c", t):
        set_field("growth temperature", m.group(1))
    for m in re.finditer(r"(?<!no\s)growth\s+at\s+([0-9]{1,3})\s*°?\s*c", t):
        set_field("growth temperature", m.group(1))
    # ranges like "grows 10–40 °C" → prefer ranges in parser_llm, here we keep exemplar temp if present

    # NaCl tolerant
    if re.search(r"\b(tolerant|grows|growth)\s+(?:in|up\s+to|to|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t):
        set_field("nacl tolerant (>=6%)", "Positive")
    if re.search(r"\bno\s+growth\s+(?:in|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t):
        set_field("nacl tolerant (>=6%)", "Negative")
    if re.search(r"\bnacl\s+tolerant\b", t):
        set_field("nacl tolerant (>=6%)", "Positive")

    # Media (exclude TSI)
    diag_exclude = ["triple sugar iron", "tsi"]
    media_hits = re.findall(r"\bon\s+([a-z0-9\-\+ ]+?)\s+agar\b", t)
    collected_media: List[str] = []
    for mname in media_hits:
        lowname = mname.strip().lower()
        if any(ex in lowname for ex in diag_exclude):
            continue
        up = lowname.upper().replace(" ", "")
        if up == "XLD":
            pretty = "XLD Agar"
        elif up in {"MACCONKEY","MAC"}:
            pretty = "MacConkey Agar"
        elif up in {"BLOOD","BA","SSA"}:
            pretty = "Blood Agar"
        else:
            pretty = mname.strip().title() + " Agar"
        canon = next((w for w in MEDIA_WHITELIST if w.lower() == pretty.lower()), pretty)
        if canon not in collected_media:
            collected_media.append(canon)

    bare_media = re.findall(r"\bon\s+(xld|macconkey|blood|tsa|bhi|cba)\b", t)
    for bm in bare_media:
        key = bm.lower()
        pretty = MEDIA_ABBREV.get(key) or ("XLD Agar" if key=="xld" else "MacConkey Agar" if key=="macconkey" else "Blood Agar" if key=="blood" else None)
        if pretty and pretty not in collected_media:
            collected_media.append(pretty)

    if collected_media:
        _set_field_safe(out, "Media Grown On", "; ".join(collected_media))

    # Colony morphology
    cm_value = normalize_cm_phrase(raw)
    if cm_value:
        _set_field_safe(out, "Colony Morphology", cm_value)

    # 🔎 Learned test patterns (additional to core). We infer polarity from matched text.
    learned_map = [
        ("catalase", CATALASE_PATTERNS),
        ("oxidase", OXIDASE_PATTERNS),
        ("indole", INDOLE_PATTERNS),
        ("vp", VP_PATTERNS),
        ("methyl red", MR_PATTERNS),
        ("urease", UREASE_PATTERNS),
        ("citrate", CITRATE_PATTERNS),
        ("h2s", H2S_PATTERNS),
        ("coagulase", COAGULASE_PATTERNS),
        ("lipase", LIPASE_PATTERNS),
        ("esculin hydrolysis", ESCULIN_PATTERNS),
        ("dnase", DNASE_PATTERNS),
        ("gelatin", GELATIN_PATTERNS),
        ("nitrate", NITRATE_PATTERNS),
        ("decarboxylase", DECARBOXYLASE_PATTERNS),
    ]
    for key_like, patterns in learned_map:
        for pat in patterns:
            m = re.search(pat, t, flags=re.I)
            if not m:
                continue
            matched = m.group(0).lower()
            if re.search(r"\+|positive|detected|produced", matched):
                set_field(key_like, "Positive")
            elif re.search(r"\-|negative|not\s+detected|absent|not\s+produced", matched):
                set_field(key_like, "Negative")
            elif re.search(r"variable|weak|trace|slight|inconsistent|equivocal", matched):
                set_field(key_like, "Variable")

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Normalization & Haemolysis bridge to your schema
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

    # Haemolysis Type → Haemolysis
    ht = alias.get("haemolysis type"); h = alias.get("haemolysis")
    if ht in out and h in fields:
        tval = out.get(ht, "")
        if tval in {"Alpha", "Beta"}:
            out[h] = "Positive"
        elif tval in {"Gamma", "None"}:
            out[h] = "Variable"  # your chosen policy

    # Clamp/dedupe media
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

    # Deduplicate colony morphology
    if "Colony Morphology" in out and out["Colony Morphology"]:
        chunks = [c.strip() for c in out["Colony Morphology"].split(";") if c.strip()]
        seen = set(); cleaned = []
        for c in chunks:
            if c not in seen:
                cleaned.append(c); seen.add(c)
        out["Colony Morphology"] = "; ".join(cleaned)

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Deterministic main entry
# ──────────────────────────────────────────────────────────────────────────────
def parse_input_free_text(user_text: str, prior_facts: Dict | None = None, db_fields: List[str] | None = None) -> Dict:
    if not user_text or not user_text.strip():
        return {}
    db_fields = db_fields or []
    ferm = extract_fermentations(user_text, db_fields)
    bio  = extract_biochem(user_text, db_fields)
    merged = {}
    merged.update(ferm)
    merged.update(bio)
    if prior_facts:
        merged.update(prior_facts)
    return normalize_to_schema(merged, db_fields)

# ──────────────────────────────────────────────────────────────────────────────
# What-If utilities
# ──────────────────────────────────────────────────────────────────────────────
def apply_what_if(base_json: Dict[str, str], what_if_text: str, db_fields: List[str]) -> Dict[str, str]:
    out = dict(base_json or {})
    t = normalize_text(what_if_text)
    alias = build_alias_map(db_fields)

    # "set/make <test> to <polarity>"
    for m in re.finditer(r"\b(?:set|make)\s+([a-z0-9 \-/]+?)\s+(?:to|as)?\s*(positive|negative|variable|\+|\-)\b", t):
        test = m.group(1).strip()
        pol = m.group(2).strip()
        test_low = test.lower()
        if test_low in ABBREV_TO_FIELD:
            test = ABBREV_TO_FIELD[test_low]
        target = alias.get(test.lower(), test)
        if target in normalize_columns(db_fields):
            out[target] = _canon_value(target, pol)

    # "what if <test> <polarity>"
    for m in re.finditer(r"\bwhat\s+if\s+([a-z0-9 \-/]+?)\s+(positive|negative|variable|\+|\-)\b", t):
        test = m.group(1).strip()
        pol = m.group(2).strip()
        test_low = test.lower()
        if test_low in ABBREV_TO_FIELD:
            test = ABBREV_TO_FIELD[test_low]
        target = build_alias_map(db_fields).get(test.lower(), test)
        if target in normalize_columns(db_fields):
            out[target] = _canon_value(target, pol)

    # shorthand: "mr -, vp +"
    for m in re.finditer(r"\b([a-z]{2,5})\s*([+\-])\b", t):
        ab = m.group(1).lower()
        pol = m.group(2)
        if ab in ABBREV_TO_FIELD:
            fname = ABBREV_TO_FIELD[ab]
            out[fname] = _canon_value(fname, "Positive" if pol == "+" else "Negative")

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Feedback logging (append one structured case)
# ──────────────────────────────────────────────────────────────────────────────
def _diff_expected_vs_got(expected: Dict[str, str], got: Dict[str, str]) -> Tuple[List[str], List[Tuple[str,str,str]]]:
    missing, mismatched = [], []
    for k, v in expected.items():
        if k not in got:
            missing.append(k); continue
        if str(got[k]) != str(v):
            mismatched.append((k, v, got[k]))
    return missing, mismatched

def append_feedback_case(name: str, text: str, expected: Dict[str,str], got: Dict[str,str], path: str = FEEDBACK_PATH):
    """Add a feedback record capturing mismatches for later learning."""
    missing, mismatched = _diff_expected_vs_got(expected, got)
    if not missing and not mismatched:
        return  # nothing to log

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "text": text,
        "errors": []
    }
    for k in missing:
        record["errors"].append({"field": k, "expected": expected.get(k, ""), "got": ""})
    for (k, e, g) in mismatched:
        record["errors"].append({"field": k, "expected": e, "got": g})

    data = _load_json(path, default=[])
    data.append(record)
    _save_json(path, data)
# ──────────────────────────────────────────────────────────────────────────────
# Gold Spec tests (starter set; same spirit as parser_llm)
# ──────────────────────────────────────────────────────────────────────────────
GOLD_SPEC: List[Tuple[str, Dict[str, str], str]] = [
    # (paragraph, expected_dict, name)
    (
        "Gram-negative rod, motile, non-spore-forming. No haemolysis on blood agar. "
        "Oxidase negative, catalase positive, indole negative. Urease negative, citrate positive, MR positive, VP negative. "
        "Produces H2S on TSI. Nitrate reduced. Gelatin hydrolysis negative, DNase negative, esculin hydrolysis negative. "
        "Does not produce coagulase or lipase. Grows at 37 °C (not at 45 °C), facultative anaerobe, not tolerant of 6 % NaCl. "
        "Ferments glucose, maltose, mannitol, arabinose, xylose, trehalose, but not lactose, sucrose, raffinose, inositol, or rhamnose. ONPG negative. "
        "Lysine and ornithine decarboxylases positive; arginine dihydrolase negative. No capsule observed.",
        {
            "Gram Stain":"Negative","Shape":"Rods","Motility":"Positive","Spore Formation":"Negative",
            "Haemolysis Type":"Gamma","Haemolysis":"Variable","Oxidase":"Negative","Catalase":"Positive","Indole":"Negative",
            "Urease":"Negative","Citrate":"Positive","Methyl Red":"Positive","VP":"Negative","H2S":"Positive","Nitrate Reduction":"Positive",
            "Gelatin Hydrolysis":"Negative","Dnase":"Negative","Esculin Hydrolysis":"Negative","Coagulase":"Negative","Lipase Test":"Negative",
            "Growth Temperature":"37","Oxygen Requirement":"Facultative Anaerobe","NaCl Tolerant (>=6%)":"Negative",
            "Glucose Fermentation":"Positive","Maltose Fermentation":"Positive","Mannitol Fermentation":"Positive","Arabinose Fermentation":"Positive",
            "Xylose Fermentation":"Positive","Trehalose Fermentation":"Positive","Lactose Fermentation":"Negative","Sucrose Fermentation":"Negative",
            "Raffinose Fermentation":"Negative","Inositol Fermentation":"Negative","Rhamnose Fermentation":"Negative","ONPG":"Negative","Capsule":"Negative",
            "Media Grown On":"Blood Agar; Nutrient Agar; MacConkey Agar"
        },
        "Salmonella enterica (classic)"
    ),
    (
        "Gram-positive cocci. Beta-haemolytic on blood agar. Catalase positive, coagulase positive, DNase positive. "
        "Oxidase negative. Indole negative. VP positive, MR negative. Citrate variable. Urease variable. H2S negative. "
        "Grows at 37 °C; aerobic or facultative. Non-motile, non-spore-forming. "
        "Ferments glucose, mannitol, sucrose; does not ferment lactose or xylose. ONPG negative. NaCl tolerant up to 6%.",
        {
            "Gram Stain":"Positive","Shape":"Cocci","Haemolysis Type":"Beta","Haemolysis":"Positive",
            "Catalase":"Positive","Coagulase":"Positive","Dnase":"Positive","Oxidase":"Negative","Indole":"Negative",
            "VP":"Positive","Methyl Red":"Negative","Citrate":"Variable","Urease":"Variable","H2S":"Negative",
            "Growth Temperature":"37","Oxygen Requirement":"Facultative Anaerobe","Motility":"Negative","Spore Formation":"Negative",
            "Glucose Fermentation":"Positive","Mannitol Fermentation":"Positive","Sucrose Fermentation":"Positive",
            "Lactose Fermentation":"Negative","Xylose Fermentation":"Negative","ONPG":"Negative","NaCl Tolerant (>=6%)":"Positive",
            "Media Grown On":"Blood Agar; Nutrient Agar"
        },
        "Staphylococcus aureus"
    ),
    (
        "Gram-positive short rods, tumbling motility at room temperature; catalase positive, oxidase negative. "
        "Beta-haemolysis weak. Indole negative. VP negative, MR variable. Urease negative, citrate negative, H2S negative. "
        "Grows at 4 °C but not 45 °C. Facultative anaerobe. Non-spore-forming. "
        "Ferments glucose, maltose; does not ferment lactose, xylose, or mannitol. ONPG negative. Esculin hydrolysis positive.",
        {
            "Gram Stain":"Positive","Shape":"Short Rods","Motility":"Positive","Catalase":"Positive","Oxidase":"Negative",
            "Haemolysis Type":"Beta","Haemolysis":"Positive","Indole":"Negative","VP":"Negative","Methyl Red":"Variable",
            "Urease":"Negative","Citrate":"Negative","H2S":"Negative","Oxygen Requirement":"Facultative Anaerobe","Spore Formation":"Negative",
            "Glucose Fermentation":"Positive","Maltose Fermentation":"Positive","Lactose Fermentation":"Negative","Xylose Fermentation":"Negative",
            "Mannitol Fermentation":"Negative","ONPG":"Negative","Esculin Hydrolysis":"Positive","Growth Temperature":"4",
            "Media Grown On":"Blood Agar; Nutrient Agar"
        },
        "Listeria monocytogenes"
    ),
    (
        "Gram-negative rods, oxidase positive, catalase positive, indole negative. Motile, non-fermenter on MacConkey (NLF). "
        "Produces pigments; beta-haemolysis may be observed. Urease variable, citrate positive, H2S negative. "
        "Aerobic. Grows at 37 °C and 42 °C. Gelatin hydrolysis positive. DNase negative. Nitrate reduced. "
        "Does not ferment lactose, sucrose, or mannitol; glucose fermentation negative or variable. ONPG negative. "
        "NaCl tolerance variable.",
        {
            "Gram Stain":"Negative","Shape":"Rods","Oxidase":"Positive","Catalase":"Positive","Indole":"Negative","Motility":"Positive",
            "Haemolysis Type":"Beta","Haemolysis":"Positive",
            "Urease":"Variable","Citrate":"Positive","H2S":"Negative","Oxygen Requirement":"Aerobic","Growth Temperature":"37",
            "Gelatin Hydrolysis":"Positive","Dnase":"Negative","Nitrate Reduction":"Positive",
            "Lactose Fermentation":"Negative","Sucrose Fermentation":"Negative","Mannitol Fermentation":"Negative","Glucose Fermentation":"Variable","ONPG":"Negative",
            "NaCl Tolerant (>=6%)":"Variable","Media Grown On":"MacConkey Agar; Blood Agar"
        },
        "Pseudomonas aeruginosa"
    ),
    (
        "Gram-negative, motile rods. Facultative anaerobe. Colonies on nutrient agar are 2–3 mm, smooth, convex, moist, grey-cream and slightly translucent. "
        "Grows on Blood, Nutrient and MacConkey agar (pink colonies). Catalase positive, oxidase negative. Indole negative. MR negative, VP positive. "
        "Citrate positive. Urease variable. H2S not produced. Nitrate reduced. Gelatin hydrolysis variable. Esculin hydrolysis positive. DNase negative. ONPG positive. "
        "Grows at 37 °C. Lysine, ornithine and arginine decarboxylases positive. Coagulase negative. Lipase negative. "
        "Fermented: lactose, glucose, sucrose, mannitol, xylose, arabinose, inositol, maltose, trehalose; raffinose variable. "
        "Haemolysis not observed.",
        {
            "Gram Stain":"Negative","Shape":"Rods","Motility":"Positive","Oxygen Requirement":"Facultative Anaerobe",
            "Colony Morphology":"2/3mm; Smooth; Convex; Moist; Grey; Cream; Translucent",
            "Media Grown On":"Nutrient Agar; MacConkey Agar; Blood Agar",
            "Catalase":"Positive","Oxidase":"Negative","Indole":"Negative","Methyl Red":"Negative","VP":"Positive",
            "Citrate":"Positive","Urease":"Variable","H2S":"Negative","Nitrate Reduction":"Positive",
            "Gelatin Hydrolysis":"Variable","Esculin Hydrolysis":"Positive","Dnase":"Negative","ONPG":"Positive",
            "Growth Temperature":"37","Coagulase":"Negative","Lipase Test":"Negative",
            "Lysine Decarboxylase":"Positive","Ornitihine Decarboxylase":"Positive","Arginine dihydrolase":"Positive",
            "Lactose Fermentation":"Positive","Glucose Fermentation":"Positive","Sucrose Fermentation":"Positive","Mannitol Fermentation":"Positive",
            "Xylose Fermentation":"Positive","Arabinose Fermentation":"Positive","Inositol Fermentation":"Positive","Maltose Fermentation":"Positive",
            "Trehalose Fermentation":"Positive","Raffinose Fermentation":"Variable",
            "Haemolysis Type":"Gamma","Haemolysis":"Variable"
        },
        "Enterobacter cloacae complex"
    ),
]

def _default_db_fields_for_tests() -> List[str]:
    return [
        "Genus","Gram Stain","Shape","Catalase","Oxidase","Colony Morphology","Haemolysis","Haemolysis Type","Indole",
        "Growth Temperature","Media Grown On","Motility","Capsule","Spore Formation","Oxygen Requirement","Methyl Red","VP",
        "Citrate","Urease","H2S","Lactose Fermentation","Glucose Fermentation","Sucrose Fermentation","Nitrate Reduction",
        "Lysine Decarboxylase","Ornitihine Decarboxylase","Arginine dihydrolase","Gelatin Hydrolysis","Esculin Hydrolysis",
        "Dnase","ONPG","NaCl Tolerant (>=6%)","Lipase Test","Xylose Fermentation","Rhamnose Fermentation","Mannitol Fermentation",
        "Sorbitol Fermentation","Maltose Fermentation","Arabinose Fermentation","Raffinose Fermentation","Inositol Fermentation",
        "Trehalose Fermentation","Coagulase"
    ]

def run_gold_tests(verbose: bool = True) -> bool:
    cols = _default_db_fields_for_tests()
    passed = 0
    for i, (para, expected, name) in enumerate(GOLD_SPEC, start=1):
        pred = parse_input_free_text(para, db_fields=cols)
        missing, mismatched = _diff_expected_vs_got(expected, pred)
        ok = (not missing and not mismatched)
        if verbose:
            print("="*80)
            print(f"[{i}] {'PASS' if ok else 'FAIL'} — {name}")
            print("Predicted JSON:\n", json.dumps(pred, indent=2, ensure_ascii=False))
            if not ok:
                if missing:
                    print("\nMissing keys:\n", missing)
                if mismatched:
                    print("\nMismatched keys (pred vs exp):")
                    for (k, e, g) in mismatched:
                        print(f" - {k}: pred='{g}'  exp='{e}'")
        if not ok:
            append_feedback_case(name, para, expected, pred, FEEDBACK_PATH)
        passed += int(ok)
    if verbose:
        print("="*80)
        print(f"Gold Spec: {passed}/{len(GOLD_SPEC)} passed.")
        print(f"Feedback written to {FEEDBACK_PATH} for failures.")
    return passed == len(GOLD_SPEC)

# ──────────────────────────────────────────────────────────────────────────────
# Self-learning analyzer (3-strike rule → memory)
# ──────────────────────────────────────────────────────────────────────────────
def analyze_feedback_and_learn(feedback_path: str = FEEDBACK_PATH, memory_path: str = MEMORY_PATH):
    """
    Scans parser_feedback.json and increments counters per (field, expected→got).
    When a field accumulates ≥3 mismatches, we propose a stronger regex and
    store that into parser_memory.json under "auto_heuristics".
    """
    if not os.path.exists(feedback_path):
        print("ℹ️ No feedback file found.")
        return

    try:
        feedback = _load_json(feedback_path, default=[])
        if not feedback:
            print("ℹ️ Feedback file is empty.")
            return
    except Exception as e:
        print(f"⚠️ Feedback load failed: {e}")
        return

    # Count recurring issues
    pair_counts: Dict[str, int] = {}  # key = f"{field}|{expected}|{got}"
    field_counts: Dict[str, int] = {}
    suggestions: List[str] = []

    for case in feedback:
        for err in case.get("errors", []):
            field = err.get("field","").strip()
            exp = (err.get("expected") or "").strip().lower()
            got = (err.get("got") or "").strip().lower()
            if not field:
                continue
            field_counts[field] = field_counts.get(field, 0) + 1
            key = f"{field}|{exp}|{got}"
            pair_counts[key] = pair_counts.get(key, 0) + 1
            # Heuristic: only suggest if exp vs got are substantively different
            sim = difflib.SequenceMatcher(None, got, exp).ratio()
            if sim < 0.6:
                suggestions.append(f"Consider adjusting pattern for '{field}' — often parsed '{got}' instead of '{exp}'")

    auto_heuristics = {}
    for key, count in pair_counts.items():
        field, exp, got = key.split("|", 2)
        if count >= 3:
            # Build a generic learned rule description
            rule_text = ""
            if any(tok in exp for tok in ["positive","+","detected","produced"]):
                rule_text = f"Add stronger POSITIVE regex for '{field}' (seen {count}×)"
            elif any(tok in exp for tok in ["negative","-","not","absent"]):
                rule_text = f"Add stronger NEGATIVE regex for '{field}' (seen {count}×)"
            else:
                rule_text = f"Add stronger pattern for '{field}' (seen {count}×)"
            # Save both the rule and the preferred polarity guess
            polarity_hint = "positive" if "positive" in exp or "+" in exp or "detected" in exp or "produced" in exp else \
                            "negative" if "negative" in exp or "-" in exp or "not" in exp or "absent" in exp else "generic"

            auto_heuristics[field] = {
                "rule": rule_text,
                "count": count,
                "expected_hint": exp,
                "polarity_hint": polarity_hint
            }

    memory = _load_json(memory_path, default={})
    memory.setdefault("history", []).append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_error_fields": sorted(field_counts.items(), key=lambda x: -x[1])[:10],
        "suggestions": suggestions[:20]
    })
    memory["auto_heuristics"] = auto_heuristics
    _save_json(memory_path, memory)
    print(f"🧠 Learned {len(suggestions)} hints; {len(auto_heuristics)} fields reached 3-strike threshold.")

# ──────────────────────────────────────────────────────────────────────────────
# Auto-updating regex patcher (inject learned patterns into this file)
# ──────────────────────────────────────────────────────────────────────────────
def auto_update_parser_regex(memory_path: str = MEMORY_PATH, parser_file: str = "parser_basic.py"):
    """
    Reads parser_memory.json and injects concrete regex lines into the pattern lists:
      CATALASE_PATTERNS, OXIDASE_PATTERNS, …, FERMENTATION_PATTERNS
    For fermentation fields, we embed the sugar base in the regex so the extractor
    can infer which column to set.
    """
    if not os.path.exists(memory_path):
        print("ℹ️ No parser memory file found; skipping regex update.")
        return

    mem = _load_json(memory_path, default={})
    auto_heuristics = mem.get("auto_heuristics", {})
    if not auto_heuristics:
        print("ℹ️ No new regex heuristics to apply.")
        return

    try:
        with open(parser_file, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"❌ Could not read parser file: {e}")
        return

    # Map field name → pattern list variable
    pattern_lists = {
        "oxidase": "OXIDASE_PATTERNS",
        "catalase": "CATALASE_PATTERNS",
        "indole": "INDOLE_PATTERNS",
        "vp": "VP_PATTERNS",
        "methyl red": "MR_PATTERNS",
        "urease": "UREASE_PATTERNS",
        "citrate": "CITRATE_PATTERNS",
        "h2s": "H2S_PATTERNS",
        "coagulase": "COAGULASE_PATTERNS",
        "lipase": "LIPASE_PATTERNS",
        "esculin": "ESCULIN_PATTERNS",
        "dnase": "DNASE_PATTERNS",
        "gelatin": "GELATIN_PATTERNS",
        "nitrate": "NITRATE_PATTERNS",
        "decarboxylase": "DECARBOXYLASE_PATTERNS",
        # Fermentations handled specially below
    }

    updated = 0
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def inject_into_list(list_name: str, regex_literal: str):
        nonlocal code, updated
        pattern_block_regex = rf"({list_name}\s*:\s*List\[str\]\s*=\s*\[|{list_name}\s*=\s*\[)([^\]]*)(\])"
        new_code, count = re.subn(
            pattern_block_regex,
            rf"\1\2    {regex_literal},  # auto-learned {ts}\n\3",
            code,
            flags=re.S,
        )
        if count > 0:
            code = new_code
            updated += 1

    for field, info in auto_heuristics.items():
        fld_low = field.lower()

        # Fermentation fields: create patterns that include the sugar base
        if "fermentation" in fld_low:
            base = re.sub(r"\s*fermentation\s*$", "", fld_low).strip()
            if not base:
                continue
            pol = info.get("polarity_hint", "generic").lower()
            if pol == "positive":
                learned_regex = rf'r"\b{re.escape(base)}\b[^.]*?(?:\+|positive|detected|produced)\b"'
            elif pol == "negative":
                learned_regex = rf'r"\b(?:non[-\s]?ferment(?:er)?\s+of\s+|no\s+fermentation\s+of\s+|{re.escape(base)}\b[^.]*?(?:\-|negative|not\s+detected|not\s+ferment))"'
            else:
                learned_regex = rf'r"\b{re.escape(base)}\b\s+reaction\b"'
            inject_into_list("FERMENTATION_PATTERNS", learned_regex)
            continue

        # Non-fermentation: map to a list name
        list_name = None
        for key, lname in pattern_lists.items():
            if key in fld_low:
                list_name = lname
                break
        if not list_name:
            continue

        pol = info.get("polarity_hint", "generic").lower()
        if pol == "positive":
            learned_regex = rf'r"\b{re.escape(fld_low)}\b\s*(?:test\s*)?(?:\+|positive|detected|produced)\b"'
        elif pol == "negative":
            learned_regex = rf'r"\b{re.escape(fld_low)}\b\s*(?:test\s*)?(?:\-|negative|not\s+detected|absent|not\s+produced)\b"'
        else:
            learned_regex = rf'r"\b{re.escape(fld_low)}\b\s+reaction\b"'

        inject_into_list(list_name, learned_regex)

    if updated > 0:
        # Append a summary log at EOF
        code += f"\n\n# === AUTO-LEARNED PATTERNS SUMMARY ({ts}) ===\n"
        for f, r in auto_heuristics.items():
            code += f"# {f}: {r.get('rule')} (seen {r.get('count')}×; hint={r.get('polarity_hint')})\n"
        try:
            with open(parser_file, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"🧩 Injected {updated} learned regex snippets into {parser_file}.")
        except Exception as e:
            print(f"❌ Failed to write updates: {e}")
    else:
        print("ℹ️ No matching pattern lists for learned items; no changes made.")

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
        wi_idx = sys.argv.index("--whatif")
        wi_text = sys.argv[wi_idx+1] if wi_idx+1 < len(sys.argv) else "set catalase to negative"
        cols = _default_db_fields_for_tests()
        base_para, expected, name = GOLD_SPEC[0]
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

