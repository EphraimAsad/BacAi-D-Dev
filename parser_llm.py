# parser_llm.py — v4 (Ollama-only + Full Regex + Self-Learning + Auto-Patching + Gold Tests)
# ──────────────────────────────────────────────────────────────────────────────
# What you get:
#   • Ollama Cloud LLM parsing (primary) with JSON extraction
#   • Deterministic regex enrichment for morphology/biochem/fermentations/media
#   • Persistent self-learning from gold_tests + live runs (3-strike rule)
#   • Auto-injection of learned regex lines back into this file (permanent)
#   • CLI: python parser_llm.py --test  → runs gold_tests.json, learns, patches
#
# Files used:
#   • gold_tests.json          ← optional input for gold-testing
#   • parser_feedback.json     ← logs mismatches from tests/runs
#   • parser_memory.json       ← learned summaries & auto_heuristics
#
# Env:
#   • OLLAMA_API_KEY           ← your cloud API key
#   • LOCAL_MODEL              ← default "deepseek-v3.1:671b"
#   • BACTAI_STRICT_MODE       ← "1" for strict schema-only output
#
# Public API:
#   parse_input_free_text(text, prior_facts=None, db_fields=None) -> Dict[str,str]
#   apply_what_if(user_text, prior_result, db_fields) -> Dict[str,str]
#   run_gold_tests()  # CLI with --test
#   analyze_feedback_and_learn()  # called automatically by runner helpers
#   auto_update_parser_regex()    # writes learned regex into *_PATTERNS lists
#   enable_self_learning_autopatch(run_tests: bool = False)
#
# NOTE: This module uses the installed `ollama` Python package which supports
#       Ollama Cloud when your environment has OLLAMA_API_KEY.
#       If the LLM call fails, we fall back to the deterministic regex layer.
# ──────────────────────────────────────────────────────────────────────────────

import os
import re
import json
import sys
import math
import difflib
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional

# Fallback deterministic parser (your local baseline)
try:
    from parser_basic import parse_input_free_text as fallback_parser
except Exception:
    fallback_parser = None  # We'll handle if unavailable

# ──────────────────────────────────────────────────────────────────────────────
# Persistent storage paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.getcwd(), "data")
GOLD_TESTS_PATH = os.path.join(os.getcwd(), "gold_tests.json")
FEEDBACK_PATH = os.path.join(os.getcwd(), "parser_feedback.json")
MEMORY_PATH = os.path.join(os.getcwd(), "parser_memory.json")

def _ensure_data_dir():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        pass

def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: str, obj):
    try:
        _ensure_data_dir()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Schema allowed values (aligned with your Excel)
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_VALUES: Dict[str, Set[str]] = {
    "Gram Stain": {"Positive", "Negative", "Variable"},
    "Shape": {"Cocci", "Rods", "Bacilli", "Spiral", "Short Rods"},
    "Catalase": {"Positive", "Negative", "Variable"},
    "Oxidase": {"Positive", "Negative", "Variable"},
    "Colony Morphology": set(),
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

# Media whitelist (for clamping names; we still accept others; exclude TSI explicitly)
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
MEDIA_EXCLUDE_TERMS = {"tsi", "triple sugar iron"}

# Abbreviations
BIOCHEM_ABBR = {
    "mr": "Methyl Red",
    "vp": "VP",
    "ldc": "Lysine Decarboxylase",
    "odc": "Ornitihine Decarboxylase",
    "adh": "Arginine dihydrolase",
    "lf": "Lactose Fermentation",
    "nlf": "Lactose Fermentation",
}
MEDIA_ABBR = {
    "tsa": "Tryptic Soy Agar",
    "bhi": "Brain Heart Infusion Agar",
    "cba": "Columbia Blood Agar",
    "ssa": "Blood Agar",
    "ba": "Blood Agar",
    "xld": "XLD Agar",
    "macconkey": "MacConkey Agar",
}

# Polarity fields
POLARITY_FIELDS = {
    "Catalase","Oxidase","Haemolysis","Indole","Motility","Capsule","Spore Formation",
    "Methyl Red","VP","Citrate","Urease","H2S","Lactose Fermentation","Glucose Fermentation",
    "Sucrose Fermentation","Nitrate Reduction","Lysine Decarboxylase","Ornitihine Decarboxylase",
    "Arginine dihydrolase","Gelatin Hydrolysis","Esculin Hydrolysis","Dnase","ONPG",
    "NaCl Tolerant (>=6%)","Lipase Test","Xylose Fermentation","Rhamnose Fermentation",
    "Mannitol Fermentation","Sorbitol Fermentation","Maltose Fermentation","Arabinose Fermentation",
    "Raffinose Fermentation","Inositol Fermentation","Trehalose Fermentation","Coagulase"
}

# Value synonyms & polarity
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
        "facultative anaerobic": "Facultative Anaerobe",
        "aerobe": "Aerobic","aerobic": "Aerobic",
        "anaerobe": "Anaerobic","anaerobic": "Anaerobic",
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

# Negation & variable cues (±5 token window)
NEGATION_CUES = [
    "not produced","no production","not observed","none observed","no reaction",
    "absent","without production","not detected","does not","did not","fails to",
    "unable to","no growth","non-fermenter","nonfermenter","non fermenter","no haemolysis","no hemolysis"
]
VARIABLE_CUES = ["variable","inconsistent","weak","trace","slight","equivocal","irregular"]

# Colony morphology controlled vocab
CM_TOKENS = {
    "1/3mm","1/2mm","2/3mm","2/4mm","0.5/1mm","0.5mm/2mm","1mm","2mm","3mm",
    "tiny","small","medium","large","pinpoint","subsurface","satellite",
    "round","circular","convex","flat","domed","heaped","fried egg",
    "smooth","rough","wrinkled","granular","mucoid","glistening","dull","matte",
    "shiny","sticky","adherent","powdery","chalk","leathery","velvet","crumbly",
    "ground glass","irregular edges","spreading","swarming","corrode","pit","ropey","butyrous","waxy",
    "opaque","translucent","colourless","colorless",
    "dry","moist",
    "white","grey","gray","cream","off-white","yellow","pale yellow","orange","pink","coral","red","green",
    "violet","purple","black","brown","beige","tan","blue",
    "bright","pigmented","iridescent","ring","dingers ring"
}

# ──────────────────────────────────────────────────────────────────────────────
# Pattern lists used by the auto-patcher
# The self-learning system will append new regex lines into these lists.
# Extraction routines consult these lists in addition to hardcoded patterns.
# ──────────────────────────────────────────────────────────────────────────────

OXIDASE_PATTERNS = [
    r"\boxidase\s*(?:test)?\s*(?:\+|positive|detected)\b",
    r"\boxidase\s*(?:test)?\s*(?:\-|negative|not\s+detected|absent)\b",
]
CATALASE_PATTERNS = [
    r"\bcatalase\s*(?:test)?\s*(?:\+|positive|detected)\b",
    r"\bcatalase\s*(?:test)?\s*(?:\-|negative|not\s+detected|absent)\b",
]
COAGULASE_PATTERNS = [
    r"\bcoagulase\s*(?:test)?\s*(?:\+|positive|detected)\b",
    r"\bcoagulase\s*(?:test)?\s*(?:\-|negative|not\s+detected|absent)\b",
]
INDOLE_PATTERNS = [
    r"\bindole\s*(?:test)?\s*(?:\+|positive|detected)\b",
    r"\bindole\s*(?:test)?\s*(?:\-|negative|not\s+detected|absent)\b",
]
UREASE_PATTERNS = [
    r"\burease\s*(?:test)?\s*(?:\+|positive|detected)\b",
    r"\burease\s*(?:test)?\s*(?:\-|negative|not\s+detected|absent)\b",
    r"\burease\s*(?:test)?\s*(?:variable|weak|trace)\b",
]
CITRATE_PATTERNS = [
    r"\bcitrate\s*(?:test)?\s*(?:\+|positive)\b",
    r"\bcitrate\s*(?:test)?\s*(?:\-|negative)\b",
]
MR_PATTERNS = [
    r"\bmethyl\s+red\s*(?:test)?\s*(?:\+|positive)\b",
    r"\bmethyl\s+red\s*(?:test)?\s*(?:\-|negative)\b",
    r"\bmr\s*(?:test)?\s*(?:\+|positive)\b",
    r"\bmr\s*(?:test)?\s*(?:\-|negative)\b",
]
VP_PATTERNS = [
    r"\bvp\s*(?:test)?\s*(?:\+|positive)\b",
    r"\bvp\s*(?:test)?\s*(?:\-|negative)\b",
    r"\bvoges[\s\-]?proskauer\s*(?:test)?\s*(?:\+|positive)\b",
    r"\bvoges[\s\-]?proskauer\s*(?:test)?\s*(?:\-|negative)\b",
]
H2S_PATTERNS = [
    r"\bh\s*2\s*s\s+(?:\+|positive|detected|produced)\b",
    r"\bh\s*2\s*s\s+(?:\-|negative|not\s+detected|not\s+produced)\b",
    r"\bproduces\s+h\s*2\s*s\b",
    r"\bno\s+h\s*2\s*s\s+production\b",
    r"\\bh2s\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (5x)
]
NITRATE_PATTERNS = [
    r"\breduces\s+nitrate\b",
    r"\bdoes\s+not\s+reduce\s+nitrate\b",
    r"\bnitrate\s+reduction\s+(?:\+|positive)\b",
    r"\bnitrate\s+reduction\s+(?:\-|negative)\b",
]
ESCULIN_PATTERNS = [
    r"\besculin\s+hydrolysis\s*(?:\+|positive)\b",
    r"\besculin\s+hydrolysis\s*(?:\-|negative)\b",
    r"\\besculin\ hydrolysis\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (4x)
]
DNASE_PATTERNS = [
    r"\bdnase\s*(?:test)?\s*(?:\+|positive)\b",
    r"\bdnase\s*(?:test)?\s*(?:\-|negative)\b",
    r"\\bdnase\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (4x)
]
GELATIN_PATTERNS = [
    r"\bgelatin\s+(?:liquefaction|hydrolysis)\s*(?:\+|positive)\b",
    r"\bgelatin\s+(?:liquefaction|hydrolysis)\s*(?:\-|negative)\b",
    r"\\bgelatin\ hydrolysis\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (15x)
]
LIPASE_PATTERNS = [
    r"\blipase\s*(?:test)?\s*(?:\+|positive)\b",
    r"\blipase\s*(?:test)?\s*(?:\-|negative)\b",
]
DECARBOXYLASE_PATTERNS = [
    r"\blysine\s+decarboxylase\s+(?:\+|positive|detected)\b",
    r"\blysine\s+decarboxylase\s+(?:\-|negative|not\s+detected)\b",
    r"\bornithine\s+decarboxylase\s+(?:\+|positive|detected)\b",
    r"\bornithine\s+decarboxylase\s+(?:\-|negative|not\s+detected)\b",
    r"\bornitihine\s+decarboxylase\s+(?:\+|positive|detected)\b",
    r"\bornitihine\s+decarboxylase\s+(?:\-|negative|not\s+detected)\b",
    r"\barginine\s+dihydrolase\s+(?:\+|positive|detected)\b",
    r"\barginine\s+dihydrolase\s+(?:\-|negative|not\s+detected)\b",
    r"\b(ldc|odc|adh)\s*(?:\+|positive|-|negative)\b",
    r"\\bornitihine\ decarboxylase\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (4x)
]
FERMENTATION_PATTERNS = [
    r"(?:ferments?|utilizes?|produces?\s+acid\s+from)\s+([a-z0-9\.\-%\s,/&    r"\\blactose\ fermentation\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (71x)
    r"\\bxylose\ fermentation\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (16x)
    r"\\bmannitol\ fermentation\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (32x)
    r"\\bsucrose\ fermentation\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (39x)
    r"\\bsorbitol\ fermentation\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (10x)
    r"\\barabinose\ fermentation\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (4x)
    r"\\bglucose\ fermentation\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (83x)
    r"\\bfructose\ fermentation\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (3x)
    r"\\bmaltose\ fermentation\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (5x)
    r"\\brhamnose\ fermentation\\b.*(?:positive|detected|produced)",  # auto-learned 2025-11-10 21:17:08 (5x)
]+)(?:\.|;|,|$)",
    r"(?:does\s+not|doesn't|cannot|unable\s+to)\s+(?:ferment|utilize)\s+([a-z0-9\.\-%\s,/&]+)",
    r"(?:ferments?|utilizes?)[^.]*?\bbut\s+not\s+([\w\s,;.&-]+)",
    r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b",  # shorthand
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: columns, normalization, aliases, tokens
# ──────────────────────────────────────────────────────────────────────────────

_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def normalize_columns(db_fields: List[str]) -> List[str]:
    return [f for f in (db_fields or []) if f and f.strip().lower() != "genus"]

def normalize_text(raw: str) -> str:
    t = raw or ""
    t = t.replace("°", " °")
    t = t.translate(_SUBSCRIPT_DIGITS)
    t = (t.replace("\u2010","-").replace("\u2011","-").replace("\u2012","-")
           .replace("\u2013","-").replace("\u2014","-").replace("–","-").replace("—","-"))
    t = re.sub(r"hemolys", "haemolys", t, flags=re.I)   # US→UK
    t = re.sub(r"gray", "grey", t, flags=re.I)          # prefer grey
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
    items = [t.strip() for t in s.split(",") if t.strip()]
    return [re.sub(r"[.,;:\s]+$", "", i) for i in items]

def _canon_value(field: str, value: str) -> str:
    v = (value or "").strip()
    if not v:
        return v
    if field in POLARITY_FIELDS:
        low = v.lower()
        pol = VALUE_SYNONYMS.get("*POLARITY*", {})
        if low in pol:
            v = pol[low]
        else:
            if re.fullmatch(r"\+|positive|pos", low): v = "Positive"
            elif re.fullmatch(r"\-|negative|neg", low): v = "Negative"
            elif any(tok in low for tok in ["weak","variable","trace","slight"]): v = "Variable"
    low = v.lower()
    if field in VALUE_SYNONYMS:
        v = VALUE_SYNONYMS[field].get(low, v)
    allowed = ALLOWED_VALUES.get(field)
    if allowed and v not in allowed:
        tv = v.title()
        if tv in allowed: v = tv
    return v

def _set_field_safe(out: Dict[str, str], key: str, val: str):
    if not val:
        return
    cur = out.get(key)
    if cur is None:
        out[key] = val
        return
    if cur == "Variable" and val in {"Positive","Negative"}:
        out[key] = val
        return
    out[key] = val

def _in_window(tokens: List[str], idx: int, window: int, cues: List[str]) -> bool:
    start = max(0, idx - window)
    end = min(len(tokens), idx + window + 1)
    span = " ".join(tokens[start:end])
    return any(c in span for c in cues)

def _any_negation_near(tokens: List[str], idx: int, window: int = 5) -> bool:
    return _in_window(tokens, idx, window, NEGATION_CUES)

def _any_variable_near(tokens: List[str], idx: int, window: int = 5) -> bool:
    return _in_window(tokens, idx, window, VARIABLE_CUES)

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
    # Sheet typo → canonical
    add("glucose fermantation","Glucose Fermentation")

    # Fermentation bases
    for f in fields:
        if f.lower().endswith(" fermentation"):
            base = f[:-12].strip().lower()
            alias[base] = f

    # Media & abbrev
    for m in MEDIA_WHITELIST:
        alias[m.lower()] = "Media Grown On"
    for abbr, full in MEDIA_ABBR.items():
        alias[abbr.lower()] = "Media Grown On"

    return alias

# ──────────────────────────────────────────────────────────────────────────────
# Colony morphology normalization
# ──────────────────────────────────────────────────────────────────────────────
def _split_color_hyphens(s: str) -> List[str]:
    parts = []
    for token in re.split(r"[;/,]", s):
        token = token.strip()
        if "-" in token and any(c in token for c in ["grey","gray","white","cream","yellow","orange","pink","red","green","blue","brown","beige","tan","black","purple","violet","off"]):
            parts.extend([p.strip() for p in token.split("-") if p.strip()])
        else:
            parts.append(token)
    return parts

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

    for s in spans:
        for mm in re.findall(r"(?:\d+(?:\.\d+)?\/\d+(?:\.\d+)?mm|\d+(?:\.\d+)?mm|0\.5\/1mm|0\.5mm\/2mm|1\/3mm|2\/3mm|2\/4mm)", s):
            add(mm)
        s_norm = " " + re.sub(r"[,;/]", " ", s) + " "
        multi = ["ground glass","irregular edges","fried egg","dingers ring","off-white","pale yellow","cream-white","grey-cream","mucoid ropey","butyrous"]
        for mword in multi:
            if f" {mword} " in s_norm:
                add(mword)
        parts = re.split(r"[,;:/\s]+", s)
        hyphen_fixed = []
        for p in parts:
            hyphen_fixed.extend(_split_color_hyphens(p))
        for p in hyphen_fixed:
            low = p.strip().lower()
            if low == "colorless": low = "colourless"
            if low in CM_TOKENS:
                add(low)
            if low in {"off-white","pale","pale-yellow","cream-white","grey-cream","ropey","butyrous"}:
                add(low.replace("-", " "))

    order_groups = [
        {"1/3mm","1/2mm","2/3mm","2/4mm","0.5/1mm","0.5mm/2mm","1mm","2mm","3mm","tiny","small","medium","large","pinpoint","subsurface","satellite"},
        {"round","circular","convex","flat","domed","heaped","fried egg"},
        {"smooth","rough","wrinkled","granular","mucoid","glistening","dull","matte","shiny","sticky","adherent","powdery","chalk","leathery","velvet","crumbly",
         "ground glass","irregular edges","spreading","swarming","corrode","pit","ring","dingers ring","bright","pigmented","ropey","butyrous","waxy"},
        {"opaque","translucent","colourless"},
        {"dry","moist"},
        {"white","grey","gray","cream","off-white","yellow","pale yellow","orange","pink","coral","red","green","violet","purple","black","brown","beige","tan","blue"},
    ]
    ordered: List[str] = []
    seen = set()
    for grp in order_groups:
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
        if item == "Cream; White": flat.extend(["Cream","White"])
        else: flat.append(item)
    return "; ".join(flat)

# ──────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ──────────────────────────────────────────────────────────────────────────────
def _summarize_field_categories(db_fields: List[str]) -> Dict[str, List[str]]:
    cats = {"Morphology": [], "Enzyme": [], "Fermentation": [], "Other": []}
    for f in normalize_columns(db_fields):
        l = f.lower()
        if any(k in l for k in ["gram","shape","morphology","motility","capsule","spore","oxygen requirement","media grown"]):
            cats["Morphology"].append(f)
        elif any(k in l for k in ["oxidase","catalase","urease","coagulase","lipase","indole","citrate","vp","methyl red","gelatin","dnase","nitrate","h2s","esculin"]):
            cats["Enzyme"].append(f)
        elif "fermentation" in l or "utilization" in l:
            cats["Fermentation"].append(f)
        else:
            cats["Other"].append(f)
    return cats

def build_prompt_text(user_text: str, cats: Dict[str, List[str]], prior_facts=None) -> str:
    prior = json.dumps(prior_facts or {}, indent=2)
    morph = ", ".join(cats["Morphology"][:10])
    enz = ", ".join(cats["Enzyme"][:10])
    ferm = ", ".join(cats["Fermentation"][:10])
    other = ", ".join(cats["Other"][:10])
    return (
        "Parse the observation into JSON of microbiology fields. "
        "Focus on morphology, enzyme tests, oxygen/growth, and media. "
        "Fermentations can be left if ambiguous; we also apply rule-based parsing. "
        "Return compact JSON with keys from the provided schema; unmentioned fields='Unknown'.\n"
        f"Schema hints — Morphology: {morph}\n"
        f"Enzyme: {enz}\n"
        f"Fermentation: {ferm}\n"
        f"Other: {other}\n\n"
        f"Previous facts:\n{prior}\n\n"
        f"Observation:\n{user_text}"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Apply learned pattern lists (from *_PATTERNS) during extraction
# ──────────────────────────────────────────────────────────────────────────────
def _apply_learned_patterns(field_name: str, patterns: List[str], text: str, tokens: List[str], out: Dict[str,str], alias: Dict[str,str]):
    """
    For each pattern, decide Positive/Negative/Variable by presence of cue words.
    This is what makes auto-inserted rules actually affect parsing.
    """
    key = alias.get(field_name.lower(), field_name)
    if not key:
        key = field_name  # fallback

    for pat in patterns:
        try:
            for m in re.finditer(pat, text, flags=re.I|re.S):
                span = m.group(0).lower()
                val = None
                if re.search(r"\b(\+|positive|detected|produced)\b", span): val = "Positive"
                if re.search(r"\b(\-|negative|not\s+detected|absent|not\s+produced)\b", span): val = "Negative"
                if re.search(r"\b(variable|weak|trace|slight|inconsistent)\b", span): val = "Variable"
                if val:
                    _set_field_safe(out, key, _canon_value(key, val))
        except re.error:
            # If a learned regex is malformed, skip gracefully
            continue

# ──────────────────────────────────────────────────────────────────────────────
# Fermentation extraction (+ shorthands + negatives and "but not" lists)
# ──────────────────────────────────────────────────────────────────────────────
def extract_fermentations_regex(text: str, db_fields: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    t = normalize_text(text)
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)
    tokens = t.split()

    ferm_fields = [f for f in fields if f.lower().endswith(" fermentation")]
    base_to_field = {f[:-12].strip().lower(): f for f in ferm_fields}

    def set_field_by_base(base: str, val: str):
        b = _normalize_token(base)
        if b in base_to_field:
            _set_field_safe(out, base_to_field[b], _canon_value(base_to_field[b], val))
        elif b in alias and alias[b] in fields:
            _set_field_safe(out, alias[b], _canon_value(alias[b], val))

    # 1) Learned generic patterns (from FERMENTATION_PATTERNS)
    for pat in FERMENTATION_PATTERNS:
        for m in re.finditer(pat, t, flags=re.I|re.S):
            grp = m.groups()
            if not grp: 
                continue
            # 1a) Positive lists: group(1) — tokens until punctuation
            if len(grp) >= 1 and grp[0]:
                span = grp[0]
                # cut off "but not ..." inside the same positive sentence
                span = re.split(r"(?i)\bbut\s+not\b", span)[0]
                for a in _tokenize_list(span):
                    set_field_by_base(a, "Positive")
            # 1b) "but not ..." captured in group(1) for second regex
            # handled by explicit negative block below

    # 2) Explicit negative lists
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

    # 3) Negatives after "but not …"
    for m in re.finditer(r"(?:ferments?|utilizes?)[^.]*?\bbut\s+not\s+([\w\s,;.&-]+)", t, flags=re.I):
        seg = m.group(1)
        seg = re.sub(r"\bor\b", ",", seg, flags=re.I)
        seg = re.sub(r"\bnor\b", ",", seg, flags=re.I)
        for a in _tokenize_list(seg):
            set_field_by_base(a, "Negative")

    # 4) Shorthand: "lactose +" / "rhamnose -"
    for m in re.finditer(r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b", t, flags=re.I):
        a, sign = m.group(1), m.group(2)
        set_field_by_base(a, "Positive" if sign == "+" else "Negative")

    # 5) Variable adjectives per-sugar
    for base in list(base_to_field.keys()):
        if re.search(rf"\b{re.escape(base)}\b\s+(?:variable|inconsistent|weak|trace|slight|irregular)", t, flags=re.I):
            set_field_by_base(base, "Variable")

    # 6) LF/NLF short forms
    if re.search(r"\bnlf\b", t):
        set_field_by_base("lactose", "Negative")
    if re.search(r"\blf\b", t) and "Lactose Fermentation" not in out:
        set_field_by_base("lactose", "Positive")

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Biochemical / morphology / oxygen / media extraction
# Includes consultation of learned pattern lists for each field.
# ──────────────────────────────────────────────────────────────────────────────
def extract_biochem_regex(text: str, db_fields: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    raw = text or ""
    t = normalize_text(raw)
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)
    tokens = t.split()

    def set_field(k_like: str, val: str):
        target = alias.get(k_like.lower(), k_like)
        if target in fields:
            _set_field_safe(out, target, _canon_value(target, val))

    # 0) Apply learned pattern lists up front, so they can set values early
    _apply_learned_patterns("oxidase", OXIDASE_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("catalase", CATALASE_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("coagulase", COAGULASE_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("indole", INDOLE_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("urease", UREASE_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("citrate", CITRATE_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("methyl red", MR_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("vp", VP_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("h2s", H2S_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("nitrate reduction", NITRATE_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("esculin hydrolysis", ESCULIN_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("dnase", DNASE_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("gelatin hydrolysis", GELATIN_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("lipase test", LIPASE_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("lysine decarboxylase", DECARBOXYLASE_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("ornithine decarboxylase", DECARBOXYLASE_PATTERNS, t, tokens, out, alias)
    _apply_learned_patterns("arginine dihydrolase", DECARBOXYLASE_PATTERNS, t, tokens, out, alias)

    # 1) Gram
    if re.search(r"\bgram[-\s]?positive\b", t) and not re.search(r"\bgram[-\s]?negative\b", t):
        set_field("gram stain", "Positive")
    elif re.search(r"\bgram[-\s]?negative\b", t) and not re.search(r"\bgram[-\s]?positive\b", t):
        set_field("gram stain", "Negative")

    # 2) Shape
    if re.search(r"\bcocci\b", t): set_field("shape", "Cocci")
    if re.search(r"\brods?\b|bacilli\b", t): set_field("shape", "Rods")
    if re.search(r"\bspiral\b", t): set_field("shape", "Spiral")
    if re.search(r"\bshort\s+rods\b", t): set_field("shape", "Short Rods")

    # 3) Motility (positive/negative)
    if re.search(r"\bnon[-\s]?motile\b", t): set_field("motility", "Negative")
    elif re.search(r"\bmotile\b", t): set_field("motility", "Positive")

    # 4) Capsule
    if re.search(r"\b(capsulated|encapsulated)\b", t):
        set_field("capsule", "Positive")
    if re.search(r"\bnon[-\s]?capsulated\b|\bcapsule\s+absent\b", t):
        set_field("capsule", "Negative")
    if re.search(r"\bcapsule\s+(?:variable|inconsistent|weak)\b", t):
        set_field("capsule", "Variable")

    # 5) Spore formation
    if re.search(r"\bnon[-\s]?spore[-\s]?forming\b|\bno\s+spores?\b", t):
        set_field("spore formation", "Negative")
    if re.search(r"\bspore[-\s]?forming\b|\bspores?\s+present\b", t):
        set_field("spore formation", "Positive")

    # 6) Oxygen requirement
    if re.search(r"\bintracellular\b", t): set_field("oxygen requirement", "Intracellular")
    elif re.search(r"\bcapnophil(ic|e)\b", t): set_field("oxygen requirement", "Capnophilic")
    elif re.search(r"\bmicroaerophil(ic|e)\b", t): set_field("oxygen requirement", "Microaerophilic")
    elif re.search(r"\bfacultative\b", t) or re.search(r"\bfacultative\s+anaerob", t):
        set_field("oxygen requirement", "Facultative Anaerobe")
    elif re.search(r"\baerobic\b", t): set_field("oxygen requirement", "Aerobic")
    elif re.search(r"\banaerobic\b", t): set_field("oxygen requirement", "Anaerobic")

    # 7) Generic enzyme tests (beyond learned lists)
    generic_tests = ["catalase","oxidase","coagulase","urease","lipase","indole","citrate","vp","methyl red","gelatin","dnase","nitrate reduction","nitrate","h2s","esculin hydrolysis","onpg"]
    for test in generic_tests:
        # Positive
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:\+|positive|detected|produced)\b", t):
            set_field(test, "Positive")
        # Negative
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:\-|negative|not\s+detected|not\s+produced|absent)\b", t):
            set_field(test, "Negative")
        # Variable
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:variable|weak|trace|slight)\b", t):
            set_field(test, "Variable")

    # 8) Special H2S phrasing
    if re.search(r"\bh\s*2\s*s\s+not\s+produced\b", t): set_field("h2s", "Negative")
    if re.search(r"\bproduces\s+h\s*2\s*s\b", t): set_field("h2s", "Positive")

    # Nitrate alternate phrasing
    if re.search(r"\breduces\s+nitrate\b", t): set_field("nitrate reduction", "Positive")
    if re.search(r"\bdoes\s+not\s+reduce\s+nitrate\b", t): set_field("nitrate reduction", "Negative")

    # 9) Haemolysis Type
    if re.search(r"\b(beta|β)[-\s]?haem", t):
        set_field("haemolysis type", "Beta")
    elif re.search(r"\b(alpha|α)[-\s]?haem", t):
        set_field("haemolysis type", "Alpha")
    elif re.search(r"\b(gamma|γ)[-\s]?haem\b", t) or re.search(r"\bno\s+haemolysis\b|\bhaemolysis\s+not\s+observed\b", t):
        set_field("haemolysis type", "Gamma")

    # 10) Growth temperature (ranges and singles)
    range1 = re.search(r"grows\s+(\d{1,2})\s*(?:–|-|to)\s*(\d{1,2})\s*°?\s*c", t)
    range2 = re.search(r"growth\s+(?:between|from)\s+(\d{1,2})\s*(?:and|to)\s*(\d{1,2})\s*°?\s*c", t)
    if range1:
        low, high = range1.group(1), range1.group(2)
        set_field("growth temperature", f"{low}//{high}")
    elif range2:
        low, high = range2.group(1), range2.group(2)
        set_field("growth temperature", f"{low}//{high}")
    for m in re.finditer(r"(?<!no\s)grows\s+(?:well\s+)?at\s+([0-9]{1,3})\s*°?\s*c", t):
        if out.get("Growth Temperature", "").find("//") == -1:
            set_field("growth temperature", m.group(1))

    # 11) NaCl tolerant
    if re.search(r"\b(tolerant|grows|growth)\s+(?:in|up\s+to|to|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t): set_field("nacl tolerant (>=6%)", "Positive")
    if re.search(r"\bno\s+growth\s+(?:in|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t): set_field("nacl tolerant (>=6%)", "Negative")
    if re.search(r"\bnacl\s+tolerant\b", t): set_field("nacl tolerant (>=6%)", "Positive")

    # 12) Media detection (exclude TSI)
    collected_media: List[str] = []
    candidate_media = set()
    for name in ["blood", "macconkey", "xld", "nutrient", "tsa", "bhi", "cba", "ba", "ssa", "chocolate", "emb"]:
        if re.search(rf"\b{name}\b", t): candidate_media.add(name)
    for m in re.finditer(r"\b([a-z0-9\-\+ ]+)\s+agar\b", t):
        lowname = m.group(1).strip().lower()
        if not any(ex in lowname for ex in MEDIA_EXCLUDE_TERMS):
            candidate_media.add(lowname + " agar")

    def canon_media(name: str) -> Optional[str]:
        if name in {"xld"}: return "XLD Agar"
        if name in {"macconkey"}: return "MacConKey Agar" if False else "MacConkey Agar"  # keep as MacConkey
        if name in {"blood","ba","ssa"}: return "Blood Agar"
        if name == "nutrient": return "Nutrient Agar"
        if name == "tsa": return "Tryptic Soy Agar"
        if name == "bhi": return "Brain Heart Infusion Agar"
        if name == "cba": return "Columbia Blood Agar"
        if name.endswith(" agar"):
            return name[:-5].strip().title() + " Agar"
        return None

    for nm in candidate_media:
        pretty = canon_media(nm)
        if not pretty:
            continue
        canon = next((w for w in MEDIA_WHITELIST if w.lower() == pretty.lower()), pretty)
        if canon not in collected_media:
            collected_media.append(canon)

    if collected_media:
        _set_field_safe(out, "Media Grown On", "; ".join(collected_media))

    # 13) Colony morphology
    cm_value = normalize_cm_phrase(raw)
    if cm_value:
        _set_field_safe(out, "Colony Morphology", cm_value)

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Normalize to schema + haemolysis bridge + tidy media & morphology
# ──────────────────────────────────────────────────────────────────────────────
def normalize_to_schema(parsed: Dict[str, str], db_fields: List[str]) -> Dict[str, str]:
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)
    out: Dict[str, str] = {}
    strict = os.getenv("BACTAI_STRICT_MODE", "0") == "1"

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
        else:
            if not strict:
                pass

    # Bridge: Haemolysis Type → Haemolysis
    ht = alias.get("haemolysis type"); h = alias.get("haemolysis")
    if ht in out and h in fields:
        tval = out.get(ht, "")
        if tval in {"Alpha", "Beta"}: out[h] = "Positive"
        elif tval in {"Gamma", "None"}: out[h] = "Variable"

    # Clamp media & de-dupe
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

    # De-dupe colony morphology
    if "Colony Morphology" in out and out["Colony Morphology"]:
        chunks = [c.strip() for c in out["Colony Morphology"].split(";") if c.strip()]
        seen = set(); cleaned = []
        for c in chunks:
            if c not in seen:
                cleaned.append(c); seen.add(c)
        out["Colony Morphology"] = "; ".join(cleaned)

    return out

# ──────────────────────────────────────────────────────────────────────────────
# MAIN: Parse (LLM + regex enrichment) → normalize
# ──────────────────────────────────────────────────────────────────────────────
def parse_input_free_text(user_text: str, prior_facts: Dict | None = None, db_fields: List[str] | None = None) -> Dict:
    if not (user_text and user_text.strip()):
        return {}
    db_fields = db_fields or []
    cats = _summarize_field_categories(db_fields)

    # 1) Optional few-shot from feedback (last 5 failures)
    feedback_examples = _load_json(FEEDBACK_PATH, [])
    feedback_tail = feedback_examples[-5:] if feedback_examples else []
    feedback_context = ""
    for f in feedback_tail:
        name = f.get("name","case")
        txt = f.get("text","")
        errs = f.get("errors", [])
        feedback_context += f"\nExample failed: {name}\nInput: {txt}\nErrors: {errs}\n"

    # 2) LLM pass (Ollama-only). If fail → fallback_parser or regex-only path.
    llm_parsed = {}
    try:
        import ollama
        prompt = build_prompt_text(user_text + ("\n\nPast mistakes:\n" + feedback_context if feedback_context else ""), cats, prior_facts)
        model_name = os.getenv("LOCAL_MODEL", "deepseek-v3.1:671b")
        out = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        m = re.search(r"\{.*\}", out["message"]["content"], re.S)
        llm_parsed = json.loads(m.group(0)) if m else {}
    except Exception as e:
        # If installed fallback exists, use it; otherwise continue with regex only
        if fallback_parser is not None:
            try:
                llm_parsed = fallback_parser(user_text, prior_facts, db_fields)
            except Exception:
                llm_parsed = {}
        else:
            llm_parsed = {}

    # 3) Regex enrichment
    regex_ferm = extract_fermentations_regex(user_text, db_fields)
    regex_bio  = extract_biochem_regex(user_text, db_fields)

    # 4) Merge (regex wins on specificity conflicts)
    merged = {}
    if prior_facts: merged.update(prior_facts)
    merged.update(llm_parsed or {})
    merged.update(regex_ferm)
    merged.update(regex_bio)

    # 5) Normalize
    normalized = normalize_to_schema(merged, db_fields)
    return normalized

# ──────────────────────────────────────────────────────────────────────────────
# WHAT-IF helper
# ──────────────────────────────────────────────────────────────────────────────
def apply_what_if(user_text: str, prior_result: Dict[str, str], db_fields: List[str]) -> Dict[str, str]:
    if not (user_text and prior_result):
        return prior_result or {}

    alias = build_alias_map(db_fields)
    txt = normalize_text(user_text)
    patterns = [
        r"what\s+if\s+([a-z\s]+?)\s+(?:is|was|were|became|becomes|turned|changed\s+to)\s+([a-z\+\-]+)",
        r"suppose\s+([a-z\s]+?)\s+(?:is|was|were|became|becomes)\s+([a-z\+\-]+)",
        r"if\s+it\s+(?:is|was|were)\s+([a-z\s]+?)\s*(?:instead)?\s*(?:of|to)?\s*([a-z\+\-]+)?",
        r"change\s+([a-z\s]+?)\s+to\s+([a-z\+\-]+)"
    ]
    for pat in patterns:
        m = re.search(pat, txt)
        if m:
            field = (m.group(1) or "").strip()
            new_val = (m.group(2) or "").strip() if m.lastindex and m.lastindex >= 2 else ""
            target = alias.get(field.lower(), None) or field.title()
            new_val = _canon_value(target, new_val.title())
            if target in prior_result and new_val:
                new_dict = dict(prior_result)
                new_dict[target] = new_val
                return new_dict
    return prior_result

# ──────────────────────────────────────────────────────────────────────────────
# GOLD TESTS: load from gold_tests.json (if present), run, and learn
# ──────────────────────────────────────────────────────────────────────────────
def _diff_for_feedback(expected: Dict[str,str], got: Dict[str,str]) -> List[Dict[str,str]]:
    diffs = []
    for k, exp in expected.items():
        g = got.get(k, None)
        if g is None:
            diffs.append({"field": k, "expected": exp, "got": ""})
        elif str(g) != str(exp):
            diffs.append({"field": k, "expected": exp, "got": g})
    return diffs

def _log_feedback_case(name: str, text: str, diffs: List[Dict[str,str]]):
    if not diffs:
        return
    feedback = _load_json(FEEDBACK_PATH, [])
    feedback.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "text": text,
        "errors": diffs
    })
    _save_json(FEEDBACK_PATH, feedback)

def run_gold_tests(db_fields: Optional[List[str]] = None) -> Tuple[int,int]:
    print("Running Gold Tests...")
    tests = _load_json(GOLD_TESTS_PATH, [])
    if not tests:
        print("No gold_tests.json found or file empty. Skipping.")
        return (0, 0)

    if db_fields is None:
        # Minimal schema superset
        db_fields = [
            "Genus","Gram Stain","Shape","Catalase","Oxidase","Colony Morphology","Haemolysis","Haemolysis Type","Indole",
            "Growth Temperature","Media Grown On","Motility","Capsule","Spore Formation","Oxygen Requirement","Methyl Red","VP",
            "Citrate","Urease","H2S","Lactose Fermentation","Glucose Fermentation","Sucrose Fermentation","Nitrate Reduction",
            "Lysine Decarboxylase","Ornitihine Decarboxylase","Arginine dihydrolase","Gelatin Hydrolysis","Esculin Hydrolysis","Dnase",
            "ONPG","NaCl Tolerant (>=6%)","Lipase Test","Xylose Fermentation","Rhamnose Fermentation","Mannitol Fermentation",
            "Sorbitol Fermentation","Maltose Fermentation","Arabinose Fermentation","Raffinose Fermentation","Inositol Fermentation","Trehalose Fermentation","Coagulase"
        ]

    total, passed = 0, 0
    for case in tests:
        total += 1
        name = case.get("name", f"case_{total}")
        input_text = case.get("input", "")
        expected_raw = case.get("expected", {})

        # 🧩 PATCH: only keep expected fields that exist in the current DB schema
        expected = {k: v for k, v in expected_raw.items() if k in db_fields}

        got = parse_input_free_text(input_text, db_fields=db_fields)
        diffs = _diff_for_feedback(expected, got)
        ok = (not diffs)

        if ok:
            passed += 1
            print(f"✅ {name} passed.")
        else:
            print(f"❌ {name} failed.")
            for d in diffs[:10]:
                print(f"   - {d['field']}: expected '{d['expected']}' got '{d['got']}'")
            _log_feedback_case(name, input_text, diffs)

    print(f"Gold Tests: {passed}/{total} passed.")
    return (passed, total)
# ──────────────────────────────────────────────────────────────────────────────
# 🧠 Self-learning: analyze feedback → memory (3-strike heuristics)
# ──────────────────────────────────────────────────────────────────────────────
def analyze_feedback_and_learn(feedback_path=FEEDBACK_PATH, memory_path=MEMORY_PATH):
    feedback = _load_json(feedback_path, [])
    if not feedback:
        return

    memory = _load_json(memory_path, {})
    history = memory.get("history", [])
    field_counts = {}
    suggestions = []

    for case in feedback:
        for err in case.get("errors", []):
            field = err.get("field")
            got = (err.get("got") or "").lower()
            exp = (err.get("expected") or "").lower()
            if not field:
                continue
            field_counts[field] = field_counts.get(field, 0) + 1
            sim = difflib.SequenceMatcher(None, got, exp).ratio()
            if sim < 0.6:
                suggestions.append(f"Consider adjusting pattern for '{field}' — often parsed '{got}' instead of '{exp}'")

    auto_heuristics = {}
    for field, count in field_counts.items():
        if count >= 3:
            # Decide a naive polarity direction based on what "expected" often is
            # We don’t compute full histograms here; simple default:
            rule = f"Add stronger regex matching for '{field}' with negation/positive terms"
            auto_heuristics[field] = {"rule": rule, "count": count}

    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_error_fields": sorted(field_counts.items(), key=lambda x: -x[1])[:10],
        "suggestions": suggestions[:20]
    })
    memory["history"] = history
    memory["auto_heuristics"] = auto_heuristics
    _save_json(memory_path, memory)
    print(f"🧠 Learned hints for {len(auto_heuristics)} fields; updated memory.")

# ──────────────────────────────────────────────────────────────────────────────
# 🧬 Auto-update this file’s regex lists from learned heuristics
# ──────────────────────────────────────────────────────────────────────────────
def auto_update_parser_regex(memory_path=MEMORY_PATH, parser_file=__file__):
    """Automatically patch regex pattern lists when heuristics reach threshold (safe version)."""
    memory = _load_json(memory_path, {})
    auto_heuristics = memory.get("auto_heuristics", {})
    if not auto_heuristics:
        print("ℹ️ No new regex heuristics to apply.")
        return

    try:
        with open(parser_file, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"❌ Could not read parser file: {e}")
        return

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
        "fermentation": "FERMENTATION_PATTERNS",
    }

    updated = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for field, rule in auto_heuristics.items():
        field_lower = field.lower()
        pattern_list = None
        for key, list_name in pattern_lists.items():
            if key in field_lower:
                pattern_list = list_name
                break
        if not pattern_list:
            continue

        # Build a learned regex string safely (avoid rf"..." with escapes)
        if any(k in rule["rule"].lower() for k in ["negative", "not"]):
            learned_regex = r'"\\b' + re.escape(field_lower) + r'\\b.*(?:negative|not\\s+detected|not\\s+produced)"'
        elif "positive" in rule["rule"].lower():
            learned_regex = r'"\\b' + re.escape(field_lower) + r'\\b.*(?:positive|detected|produced)"'
        else:
            learned_regex = r'"\\b' + re.escape(field_lower) + r'\\b.*reaction"'

        insertion = f"    r{learned_regex},  # auto-learned {now} ({rule['count']}x)\n"

        # Safe regex replacement — avoids writing literal \1 \2 \3
        pattern_block_regex = f"({re.escape(pattern_list)}\\s*=\\s*\\[)([^\\]]*)(\\])"
        new_code, count = re.subn(
            pattern_block_regex,
            lambda m: m.group(1) + m.group(2) + insertion + m.group(3),
            code,
            flags=re.S,
        )

        if count > 0:
            code = new_code
            updated += 1
            print(f"✅ Added learned pattern for {field} → {pattern_list}")

    # Append summary and write file if updated
    if updated > 0:
        code += f"\n\n# === AUTO-LEARNED PATTERNS SUMMARY ({now}) ===\n"
        for f, r in auto_heuristics.items():
            code += f"# {f}: {r['rule']} (seen {r['count']}x)\n"

        try:
            with open(parser_file, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"🧠 Updated {os.path.basename(parser_file)} with {updated} new regex patterns.")
        except Exception as e:
            print(f"❌ Failed to write updates: {e}")
    else:
        print("ℹ️ No matching pattern lists found; no changes made.")

# ──────────────────────────────────────────────────────────────────────────────
# Convenience bootstrap: run learning + optional gold tests + auto-patch
# Call this once from your Streamlit app startup if you want automatic upkeep.
# ──────────────────────────────────────────────────────────────────────────────
def enable_self_learning_autopatch(run_tests: bool = False, db_fields: Optional[List[str]] = None):
    """
    Typical Streamlit usage in app.py/app_chat.py:
        from parser_llm import enable_self_learning_autopatch
        enable_self_learning_autopatch(run_tests=False)
    """
    if run_tests:
        run_gold_tests(db_fields=db_fields)
    analyze_feedback_and_learn()
    auto_update_parser_regex()

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--test" in sys.argv:
        run_gold_tests()
        analyze_feedback_and_learn()
        auto_update_parser_regex()
        sys.exit(0)
    print("parser_llm.py loaded. Use --test to run gold tests (learns & patches).")


# === AUTO-LEARNED PATTERNS SUMMARY (2025-11-10 21:17:08) ===
# DNase: Add stronger regex matching for 'DNase' with negation/positive terms (seen 4x)
# Lactose Fermentation: Add stronger regex matching for 'Lactose Fermentation' with negation/positive terms (seen 71x)
# NaCl Tolerant (>=6%): Add stronger regex matching for 'NaCl Tolerant (>=6%)' with negation/positive terms (seen 20x)
# Motility: Add stronger regex matching for 'Motility' with negation/positive terms (seen 11x)
# Spore Formation: Add stronger regex matching for 'Spore Formation' with negation/positive terms (seen 38x)
# Media Grown On: Add stronger regex matching for 'Media Grown On' with negation/positive terms (seen 167x)
# Xylose Fermentation: Add stronger regex matching for 'Xylose Fermentation' with negation/positive terms (seen 16x)
# Mannitol Fermentation: Add stronger regex matching for 'Mannitol Fermentation' with negation/positive terms (seen 32x)
# Gelatin Hydrolysis: Add stronger regex matching for 'Gelatin Hydrolysis' with negation/positive terms (seen 15x)
# Growth Temperature: Add stronger regex matching for 'Growth Temperature' with negation/positive terms (seen 110x)
# Haemolysis Type: Add stronger regex matching for 'Haemolysis Type' with negation/positive terms (seen 4x)
# Haemolysis: Add stronger regex matching for 'Haemolysis' with negation/positive terms (seen 34x)
# Esculin Hydrolysis: Add stronger regex matching for 'Esculin Hydrolysis' with negation/positive terms (seen 4x)
# Colony Morphology: Add stronger regex matching for 'Colony Morphology' with negation/positive terms (seen 65x)
# Oxygen Requirement: Add stronger regex matching for 'Oxygen Requirement' with negation/positive terms (seen 32x)
# Sucrose Fermentation: Add stronger regex matching for 'Sucrose Fermentation' with negation/positive terms (seen 39x)
# H2S: Add stronger regex matching for 'H2S' with negation/positive terms (seen 5x)
# Sorbitol Fermentation: Add stronger regex matching for 'Sorbitol Fermentation' with negation/positive terms (seen 10x)
# Capsule: Add stronger regex matching for 'Capsule' with negation/positive terms (seen 4x)
# Arabinose Fermentation: Add stronger regex matching for 'Arabinose Fermentation' with negation/positive terms (seen 4x)
# Shape: Add stronger regex matching for 'Shape' with negation/positive terms (seen 33x)
# Glucose Fermentation: Add stronger regex matching for 'Glucose Fermentation' with negation/positive terms (seen 83x)
# Odour: Add stronger regex matching for 'Odour' with negation/positive terms (seen 6x)
# Growth Factors: Add stronger regex matching for 'Growth Factors' with negation/positive terms (seen 3x)
# Fructose Fermentation: Add stronger regex matching for 'Fructose Fermentation' with negation/positive terms (seen 3x)
# Glucose Oxidation: Add stronger regex matching for 'Glucose Oxidation' with negation/positive terms (seen 5x)
# Maltose Fermentation: Add stronger regex matching for 'Maltose Fermentation' with negation/positive terms (seen 5x)
# Gram Stain: Add stronger regex matching for 'Gram Stain' with negation/positive terms (seen 3x)
# Ornitihine Decarboxylase: Add stronger regex matching for 'Ornitihine Decarboxylase' with negation/positive terms (seen 4x)
# Rhamnose Fermentation: Add stronger regex matching for 'Rhamnose Fermentation' with negation/positive terms (seen 5x)
