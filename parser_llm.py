# parser_llm.py — v5 (Ollama + Regex + Self-Learning + Safe Autopatching)
# ──────────────────────────────────────────────────────────────────────────────
# What you get:
#   • Ollama LLM parsing (primary) with JSON extraction
#   • Deterministic regex enrichment for morphology/biochem/fermentations/media
#   • Persistent self-learning from gold_tests + live runs (3-strike rule idea)
#   • Auto-injection of learned regex lines back into this file (safe patcher)
#   • Sanitizers that fix bad escapes / unterminated raw strings BEFORE parsing
#   • Git auto-commit/push (Option B): triggered at the END of auto_update_parser_regex()
#   • CLI: python parser_llm.py --test  → runs gold_tests.json, learns, patches
#
# Files used:
#   • gold_tests.json          ← optional input for gold-testing
#   • parser_feedback.json     ← logs mismatches from tests/runs
#   • parser_memory.json       ← learned summaries & auto_heuristics
#
# Env (use Streamlit secrets or real env):
#   • OLLAMA_API_KEY           ← your cloud API key (if using Ollama Cloud)
#   • LOCAL_MODEL              ← default "deepseek-v3.1:671b"
#   • BACTAI_STRICT_MODE       ← "1" for strict schema-only output
#   • ENABLE_AUTO_COMMIT       ← "true" to enable auto-commit/push after patch
#   • GH_TOKEN                 ← GitHub token with repo scope
#   • GITHUB_REPO              ← e.g. "EphraimAsad/BacAi-D-Dev"
#   • GIT_BRANCH               ← default "main"
#   • GIT_USER_NAME            ← e.g. "EphraimAsad"
#   • GIT_USER_EMAIL           ← e.g. "zainasad98@gmail.com"
#   • PARSER_FILE              ← default "parser_llm.py"
#   • MEMORY_PATH              ← default "parser_memory.json"
#   • FEEDBACK_PATH            ← default "parser_feedback.json"
#
# Public API:
#   parse_input_free_text(text, prior_facts=None, db_fields=None) -> Dict[str,str]
#   apply_what_if(user_text, prior_result, db_fields) -> Dict[str,str]
#   run_gold_tests()  # CLI with --test
#   analyze_feedback_and_learn()  # called automatically by runner helpers
#   auto_update_parser_regex()    # writes learned regex into *_PATTERNS lists
#   enable_self_learning_autopatch(run_tests: bool = False)
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import json
import sys
import math
import difflib
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Load Streamlit secrets into env (if present), without overwriting existing
# ──────────────────────────────────────────────────────────────────────────────
def _load_streamlit_secrets_into_env():
    try:
        import streamlit as st
        for k, v in st.secrets.items():
            if k not in os.environ:
                os.environ[k] = str(v)
    except Exception:
        # Not in Streamlit or no secrets — fine.
        pass

_load_streamlit_secrets_into_env()

# Optional fallback parser import (kept harmless if missing)
fallback_parser = None
try:
    from parser_basic import parse_input_free_text as _basic_parser
    fallback_parser = _basic_parser
except Exception:
    fallback_parser = None

# ──────────────────────────────────────────────────────────────────────────────
# Self-healing sanitizers (SAFE: not auto-run at import; call from a function)
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize_parser_file(file_path: str) -> None:
    """
    Ensure lines that look like raw-regex items end with a closing quote + comma,
    and remove trailing junk that could produce unterminated strings.
    Only lightly touches lines that begin with r"..."/r'...' patterns.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        fixed: List[str] = []
        for line in lines:
            s = line.rstrip("\n")

            # Fix invisible control chars
            s = s.replace("\u2028", "").replace("\u2029", "")

            stripped = s.strip()
            if stripped.startswith('r"') or stripped.startswith("r'"):
                # if ends with quote but missing comma, add comma
                if stripped.endswith('"') and not stripped.endswith('",'):
                    s = s + ","
                if stripped.endswith("'") and not stripped.endswith("',"):
                    s = s + ","

            fixed.append(s + "\n")

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(fixed)
        print(f"🧩 Sanitized parser file endings: {file_path}")
    except Exception as e:
        print(f"⚠️ Failed _sanitize_parser_file: {e}")


def _sanitize_auto_learned_patterns(file_path: str) -> None:
    """
    Fix common bad-escape issues that can sneak into learned regex in this file.
    Safe, idempotent, and conservative.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original = content

        # Normalize accidental backslash+space to \s+ where it occurs literally.
        content = content.replace(r'\ ', r'\\s+')

        # Remove accidental NULs if any crept in
        content = content.replace("\x00", "")

        if content != original:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"🧩 Fixed escaped regexes & control chars in {file_path}")
        else:
            print("ℹ️ No escaped regex fixes applied.")
    except Exception as e:
        print(f"⚠️ Auto-learned pattern sanitization failed: {e}")


def _repair_parser_file(file_path: str) -> None:
    """
    Defensive pass to avoid unterminated string literals:
    - Ensures list items that clearly should end with '",' do so (best-effort).
    - Ensures brackets [ ... ] stay balanced around pattern lists.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original = content
        lines = content.splitlines()
        out_lines: List[str] = []

        inside_list = False
        open_brackets = 0

        for raw in lines:
            line = raw

            if re.search(r"_PATTERNS\s*=\s*\[", line):
                inside_list = True
                open_brackets = 1

            if inside_list:
                stripped = line.strip()
                if (stripped.startswith('r"') or stripped.startswith("r'")) and not stripped.endswith(","):
                    if stripped.endswith('"') or stripped.endswith("'"):
                        line = line + ","

            # Track bracket balance
            open_brackets += line.count("[") - line.count("]")

            if inside_list and open_brackets <= 0:
                inside_list = False

            out_lines.append(line)

        repaired = "\n".join(out_lines)

        if repaired != original:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(repaired)
            print(f"🧩 Repaired potential unterminated lines in {file_path}")
        else:
            print("ℹ️ No repair changes required.")
    except Exception as e:
        print(f"⚠️ Parser file repair failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Persistent storage paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.getcwd(), "data")
GOLD_TESTS_PATH = os.path.join(os.getcwd(), "gold_tests.json")
FEEDBACK_PATH = os.path.join(os.getcwd(), "parser_feedback.json")
MEMORY_PATH = os.path.join(os.getcwd(), "parser_memory.json")

def _ensure_data_dir() -> None:
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

def _save_json(path: str, obj) -> None:
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
    "Oxygen Requirement": {
        "Intracellular", "Aerobic", "Anaerobic",
        "Facultative Anaerobe", "Microaerophilic", "Capnophilic"
    },
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
# (Learning writes new raw regex rows directly into these lists)
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
]
DNASE_PATTERNS = [
    r"\bdnase\s*(?:test)?\s*(?:\+|positive)\b",
    r"\bdnase\s*(?:test)?\s*(?:\-|negative)\b",
]
GELATIN_PATTERNS = [
    r"\bgelatin\s+(?:liquefaction|hydrolysis)\s*(?:\+|positive)\b",
    r"\bgelatin\s+(?:liquefaction|hydrolysis)\s*(?:\-|negative)\b",
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
    r"\barginine\s+dihydrolase\s+(?:\+|positive|detected)\b",
    r"\barginine\s+dihydrolase\s+(?:\-|negative|not\s+detected)\b",
    r"\b(ldc|odc|adh)\s*(?:\+|positive|-|negative)\b",
]
FERMENTATION_PATTERNS = [
    r"\blactose\s+fermentation\b.*(?:positive|detected|produced)",
    r"\bxylose\s+fermentation\b.*(?:positive|detected|produced)",
    r"\bmannitol\s+fermentation\b.*(?:positive|detected|produced)",
    r"\bsucrose\s+fermentation\b.*(?:positive|detected|produced)",
    r"\bsorbitol\s+fermentation\b.*(?:positive|detected|produced)",
    r"\barabinose\s+fermentation\b.*(?:positive|detected|produced)",
    r"\bglucose\s+fermentation\b.*(?:positive|detected|produced)",
    r"\bmaltose\s+fermentation\b.*(?:positive|detected|produced)",
    r"\brhamnose\s+fermentation\b.*(?:positive|detected|produced)",
    r"(?:ferments?|utilizes?)[^.]*?\bbut\s+not\s+([\w\s,;.&-]+)",
    r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b",
]

# Helpers: columns, normalization, aliases, tokens
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
    add("onpg","ONPG"); add("esculin hydrolysis","Esculin Hydrolysis")
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

# Colony morphology normalization
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
    t = (text or "").lower()
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

# LLM prompt
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

# Apply learned patterns
def _apply_learned_patterns(field_name: str, patterns: List[str], text: str, tokens: List[str], out: Dict[str,str], alias: Dict[str,str]):
    """
    For each pattern, decide Positive/Negative/Variable by presence of cue words.
    """
    key = alias.get(field_name.lower(), field_name)
    if not key:
        key = field_name

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
            continue

# Fermentation extraction (+ shorthands + negatives and "but not" lists)
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

    # Learned generic patterns (from FERMENTATION_PATTERNS)
    for pat in FERMENTATION_PATTERNS:
        try:
            for m in re.finditer(pat, t, flags=re.I|re.S):
                if m.lastindex and m.lastindex >= 1:
                    span = m.group(1)
                    if span:
                        span = re.split(r"(?i)\bbut\s+not\b", span)[0]
                        for a in _tokenize_list(span):
                            set_field_by_base(a, "Positive")
        except re.error:
            continue

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

    # Negatives after "but not …"
    for m in re.finditer(r"(?:ferments?|utilizes?)[^.]*?\bbut\s+not\s+([\w\s,;.&-]+)", t, flags=re.I):
        seg = m.group(1)
        seg = re.sub(r"\bor\b", ",", seg, flags=re.I)
        seg = re.sub(r"\bnor\b", ",", seg, flags=re.I)
        for a in _tokenize_list(seg):
            set_field_by_base(a, "Negative")

    # Shorthand: "lactose +" / "rhamnose -"
    for m in re.finditer(r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b", t, flags=re.I):
        a, sign = m.group(1), m.group(2)
        set_field_by_base(a, "Positive" if sign == "+" else "Negative")

    # Variable adjectives per-sugar
    for base, field_name in base_to_field.items():
        if re.search(rf"\b{re.escape(base)}\b\s+(?:variable|inconsistent|weak|trace|slight|irregular)", t, flags=re.I):
            _set_field_safe(out, field_name, "Variable")

    # LF/NLF short forms
    if re.search(r"\bnlf\b", t):
        set_field_by_base("lactose", "Negative")
    if re.search(r"\blf\b", t) and "Lactose Fermentation" not in out:
        set_field_by_base("lactose", "Positive")

    return out

# Biochemical / morphology / oxygen / media extraction
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

    # Apply learned pattern lists up front
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

    # Gram
    if re.search(r"\bgram[-\s]?positive\b", t) and not re.search(r"\bgram[-\s]?negative\b", t):
        set_field("gram stain", "Positive")
    elif re.search(r"\bgram[-\s]?negative\b", t) and not re.search(r"\bgram[-\s]?positive\b", t):
        set_field("gram stain", "Negative")

    # Shape
    if re.search(r"\bcocci\b", t): set_field("shape", "Cocci")
    if re.search(r"\brods?\b|bacilli\b", t): set_field("shape", "Rods")
    if re.search(r"\bspiral\b", t): set_field("shape", "Spiral")
    if re.search(r"\bshort\s+rods\b", t): set_field("shape", "Short Rods")

    # Motility
    if re.search(r"\bnon[-\s]?motile\b", t): set_field("motility", "Negative")
    elif re.search(r"\bmotile\b", t): set_field("motility", "Positive")

    # Capsule
    if re.search(r"\b(capsulated|encapsulated)\b", t):
        set_field("capsule", "Positive")
    if re.search(r"\bnon[-\s]?capsulated\b|\bcapsule\s+absent\b", t):
        set_field("capsule", "Negative")
    if re.search(r"\bcapsule\s+(?:variable|inconsistent|weak)\b", t):
        set_field("capsule", "Variable")

    # Spore formation
    if re.search(r"\bnon[-\s]?spore[-\s]?forming\b|\bno\s+spores?\b", t):
        set_field("spore formation", "Negative")
    if re.search(r"\bspore[-\s]?forming\b|\bspores?\s+present\b", t):
        set_field("spore formation", "Positive")

    # Oxygen requirement
    if re.search(r"\bintracellular\b", t): set_field("oxygen requirement", "Intracellular")
    elif re.search(r"\bcapnophil(ic|e)\b", t): set_field("oxygen requirement", "Capnophilic")
    elif re.search(r"\bmicroaerophil(ic|e)\b", t): set_field("oxygen requirement", "Microaerophilic")
    elif re.search(r"\bfacultative\b", t) or re.search(r"\bfacultative\s+anaerob", t):
        set_field("oxygen requirement", "Facultative Anaerobe")
    elif re.search(r"\baerobic\b", t): set_field("oxygen requirement", "Aerobic")
    elif re.search(r"\banaerobic\b", t): set_field("oxygen requirement", "Anaerobic")

    # Generic enzyme tests (backup)
    generic_tests = ["catalase","oxidase","coagulase","urease","lipase","indole","citrate","vp","methyl red","gelatin","dnase","nitrate reduction","nitrate","h2s","esculin hydrolysis","onpg"]
    for test in generic_tests:
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:\+|positive|detected|produced)\b", t):
            set_field(test, "Positive")
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:\-|negative|not\s+detected|not\s+produced|absent)\b", t):
            set_field(test, "Negative")
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:variable|weak|trace|slight)\b", t):
            set_field(test, "Variable")

    # H2S special
    if re.search(r"\bh\s*2\s*s\s+not\s+produced\b", t): set_field("h2s", "Negative")
    if re.search(r"\bproduces\s+h\s*2\s*s\b", t): set_field("h2s", "Positive")

    # Nitrate alt phrasing
    if re.search(r"\breduces\s+nitrate\b", t): set_field("nitrate reduction", "Positive")
    if re.search(r"\bdoes\s+not\s+reduce\s+nitrate\b", t): set_field("nitrate reduction", "Negative")

    # Haemolysis Type
    if re.search(r"\b(beta|β)[-\s]?haem", t):
        set_field("haemolysis type", "Beta")
    elif re.search(r"\b(alpha|α)[-\s]?haem", t):
        set_field("haemolysis type", "Alpha")
    elif re.search(r"\b(gamma|γ)[-\s]?haem\b", t) or re.search(r"\bno\s+haemolysis\b|\bhaemolysis\s+not\s+observed\b", t):
        set_field("haemolysis type", "Gamma")

    # Growth temperature
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

    # NaCl tolerant
    if re.search(r"\b(tolerant|grows|growth)\s+(?:in|up\s+to|to|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t): set_field("nacl tolerant (>=6%)", "Positive")
    if re.search(r"\bno\s+growth\s+(?:in|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t): set_field("nacl tolerant (>=6%)", "Negative")
    if re.search(r"\bnacl\s+tolerant\b", t): set_field("nacl tolerant (>=6%)", "Positive")

    # Media detection (exclude TSI)
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
        if name in {"macconkey"}: return "MacConKey Agar" if False else "MacConKey Agar".replace("Key","key")  # maintain title-case; result "MacConkey Agar"
        if name in {"blood","ba","ssa"}: return "Blood Agar"
        if name == "nutrient": return "Nutrient Agar"
        if name == "tsa": return "Tryptic Soy Agar"
        if name == "bhi": return "Brain Heart Infusion Agar"
        if name == "cba": return "Columbia Blood Agar"
        if name.endswith(" agar"):
            return name[:-5].strip().title() + " Agar"
        return None

    # fix "MacConKey Agar" correction above to actual canonical
    def _fix_mac(name: str) -> str:
        return "MacConkey Agar" if name.lower().startswith("maccon") else name

    for nm in candidate_media:
        pretty = canon_media(nm)
        if not pretty:
            continue
        pretty = _fix_mac(pretty)
        canon = next((w for w in MEDIA_WHITELIST if w.lower() == pretty.lower()), pretty)
        if canon not in collected_media:
            collected_media.append(canon)

    if collected_media:
        _set_field_safe(out, "Media Grown On", "; ".join(collected_media))

    # Colony morphology
    cm_value = normalize_cm_phrase(raw)
    if cm_value:
        _set_field_safe(out, "Colony Morphology", cm_value)

    return out

# Normalize to schema + haemolysis bridge + tidy media & morphology
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

# MAIN: Parse (LLM + regex enrichment) → normalize
def parse_input_free_text(
    user_text: str,
    prior_facts: Optional[Dict] = None,
    db_fields: Optional[List[str]] = None
) -> Dict:
    if not (user_text and str(user_text).strip()):
        return {}
    db_fields = db_fields or []
    cats = _summarize_field_categories(db_fields)

    # Optional few-shot from feedback (last 5 failures)
    feedback_examples = _load_json(FEEDBACK_PATH, [])
    feedback_tail = feedback_examples[-5:] if feedback_examples else []
    feedback_context = ""
    for f in feedback_tail:
        name = f.get("name","case")
        txt = f.get("text","")
        errs = f.get("errors", [])
        feedback_context += f"\nExample failed: {name}\nInput: {txt}\nErrors: {errs}\n"

    # LLM pass (Ollama). If fail → fallback_parser or regex-only path.
    llm_parsed: Dict[str, str] = {}
    try:
        import ollama
        prompt = build_prompt_text(
            user_text + ("\n\nPast mistakes:\n" + feedback_context if feedback_context else ""),
            cats,
            prior_facts
        )
        model_name = os.getenv("LOCAL_MODEL", "deepseek-v3.1:671b")
        out = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        m = re.search(r"\{.*\}", out.get("message", {}).get("content", ""), re.S)
        llm_parsed = json.loads(m.group(0)) if m else {}
    except Exception:
        if fallback_parser is not None:
            try:
                llm_parsed = fallback_parser(user_text, prior_facts, db_fields)
            except Exception:
                llm_parsed = {}
        else:
            llm_parsed = {}

    # Regex enrichment
    regex_ferm = extract_fermentations_regex(user_text, db_fields)
    regex_bio  = extract_biochem_regex(user_text, db_fields)

    # Merge (regex wins)
    merged: Dict[str, str] = {}
    if prior_facts:
        merged.update(prior_facts)
    merged.update(llm_parsed or {})
    merged.update(regex_ferm)
    merged.update(regex_bio)

    # Normalize
    normalized = normalize_to_schema(merged, db_fields)
    return normalized

# ===================================================================
# 🔁 “What-if” incremental re-parse
# ===================================================================
def apply_what_if(user_text: str, prior_result: Dict[str,str], db_fields: List[str]) -> Dict[str,str]:
    """
    Re-parse a single new description while preserving known facts.
    """
    fresh = parse_input_free_text(user_text, prior_result, db_fields)
    merged = {**prior_result, **fresh}
    return normalize_to_schema(merged, db_fields)


# ===================================================================
# 🧪 Gold-standard test harness + feedback memory
# ===================================================================
def _diff_for_feedback(expected: Dict[str,str], got: Dict[str,str]) -> List[Dict[str,str]]:
    diffs = []
    for k, exp in expected.items():
        g = got.get(k)
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
    """
    Execute gold_tests.json and record mismatches to feedback.
    """
    print("🧪 Running gold tests…")
    tests = _load_json(GOLD_TESTS_PATH, [])
    if not tests:
        print("ℹ️ No gold_tests.json found or file empty.")
        return (0, 0)

    if db_fields is None:
        db_fields = list(ALLOWED_VALUES.keys())

    total, passed = 0, 0
    for case in tests:
        total += 1
        name = case.get("name", f"case_{total}")
        input_text = case.get("input", "")
        expected_raw = case.get("expected", {})

        expected = {k: v for k, v in expected_raw.items() if k in db_fields}
        got = parse_input_free_text(input_text, db_fields=db_fields)
        diffs = _diff_for_feedback(expected, got)
        ok = not diffs

        if ok:
            passed += 1
            print(f"✅ {name} passed")
        else:
            print(f"❌ {name} failed:")
            for d in diffs[:10]:
                print(f"   - {d['field']}: expected '{d['expected']}' got '{d['got']}'")
            _log_feedback_case(name, input_text, diffs)

    print(f"Gold tests: {passed}/{total} passed.")
    return (passed, total)


# ===================================================================
# 🧠 Feedback learning memory
# ===================================================================
def analyze_feedback_and_learn(feedback_path=FEEDBACK_PATH, memory_path=MEMORY_PATH):
    feedback = _load_json(feedback_path, [])
    if not feedback:
        return

    memory = _load_json(memory_path, {})
    history = memory.get("history", [])
    field_counts: Dict[str,int] = {}
    suggestions: List[str] = []

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
                suggestions.append(f"Consider adjusting pattern for '{field}' — parsed '{got}' instead of '{exp}'")

    auto_heuristics: Dict[str,Dict] = {}
    for field, count in field_counts.items():
        if count >= 3:
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
    print(f"🧠 Learned hints for {len(auto_heuristics)} fields; memory updated.")


# ===================================================================
# 🔧 Helper utilities for auto-learning regex patcher
# ===================================================================
def _escape_for_raw_regex(text: str) -> str:
    """
    Convert a human term (with spaces) into safe raw-regex token.
    """
    text = text.replace(r"\ ", r"\\s+")
    text = re.sub(r"\s+", r"\\s+", text.strip())
    return text


def _dedupe_keep_order(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _pattern_exists(existing_list: List[str], candidate: str) -> bool:
    return candidate in existing_list


# ===================================================================
# 🚀 Auto-commit to GitHub (Option B – improved diagnostics)
# ===================================================================
def auto_commit_changes(repo_path=".", commit_msg="Auto-update regex patterns"):
    """
    Commits parser_llm.py + memory files back to the remote repo if enabled.
    """
    import subprocess, shlex

    enabled = os.getenv("ENABLE_AUTO_COMMIT", "false").lower() == "true"
    token   = os.getenv("GH_TOKEN")
    repo    = os.getenv("GITHUB_REPO", "")
    branch  = os.getenv("GIT_BRANCH", "main")
    uname   = os.getenv("GIT_USER_NAME", "auto-learner")
    email   = os.getenv("GIT_USER_EMAIL", "auto@local")

    if not enabled:
        print("ℹ️ Auto-commit disabled by config.")
        return False
    if not (token and repo):
        print("⚠️ Missing GH_TOKEN or GITHUB_REPO — cannot commit.")
        return False

    print(f"🔧 Preparing auto-commit for {repo} on branch '{branch}'…")

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        subprocess.run(["git", "config", "--global", "user.name", uname], check=True, env=env)
        subprocess.run(["git", "config", "--global", "user.email", email], check=True, env=env)
        subprocess.run(["git", "add", "parser_llm.py", "parser_memory.json", "parser_feedback.json"], check=True, env=env)
        status = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, env=env)
        print(status.stdout)

        if not status.stdout.strip():
            print("ℹ️ No changes to commit.")
            return True

        subprocess.run(["git", "commit", "-m", commit_msg], check=True, env=env)

        remote_url = f"https://{token}@github.com/{repo}.git"
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True, env=env)
        subprocess.run(["git", "push", "origin", branch], check=True, env=env)
        print("✅ Changes committed & pushed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Auto-commit error: {e}")
        return False


# ===================================================================
# 🧩 Full-learning autopatcher for regex patterns
# ===================================================================
def auto_update_parser_regex(memory_path=MEMORY_PATH, parser_file=__file__):
    """
    Applies learned regex heuristics into *_PATTERNS lists.
    Deduplicates, sanitizes, commits to Git if enabled.
    """
    memory = _load_json(memory_path, {})
    auto_heuristics = memory.get("auto_heuristics", {})
    if not auto_heuristics:
        print("ℹ️ No new heuristics to apply.")
        return

    pattern_lists_map = {
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
        "nitrate reduction": "NITRATE_PATTERNS",
        "lysine decarboxylase": "DECARBOXYLASE_PATTERNS",
        "ornithine decarboxylase": "DECARBOXYLASE_PATTERNS",
        "ornitihine decarboxylase": "DECARBOXYLASE_PATTERNS",
        "arginine dihydrolase": "DECARBOXYLASE_PATTERNS",
        "lactose fermentation": "FERMENTATION_PATTERNS",
        "glucose fermentation": "FERMENTATION_PATTERNS",
        "sucrose fermentation": "FERMENTATION_PATTERNS",
        "xylose fermentation": "FERMENTATION_PATTERNS",
        "rhamnose fermentation": "FERMENTATION_PATTERNS",
        "mannitol fermentation": "FERMENTATION_PATTERNS",
        "sorbitol fermentation": "FERMENTATION_PATTERNS",
        "maltose fermentation": "FERMENTATION_PATTERNS",
        "arabinose fermentation": "FERMENTATION_PATTERNS",
        "raffinose fermentation": "FERMENTATION_PATTERNS",
        "inositol fermentation": "FERMENTATION_PATTERNS",
        "trehalose fermentation": "FERMENTATION_PATTERNS",
        "onpg": "DNASE_PATTERNS",
    }

    try:
        with open(parser_file, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"❌ Could not read parser file: {e}")
        return

    ns: Dict[str, object] = {}
    try:
        exec(compile(code, parser_file, "exec"), ns, ns)
    except Exception as e:
        print(f"⚠️ Parser introspection failed: {e}. Attempting repair…")
        _repair_parser_file(parser_file)
        _sanitize_parser_file(parser_file)
        _sanitize_auto_learned_patterns(parser_file)
        try:
            with open(parser_file, "r", encoding="utf-8") as f:
                code = f.read()
            exec(compile(code, parser_file, "exec"), ns, ns)
        except Exception as e2:
            print(f"❌ Still failed to exec after repair: {e2}")
            return

    updated = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_rule_to_list(list_name: str, raw_regex_literal: str) -> Tuple[str,bool]:
        nonlocal code
        pattern_block = rf"({re.escape(list_name)}\s*=\s*\[)([^]]*)(\])"
        m = re.search(pattern_block, code, flags=re.S)
        if not m:
            return (code, False)
        head, body, tail = m.group(1), m.group(2), m.group(3)
        existing = re.findall(r'r"[^"]*"|r\'[^\']*\'', body)
        if raw_regex_literal in existing:
            return (code, False)
        insertion = f"\n    {raw_regex_literal},  # auto-learned {now}"
        new_body = body + insertion
        new_code = code[:m.start()] + head + new_body + tail + code[m.end():]
        return (new_code, True)

    for field, rule in auto_heuristics.items():
        field_l = field.lower().strip()
        list_name = None
        if field_l in pattern_lists_map:
            list_name = pattern_lists_map[field_l]
        else:
            for k, v in pattern_lists_map.items():
                if k in field_l:
                    list_name = v
                    break
        if not list_name:
            continue

        safe_field = _escape_for_raw_regex(field_l)
        pos_pat = f'r"\\b{safe_field}\\b.*(?:positive|\\+|detected|produced)"'
        neg_pat = f'r"\\b{safe_field}\\b.*(?:negative|\\-|not\\s+detected|absent|not\\s+produced)"'
        code, ins1 = add_rule_to_list(list_name, pos_pat)
        code, ins2 = add_rule_to_list(list_name, neg_pat)
        if ins1 or ins2:
            updated += 1
            print(f"✅ Learned pattern(s) for '{field}' → {list_name}")

    if updated > 0:
        code += f"\n\n# === AUTO-LEARNED SUMMARY ({now}) ===\n"
        for f_name, rinfo in auto_heuristics.items():
            code += f"# {f_name}: {rinfo['rule']} (seen {rinfo['count']}×)\n"
        try:
            with open(parser_file, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"🧠 Updated {os.path.basename(parser_file)} with {updated} new regex patterns.")
            _sanitize_parser_file(parser_file)
            _sanitize_auto_learned_patterns(parser_file)
            _repair_parser_file(parser_file)
            auto_commit_changes(".", f"Auto-update regex ({updated} new patterns)")
        except Exception as e:
            print(f"❌ Failed to write updates: {e}")
    else:
        print("ℹ️ No unique patterns to add.")
# ===================================================================
# 🚦 Bootstrap: enable self-learning + optional gold tests
# ===================================================================
def enable_self_learning_autopatch(run_tests: bool = False, db_fields: Optional[List[str]] = None):
    """
    Typical Streamlit usage (early in app startup):
        from parser_llm import enable_self_learning_autopatch
        enable_self_learning_autopatch(run_tests=False)

    Steps:
      1) (optional) run_gold_tests()
      2) analyze_feedback_and_learn()
      3) auto_update_parser_regex()  → writes back into this file safely
      4) (optional) auto_commit_changes() is called inside auto_update when changes exist
    """
    # Defensive: a quick sanitize/repair pass before doing anything
    this_file = __file__
    _sanitize_parser_file(this_file)
    _sanitize_auto_learned_patterns(this_file)
    _repair_parser_file(this_file)

    if run_tests:
        try:
            run_gold_tests(db_fields=db_fields)
        except Exception as e:
            print(f"⚠️ run_gold_tests failed but continuing: {e}")

    try:
        analyze_feedback_and_learn()
    except Exception as e:
        print(f"⚠️ analyze_feedback_and_learn failed but continuing: {e}")

    try:
        auto_update_parser_regex()
    except Exception as e:
        print(f"⚠️ auto_update_parser_regex failed: {e}")


# ===================================================================
# 🧰 Optional: immediate auto-commit on import if toggled
#   (Useful when this module is executed in a one-off worker)
# ===================================================================
if os.getenv("ENABLE_AUTO_COMMIT_ON_IMPORT", "false").lower() == "true":
    try:
        auto_commit_changes(".", "Auto-commit on import")
    except Exception as e:
        print(f"⚠️ Auto-commit-on-import failed: {e}")


# ===================================================================
# 🖥️ CLI entrypoint
#   Examples:
#     python parser_llm.py --test
#     python parser_llm.py --learn
#     python parser_llm.py --patch
#     python parser_llm.py --commit
#     python parser_llm.py --all
# ===================================================================
if __name__ == "__main__":
    # Defensive cleanups every time we run as a script
    this_file = __file__
    _sanitize_parser_file(this_file)
    _sanitize_auto_learned_patterns(this_file)
    _repair_parser_file(this_file)

    args = set(a.lower() for a in sys.argv[1:])

    did_any = False

    if "--test" in args:
        did_any = True
        try:
            run_gold_tests()
        except Exception as e:
            print(f"❌ Gold tests failed: {e}")

    if "--learn" in args:
        did_any = True
        try:
            analyze_feedback_and_learn()
        except Exception as e:
            print(f"❌ analyze_feedback_and_learn failed: {e}")

    if "--patch" in args:
        did_any = True
        try:
            auto_update_parser_regex()
        except Exception as e:
            print(f"❌ auto_update_parser_regex failed: {e}")

    if "--commit" in args:
        did_any = True
        try:
            auto_commit_changes(".", "Manual commit via CLI")
        except Exception as e:
            print(f"❌ auto_commit_changes failed: {e}")

    if "--all" in args:
        did_any = True
        try:
            run_gold_tests()
        except Exception as e:
            print(f"❌ Gold tests failed: {e}")
        try:
            analyze_feedback_and_learn()
        except Exception as e:
            print(f"❌ analyze_feedback_and_learn failed: {e}")
        try:
            auto_update_parser_regex()
        except Exception as e:
            print(f"❌ auto_update_parser_regex failed: {e}")

    if not did_any:
        print(
            "parser_llm.py loaded.\n"
            "Use one of:\n"
            "  --test    Run gold tests and log feedback\n"
            "  --learn   Analyze feedback → memory\n"
            "  --patch   Apply learned regex updates to *_PATTERNS\n"
            "  --commit  Force a commit/push (if ENABLE_AUTO_COMMIT=true)\n"
            "  --all     test → learn → patch\n"
        )


