# parser_basic.py — v4 (Full Regex Parser + Self-Learning + Auto-Patching + Gold Tests)
# ──────────────────────────────────────────────────────────────────────────────
# Features:
#   • Pure regex parser (no LLM dependency)
#   • Covers morphology, biochem, fermentations, media, and oxygen
#   • Self-learning from feedback (3-strike heuristic)
#   • Auto-patches new regex patterns into this file permanently
#   • Gold Test framework compatible with parser_llm
#
# Files used:
#   • gold_tests.json
#   • parser_feedback.json
#   • parser_memory.json
#
# Env:
#   • BACTAI_STRICT_MODE = "1" for strict schema output
#
# Public API:
#   parse_input_free_text(text, prior_facts=None, db_fields=None)
#   enable_self_learning_autopatch(run_tests=False)
#
# CLI:
#   python parser_basic.py --test
# ──────────────────────────────────────────────────────────────────────────────

import os
import re
import sys
import json
import difflib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

# ──────────────────────────────────────────────────────────────────────────────
# Storage paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.getcwd(), "data")
GOLD_TESTS_PATH = os.path.join(os.getcwd(), "gold_tests.json")
FEEDBACK_PATH = os.path.join(os.getcwd(), "parser_feedback.json")
MEMORY_PATH = os.path.join(os.getcwd(), "parser_memory.json")

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: str, data):
    try:
        _ensure_data_dir()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Schema definitions
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_VALUES: Dict[str, Set[str]] = {
    "Gram Stain": {"Positive", "Negative", "Variable"},
    "Shape": {"Cocci", "Rods", "Bacilli", "Short Rods", "Spiral"},
    "Catalase": {"Positive", "Negative", "Variable"},
    "Oxidase": {"Positive", "Negative", "Variable"},
    "Colony Morphology": set(),
    "Haemolysis": {"Positive", "Negative", "Variable"},
    "Haemolysis Type": {"Alpha", "Beta", "Gamma", "None"},
    "Indole": {"Positive", "Negative", "Variable"},
    "Growth Temperature": set(),
    "Media Grown On": set(),
    "Motility": {"Positive", "Negative", "Variable"},
    "Capsule": {"Positive", "Negative", "Variable"},
    "Spore Formation": {"Positive", "Negative", "Variable"},
    "Oxygen Requirement": {"Aerobic", "Anaerobic", "Facultative Anaerobe", "Microaerophilic", "Capnophilic", "Intracellular"},
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
    "Coagulase": {"Positive", "Negative", "Variable"}
}

MEDIA_WHITELIST = {
    "MacConkey Agar", "Nutrient Agar", "Blood Agar", "XLD Agar",
    "Tryptic Soy Agar", "Columbia Blood Agar", "Brain Heart Infusion Agar",
    "Mannitol Salt Agar", "Chocolate Agar", "ALOA", "Palcam", "Preston",
    "BCYE Agar", "MRS", "Anaerobic Media"
}
MEDIA_EXCLUDE_TERMS = {"tsi", "triple sugar iron"}

NEGATION_CUES = [
    "not detected","not observed","no production","no growth",
    "non-fermenter","does not","fails to","unable to","absent"
]
VARIABLE_CUES = ["variable","inconsistent","weak","trace","slight"]

POLARITY_FIELDS = {k for k, v in ALLOWED_VALUES.items() if v == {"Positive", "Negative", "Variable"}}

VALUE_SYNONYMS = {
    "Gram Stain": {
        "gram positive": "Positive", "gram-negative": "Negative", "g+": "Positive", "g-": "Negative"
    },
    "Shape": {
        "rod": "Rods", "bacilli": "Bacilli", "cocci": "Cocci", "short rods": "Short Rods", "spiral": "Spiral"
    },
    "*POLARITY*": {
        "+": "Positive", "positive": "Positive", "pos": "Positive",
        "-": "Negative", "negative": "Negative", "neg": "Negative",
        "weak": "Variable", "trace": "Variable", "slight": "Variable", "variable": "Variable"
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────
def normalize_columns(fields: List[str]) -> List[str]:
    return [f for f in fields if f and f.lower() != "genus"]

def normalize_text(raw: str) -> str:
    t = raw or ""
    t = re.sub(r"[–—]", "-", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    t = re.sub(r"hemolys", "haemolys", t)
    return t

def _canon_value(field: str, value: str) -> str:
    if not value:
        return ""
    if field in POLARITY_FIELDS:
        low = value.lower()
        pol = VALUE_SYNONYMS.get("*POLARITY*", {})
        if low in pol:
            return pol[low]
        if "weak" in low or "variable" in low:
            return "Variable"
    if field in VALUE_SYNONYMS:
        low = value.lower()
        return VALUE_SYNONYMS[field].get(low, value)
    return value

def _set_field_safe(out: Dict[str, str], field: str, val: str):
    if not val:
        return
    cur = out.get(field)
    if cur is None:
        out[field] = val
    elif cur == "Variable" and val in {"Positive","Negative"}:
        out[field] = val
    else:
        out[field] = val

def build_alias_map(db_fields: List[str]) -> Dict[str, str]:
    alias = {}
    for f in db_fields:
        lower = f.lower()
        alias[lower] = f
        if "fermentation" in lower:
            alias[lower.replace(" fermentation", "")] = f
        if "decarboxylase" in lower:
            alias[lower.split()[0]] = f
        if "hydrolysis" in lower:
            alias[lower.split()[0]] = f
    alias["gram"] = "Gram Stain"
    alias["shape"] = "Shape"
    alias["oxidase"] = "Oxidase"
    alias["catalase"] = "Catalase"
    alias["coagulase"] = "Coagulase"
    alias["urease"] = "Urease"
    alias["vp"] = "VP"
    alias["mr"] = "Methyl Red"
    alias["indole"] = "Indole"
    alias["h2s"] = "H2S"
    alias["dnase"] = "Dnase"
    alias["gelatin"] = "Gelatin Hydrolysis"
    alias["lipase"] = "Lipase Test"
    alias["nitrate"] = "Nitrate Reduction"
    return alias

# ──────────────────────────────────────────────────────────────────────────────
# Pattern lists (these can be expanded automatically)
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
    r"\burease\s*(?:test)?\s*(?:\+|positive)\b",
    r"\burease\s*(?:test)?\s*(?:\-|negative|not\s+detected)\b",
]
MR_PATTERNS = [
    r"\bmethyl\s+red\s*(?:\+|positive)\b",
    r"\bmethyl\s+red\s*(?:\-|negative)\b",
]
VP_PATTERNS = [
    r"\bvp\s*(?:test)?\s*(?:\+|positive)\b",
    r"\bvp\s*(?:test)?\s*(?:\-|negative)\b",
]
CITRATE_PATTERNS = [
    r"\bcitrate\s*(?:test)?\s*(?:\+|positive)\b",
    r"\bcitrate\s*(?:test)?\s*(?:\-|negative)\b",
]
H2S_PATTERNS = [
    r"\bh\s*2\s*s\s*(?:\+|positive|produced)\b",
    r"\bh\s*2\s*s\s*(?:\-|negative|not\s+produced)\b",
]
# === CONTINUED: parser_basic.py Part 2 ===
# ──────────────────────────────────────────────────────────────────────────────
# Fermentation and other biochemical pattern extraction
# ──────────────────────────────────────────────────────────────────────────────
FERMENTATION_PATTERNS = [
    r"(?:ferments?|utilizes?|produces?\s+acid\s+from)\s+([a-z0-9\.\-%\s,/&]+)(?:\.|;|,|$)",
    r"(?:does\s+not|doesn't|cannot|unable\s+to)\s+(?:ferment|utilize)\s+([a-z0-9\.\-%\s,/&]+)",
    r"(?:ferments?|utilizes?)[^.]*?\bbut\s+not\s+([\w\s,;.&-]+)",
    r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b"
]

def extract_fermentations_regex(text: str, db_fields: List[str]) -> Dict[str, str]:
    out = {}
    t = normalize_text(text)
    alias = build_alias_map(db_fields)
    ferm_fields = [f for f in db_fields if f.lower().endswith(" fermentation")]
    base_to_field = {f[:-12].strip().lower(): f for f in ferm_fields}

    def set_field(base: str, val: str):
        b = base.strip().lower()
        if b in base_to_field:
            _set_field_safe(out, base_to_field[b], _canon_value(base_to_field[b], val))
        elif b in alias and alias[b] in db_fields:
            _set_field_safe(out, alias[b], _canon_value(alias[b], val))

    for pat in FERMENTATION_PATTERNS:
        for m in re.finditer(pat, t, flags=re.I|re.S):
            if len(m.groups()) == 2:
                sugar, sign = m.groups()
                set_field(sugar, "Positive" if sign == "+" else "Negative")
            elif m.group(1):
                for a in re.split(r"[,/&]", m.group(1)):
                    a = a.strip()
                    if not a:
                        continue
                    if "does not" in pat or "not" in pat:
                        set_field(a, "Negative")
                    else:
                        set_field(a, "Positive")

    if re.search(r"\bnlf\b", t):
        set_field("lactose", "Negative")
    if re.search(r"\blf\b", t) and "Lactose Fermentation" not in out:
        set_field("lactose", "Positive")

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Morphology / biochemical / oxygen extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_biochem_regex(text: str, db_fields: List[str]) -> Dict[str, str]:
    out = {}
    t = normalize_text(text)
    alias = build_alias_map(db_fields)

    def set_field(k, v):
        key = alias.get(k.lower(), k)
        if key in db_fields:
            _set_field_safe(out, key, _canon_value(key, v))

    # 1) Gram
    if re.search(r"\bgram[-\s]?positive\b", t):
        set_field("Gram Stain", "Positive")
    elif re.search(r"\bgram[-\s]?negative\b", t):
        set_field("Gram Stain", "Negative")

    # 2) Shape
    if re.search(r"\bcocci\b", t):
        set_field("Shape", "Cocci")
    if re.search(r"\brods?\b|bacilli\b", t):
        set_field("Shape", "Rods")
    if re.search(r"\bshort\s+rods\b", t):
        set_field("Shape", "Short Rods")
    if re.search(r"\bspiral\b", t):
        set_field("Shape", "Spiral")

    # 3) Motility
    if re.search(r"\bnon[-\s]?motile\b", t):
        set_field("Motility", "Negative")
    elif re.search(r"\bmotile\b", t):
        set_field("Motility", "Positive")

    # 4) Capsule
    if re.search(r"\bencapsulated|capsulated\b", t):
        set_field("Capsule", "Positive")
    if re.search(r"\bnon[-\s]?capsulated\b|\bcapsule\s+absent\b", t):
        set_field("Capsule", "Negative")

    # 5) Spore
    if re.search(r"\bspore[-\s]?forming\b|\bspores?\s+present\b", t):
        set_field("Spore Formation", "Positive")
    elif re.search(r"\bnon[-\s]?spore[-\s]?forming\b|\bno\s+spores?\b", t):
        set_field("Spore Formation", "Negative")

    # 6) Oxygen
    if re.search(r"\bfacultative\b", t):
        set_field("Oxygen Requirement", "Facultative Anaerobe")
    elif re.search(r"\baerobic\b", t):
        set_field("Oxygen Requirement", "Aerobic")
    elif re.search(r"\banaerobic\b", t):
        set_field("Oxygen Requirement", "Anaerobic")

    # 7) Enzyme tests from pattern lists
    pattern_map = {
        "Oxidase": OXIDASE_PATTERNS, "Catalase": CATALASE_PATTERNS, "Coagulase": COAGULASE_PATTERNS,
        "Indole": INDOLE_PATTERNS, "Urease": UREASE_PATTERNS, "Methyl Red": MR_PATTERNS,
        "VP": VP_PATTERNS, "Citrate": CITRATE_PATTERNS, "H2S": H2S_PATTERNS
    }
    for field, patterns in pattern_map.items():
        for p in patterns:
            for m in re.finditer(p, t, flags=re.I):
                val = None
                if re.search(r"\b(\+|positive|detected|produced)\b", m.group(0)):
                    val = "Positive"
                elif re.search(r"\b(\-|negative|not\s+detected|not\s+produced)\b", m.group(0)):
                    val = "Negative"
                if val:
                    set_field(field, val)

    # 8) Haemolysis
    if re.search(r"\b(beta|β)[-\s]?haem", t):
        set_field("Haemolysis Type", "Beta")
        set_field("Haemolysis", "Positive")
    elif re.search(r"\b(alpha|α)[-\s]?haem", t):
        set_field("Haemolysis Type", "Alpha")
        set_field("Haemolysis", "Positive")
    elif re.search(r"\b(gamma|γ)[-\s]?haem\b|\bno\s+haemolysis\b", t):
        set_field("Haemolysis Type", "Gamma")
        set_field("Haemolysis", "Variable")

    # 9) Growth temperature
    m = re.search(r"grows\s+(?:between|from)\s+(\d{1,2})\s*(?:to|and)\s*(\d{1,2})", t)
    if m:
        set_field("Growth Temperature", f"{m.group(1)}//{m.group(2)}")
    m2 = re.search(r"grows\s+at\s+(\d{1,2})\s*°?\s*c", t)
    if m2:
        set_field("Growth Temperature", m2.group(1))

    # 10) NaCl tolerance
    if re.search(r"\btolerant\s+(?:in|up\s+to|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t):
        set_field("NaCl Tolerant (>=6%)", "Positive")
    if re.search(r"\bno\s+growth\s+(?:in|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t):
        set_field("NaCl Tolerant (>=6%)", "Negative")

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────────────
def normalize_to_schema(parsed: Dict[str, str], db_fields: List[str]) -> Dict[str, str]:
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)
    out = {}
    strict = os.getenv("BACTAI_STRICT_MODE", "0") == "1"

    for k, v in parsed.items():
        kk = k.strip()
        if kk in fields:
            out[kk] = _canon_value(kk, v)
        elif kk.lower() in alias:
            t = alias[kk.lower()]
            if t in fields:
                out[t] = _canon_value(t, v)
        elif not strict:
            out[kk] = v

    return out

# ──────────────────────────────────────────────────────────────────────────────
# MAIN: Parse text (regex-only)
# ──────────────────────────────────────────────────────────────────────────────
def parse_input_free_text(user_text: str, prior_facts: Optional[Dict] = None, db_fields: Optional[List[str]] = None) -> Dict[str, str]:
    if not user_text:
        return {}
    db_fields = db_fields or list(ALLOWED_VALUES.keys())
    facts = prior_facts or {}
    out = dict(facts)
    out.update(extract_biochem_regex(user_text, db_fields))
    out.update(extract_fermentations_regex(user_text, db_fields))
    return normalize_to_schema(out, db_fields)

# ──────────────────────────────────────────────────────────────────────────────
# GOLD TESTS + FEEDBACK LEARNING + AUTO PATCHING
# ──────────────────────────────────────────────────────────────────────────────
def _diff_for_feedback(expected, got):
    diffs = []
    for k, exp in expected.items():
        g = got.get(k, "")
        if str(exp) != str(g):
            diffs.append({"field": k, "expected": exp, "got": g})
    return diffs

def _log_feedback_case(name, text, diffs):
    feedback = _load_json(FEEDBACK_PATH, [])
    feedback.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "text": text,
        "errors": diffs
    })
    _save_json(FEEDBACK_PATH, feedback)

def run_gold_tests(db_fields=None):
    print("Running Gold Tests (regex mode)...")
    tests = _load_json(GOLD_TESTS_PATH, [])
    if not tests:
        print("No gold_tests.json found.")
        return (0, 0)

    db_fields = db_fields or list(ALLOWED_VALUES.keys())
    passed, total = 0, 0

    for case in tests:
        total += 1
        name = case.get("name", f"case_{total}")

        # Parse the user input into structured fields
        got = parse_input_free_text(case["input"], db_fields=db_fields)

        # 🧩 PATCH: Only validate expected fields that exist in the current DB schema
        expected_raw = case.get("expected", {})
        expected = {k: v for k, v in expected_raw.items() if k in db_fields}

        diffs = _diff_for_feedback(expected, got)

        if not diffs:
            passed += 1
            print(f"✅ {name} passed.")
        else:
            print(f"❌ {name} failed.")
            for d in diffs:
                print(f"   - {d['field']}: expected '{d['expected']}' got '{d['got']}'")
            _log_feedback_case(name, case["input"], diffs)

    print(f"Gold Tests complete: {passed}/{total}")
    return (passed, total)

def analyze_feedback_and_learn():
    feedback = _load_json(FEEDBACK_PATH, [])
    if not feedback:
        return
    memory = _load_json(MEMORY_PATH, {})
    counts = {}
    for f in feedback:
        for e in f.get("errors", []):
            fld = e["field"]
            counts[fld] = counts.get(fld, 0) + 1
    auto_heuristics = {fld: {"rule": "regex adjustment", "count": c}
                       for fld, c in counts.items() if c >= 3}
    memory["auto_heuristics"] = auto_heuristics
    _save_json(MEMORY_PATH, memory)
    print(f"🧠 Learned {len(auto_heuristics)} field heuristics.")

def auto_update_parser_regex():
    memory = _load_json(MEMORY_PATH, {})
    auto = memory.get("auto_heuristics", {})
    if not auto:
        print("No new heuristics.")
        return
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print("Error reading self:", e)
        return

    updated = 0
    for field, data in auto.items():
        field_l = field.lower()
        pat_list = None
        if "oxidase" in field_l:
            pat_list = "OXIDASE_PATTERNS"
        elif "catalase" in field_l:
            pat_list = "CATALASE_PATTERNS"
        elif "coagulase" in field_l:
            pat_list = "COAGULASE_PATTERNS"
        elif "indole" in field_l:
            pat_list = "INDOLE_PATTERNS"
        elif "urease" in field_l:
            pat_list = "UREASE_PATTERNS"
        elif "h2s" in field_l:
            pat_list = "H2S_PATTERNS"
        elif "vp" in field_l:
            pat_list = "VP_PATTERNS"
        elif "methyl red" in field_l or "mr" in field_l:
            pat_list = "MR_PATTERNS"
        elif "citrate" in field_l:
            pat_list = "CITRATE_PATTERNS"
        elif "ferment" in field_l:
            pat_list = "FERMENTATION_PATTERNS"
        if not pat_list:
            continue
        new_line = f'    r"\\\\b{re.escape(field_l)}\\\\b.*(?:positive|negative)",  # learned {data["count"]}x\n'
        code = re.sub(rf"({re.escape(pat_list)}\\s*=\\s*\\[)([^\\]]*)(\\])", rf"\1\2{new_line}\3", code, flags=re.S)
        updated += 1

    if updated:
        with open(__file__, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"🧬 Updated parser_basic.py with {updated} new regex rules.")
    else:
        print("No applicable lists for auto-insert.")

# ──────────────────────────────────────────────────────────────────────────────
# Self-learning bootstrap for Streamlit
# ──────────────────────────────────────────────────────────────────────────────
def enable_self_learning_autopatch(run_tests=False, db_fields=None):
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
    print("parser_basic.py ready (regex self-learning enabled).")
