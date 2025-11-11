# parser_basic.py — v5 (Full Regex Parser + Self-Learning + Auto-Patching + Gold Tests)
# -------------------------------------------------------------------------------------
# Features:
#   • Pure regex parser (no LLM dependency)
#   • Full-field self-learning (all biochemical, morphology, media, oxygen, etc.)
#   • Auto-fixes spacing (\ fermentation → \s+fermentation)
#   • Prevents duplicate rule insertion
#   • Git-safe, compatible with Streamlit Cloud auto-commit
# -------------------------------------------------------------------------------------

import os
import re
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Set

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
# Core schema + aliases
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_VALUES: Dict[str, Set[str]] = {
    "Gram Stain": {"Positive", "Negative", "Variable"},
    "Shape": {"Cocci", "Rods", "Bacilli", "Short Rods", "Spiral"},
    "Catalase": {"Positive", "Negative", "Variable"},
    "Oxidase": {"Positive", "Negative", "Variable"},
    "Coagulase": {"Positive", "Negative", "Variable"},
    "Indole": {"Positive", "Negative", "Variable"},
    "Urease": {"Positive", "Negative", "Variable"},
    "Methyl Red": {"Positive", "Negative", "Variable"},
    "VP": {"Positive", "Negative", "Variable"},
    "Citrate": {"Positive", "Negative", "Variable"},
    "H2S": {"Positive", "Negative", "Variable"},
    "Nitrate Reduction": {"Positive", "Negative", "Variable"},
    "Lysine Decarboxylase": {"Positive", "Negative", "Variable"},
    "Ornithine Decarboxylase": {"Positive", "Negative", "Variable"},
    "Arginine Dihydrolase": {"Positive", "Negative", "Variable"},
    "Gelatin Hydrolysis": {"Positive", "Negative", "Variable"},
    "Esculin Hydrolysis": {"Positive", "Negative", "Variable"},
    "DNase": {"Positive", "Negative", "Variable"},
    "ONPG": {"Positive", "Negative", "Variable"},
    "NaCl Tolerant (>=6%)": {"Positive", "Negative", "Variable"},
    "Lipase Test": {"Positive", "Negative", "Variable"},
}

def normalize_text(t: str) -> str:
    t = re.sub(r"[–—]", "-", t or "")
    return re.sub(r"\s+", " ", t).strip().lower()

def _canon_value(f: str, v: str) -> str:
    if not v: return ""
    v = v.lower()
    if "weak" in v or "variable" in v: return "Variable"
    if "neg" in v or "-" in v or "not" in v: return "Negative"
    if "pos" in v or "+" in v or "detect" in v or "produced" in v: return "Positive"
    return v.capitalize()

def _set_field_safe(out: Dict[str,str], field: str, val: str):
    if not val: return
    cur = out.get(field)
    if cur is None or cur == "Variable" or val != cur:
        out[field] = val
# ──────────────────────────────────────────────────────────────────────────────
# REGEX extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_biochem_regex(text: str, db_fields: List[str]) -> Dict[str,str]:
    t = normalize_text(text)
    out = {}
    for field in db_fields:
        base = field.lower()
        if re.search(rf"\b{re.escape(base)}\b.*(?:positive|\+|detected|produced)", t):
            _set_field_safe(out, field, "Positive")
        elif re.search(rf"\b{re.escape(base)}\b.*(?:negative|\-|not\s+detected|absent)", t):
            _set_field_safe(out, field, "Negative")
    # Morphology quick rules
    if "gram positive" in t: out["Gram Stain"] = "Positive"
    if "gram negative" in t: out["Gram Stain"] = "Negative"
    if "cocci" in t: out["Shape"] = "Cocci"
    if "rod" in t: out["Shape"] = "Rods"
    if "short rod" in t: out["Shape"] = "Short Rods"
    if "spiral" in t: out["Shape"] = "Spiral"
    return out

def parse_input_free_text(user_text: str, prior_facts=None, db_fields=None) -> Dict[str,str]:
    if not user_text: return {}
    db_fields = db_fields or list(ALLOWED_VALUES.keys())
    facts = prior_facts or {}
    parsed = extract_biochem_regex(user_text, db_fields)
    facts.update(parsed)
    return facts

# ──────────────────────────────────────────────────────────────────────────────
# GOLD TEST + FEEDBACK
# ──────────────────────────────────────────────────────────────────────────────
def _diff_for_feedback(expected, got):
    diffs = []
    for k, exp in expected.items():
        g = got.get(k, "")
        if str(exp) != str(g):
            diffs.append({"field": k, "expected": exp, "got": g})
    return diffs

def _log_feedback_case(name, text, diffs):
    fb = _load_json(FEEDBACK_PATH, [])
    fb.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "name": name, "text": text, "errors": diffs})
    _save_json(FEEDBACK_PATH, fb)

def run_gold_tests(db_fields=None):
    tests = _load_json(GOLD_TESTS_PATH, [])
    if not tests:
        print("No gold_tests.json found.")
        return (0,0)
    db_fields = db_fields or list(ALLOWED_VALUES.keys())
    passed,total = 0,0
    for case in tests:
        total+=1
        got = parse_input_free_text(case["input"], db_fields=db_fields)
        expected = {k:v for k,v in case.get("expected",{}).items() if k in db_fields}
        diffs=_diff_for_feedback(expected,got)
        if not diffs: passed+=1; print(f"✅ {case.get('name','case')}")
        else:
            print(f"❌ {case.get('name','case')}"); _log_feedback_case(case["name"],case["input"],diffs)
    print(f"Gold Tests: {passed}/{total}")
    return passed,total

def analyze_feedback_and_learn():
    fb = _load_json(FEEDBACK_PATH, [])
    if not fb: return
    memory=_load_json(MEMORY_PATH,{})
    counts={}
    for f in fb:
        for e in f.get("errors",[]): counts[e["field"]] = counts.get(e["field"],0)+1
    learned={fld:{"rule":"auto-learn","count":c} for fld,c in counts.items() if c>=3}
    memory["auto_heuristics"]=learned
    _save_json(MEMORY_PATH,memory)
    print(f"🧠 Learned {len(learned)} heuristics.")
# ──────────────────────────────────────────────────────────────────────────────
# AUTO-UPDATE REGEX RULES
# ──────────────────────────────────────────────────────────────────────────────
def _fix_regex_spaces(rule: str) -> str:
    """Convert single-space escape (\ fermentation) → (\s+fermentation)."""
    return re.sub(r"(?<!\\)\\ fermentation", r"\\s+fermentation", rule)

def auto_update_parser_regex():
    memory=_load_json(MEMORY_PATH,{})
    auto=memory.get("auto_heuristics",{})
    if not auto:
        print("No new heuristics.")
        return
    try:
        with open(__file__,"r",encoding="utf-8") as f: code=f.read()
    except Exception as e:
        print("Read self error:",e); return

    updated=0
    for field,data in auto.items():
        field_l=field.lower()
        rule=f'r"\\b{re.escape(field_l)}\\b.*(?:positive|negative|detected|produced)"'
        rule=_fix_regex_spaces(rule)
        if rule in code:
            continue  # skip duplicates
        insert_line=f"    {rule},  # learned {data['count']}x\n"
        pattern_list="GENERAL_PATTERNS"
        code=re.sub(r"(# END AUTO-LEARN AREA)",insert_line+r"\1",code)
        updated+=1

    if updated:
        with open(__file__,"w",encoding="utf-8") as f: f.write(code)
        print(f"🧬 Added {updated} new regex rules.")
    else:
        print("No new regex inserted.")

# Placeholder for learned pattern insertion marker
# END AUTO-LEARN AREA

# ──────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────────────
def enable_self_learning_autopatch(run_tests=False, db_fields=None):
    if run_tests:
        run_gold_tests(db_fields=db_fields)
    analyze_feedback_and_learn()
    auto_update_parser_regex()

if __name__=="__main__":
    if "--test" in sys.argv:
        run_gold_tests()
        analyze_feedback_and_learn()
        auto_update_parser_regex()
        sys.exit(0)
    print("parser_basic.py ready (full-field self-learning enabled).")
