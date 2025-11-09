# parser_llm.py  —  Step 1.6f (Hybrid LLM + Regex Parser, import-safe full version)

import os, json, re
from typing import Dict, List
from parser_basic import parse_input_free_text as fallback_parser


# ==========================================================
# Helper: categorize database fields
# ==========================================================
def summarize_field_categories(db_fields: List[str]) -> Dict[str, List[str]]:
    """Group database column names into rough categories."""
    cats = {"Morphology": [], "Enzyme": [], "Fermentation": [], "Other": []}
    for f in db_fields:
        n = f.strip()
        l = n.lower()
        if l == "genus":
            continue
        if any(k in l for k in ["gram", "shape", "morphology", "motility", "capsule", "spore"]):
            cats["Morphology"].append(n)
        elif any(k in l for k in ["oxidase", "catalase", "urease", "coagulase", "lipase", "test"]):
            cats["Enzyme"].append(n)
        elif "fermentation" in l or "utilization" in l:
            cats["Fermentation"].append(n)
        else:
            cats["Other"].append(n)
    return cats


# ==========================================================
# Helpers for deterministic parsing
# ==========================================================
def _base_name(field: str) -> str:
    """Normalize DB field to its root analyte name."""
    return (
        field.lower()
        .replace(" fermentation", "")
        .replace(" utilization", "")
        .replace(" test", "")
        .strip()
    )


def _tokenize_analyte_list(s: str) -> List[str]:
    """Split 'glucose and sucrose, not lactose' → ['glucose','sucrose','lactose']"""
    s = re.sub(r"\s*(?:,|and|or|&)\s*", ",", s.strip(), flags=re.I)
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    return tokens


# ==========================================================
# Regex rule-based extractor for fermentations & shorthands
# ==========================================================
def extract_fermentations_regex(text: str, fermentation_fields: List[str]) -> Dict[str, str]:
    """Detect explicit fermentation/utilization and shorthand patterns."""
    out: Dict[str, str] = {}
    t = text.lower()

    # Build base → field mapping
    base_to_field = {}
    for f in fermentation_fields:
        base_to_field.setdefault(_base_name(f), set()).add(f)

    # 1) Positive lists  (ferments / utilizes ...)
    for m in re.finditer(r"(?:ferments|utilizes)\s+([a-z0-9\.\-%\s,/&]+)", t):
        for a in _tokenize_analyte_list(m.group(1)):
            b = a.replace("(", "").replace(")", "").strip().lower()
            if b in base_to_field:
                for field in base_to_field[b]:
                    out[field] = "Positive"

    # 2) Negative lists  (does not ferment / non-fermenter for ...)
    neg_pats = [
        r"(?:does\s+not\s+(?:ferment|utilize)|cannot\s+(?:ferment|utilize)|unable\s+to\s+(?:ferment|utilize))\s+([a-z0-9\.\-%\s,/&]+)",
        r"non[-\s]?fermenter\s+(?:for|of)?\s+([a-z0-9\.\-%\s,/&]+)",
    ]
    for pat in neg_pats:
        for m in re.finditer(pat, t):
            for a in _tokenize_analyte_list(m.group(1)):
                b = a.replace("(", "").replace(")", "").strip().lower()
                if b in base_to_field:
                    for field in base_to_field[b]:
                        out[field] = "Negative"

    # 3) Shorthand:  'lactose -', 'rhamnose +'
    for m in re.finditer(r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b", t):
        a, sign = m.group(1).lower(), m.group(2)
        if a in base_to_field:
            for field in base_to_field[a]:
                out[field] = "Positive" if sign == "+" else "Negative"

    # 4) ONPG and NaCl tolerance
    if re.search(r"\bonpg\s*(?:test)?\s*(\+|positive)\b", t):
        out["ONPG Test"] = "Positive"
    elif re.search(r"\bonpg\s*(?:test)?\s*(\-|negative)\b", t):
        out["ONPG Test"] = "Negative"

    if re.search(r"\bgrows\s+in\s+[0-9\.]+\s*%?\s*na\s*cl\b", t):
        out["NaCl Tolerance"] = "Positive"
    if re.search(r"\bno\s+growth\s+in\s+[0-9\.]+\s*%?\s*na\s*cl\b", t):
        out["NaCl Tolerance"] = "Negative"

    return out


# ==========================================================
# Prompt builders for the LLM step
# ==========================================================
def build_prompt(user_text: str, cats: Dict[str, List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    morph = ", ".join(cats["Morphology"][:10])
    enz = ", ".join(cats["Enzyme"][:10])
    ferm = ", ".join(cats["Fermentation"][:10])
    other = ", ".join(cats["Other"][:10])
    system = (
        "You parse microbiology observations into structured results. "
        "Focus on morphology, enzyme, and growth traits. "
        "Fermentation fields are handled separately by regex rules. "
        "Return JSON; unmentioned fields='Unknown'.\n"
        f"Morphology: {morph}\nEnzyme: {enz}\nFermentation examples: {ferm}\nOther: {other}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Previous facts:\n{prior}\nObservation:\n{user_text}"},
    ]


def build_prompt_text(user_text: str, cats: Dict[str, List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    return (
        "Extract morphology, enzyme, and growth results from this microbiology description. "
        "Leave carbohydrate fermentations to regex rules. "
        "Return JSON; unmentioned fields='Unknown'.\n\n"
        f"Previous facts:\n{prior}\nObservation:\n{user_text}"
    )


# ==========================================================
# MAIN ENTRY POINT — what app_chat.py imports
# ==========================================================
def parse_input_free_text(
    user_text: str,
    prior_facts: Dict | None = None,
    db_fields: List[str] | None = None,
) -> Dict:
    """Main parser combining LLM and deterministic extraction."""
    if not user_text.strip():
        return {}

    db_fields = [f for f in (db_fields or []) if f.lower() != "genus"]
    cats = summarize_field_categories(db_fields)

    # --- Step 1: LLM parsing (fallback-safe)
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
            parsed = json.loads(resp.choices[0].message.content)
        else:
            import ollama
            prompt = build_prompt_text(user_text, cats, prior_facts)
            out = ollama.chat(
                model=os.getenv("LOCAL_MODEL", "llama3"),
                messages=[{"role": "user", "content": prompt}],
            )
            m = re.search(r"\{.*\}", out["message"]["content"], re.S)
            parsed = json.loads(m.group(0)) if m else {}
    except Exception as e:
        print("⚠️ LLM parser failed — fallback:", e)
        parsed = fallback_parser(user_text, prior_facts)

    # --- Step 2: Deterministic regex add-ons
    regex_hits = extract_fermentations_regex(user_text, cats["Fermentation"])
    parsed.update(regex_hits)

    return parsed
