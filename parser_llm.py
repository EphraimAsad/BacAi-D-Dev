import os, json, re
from typing import Dict, List
from parser_basic import parse_input_free_text as fallback_parser


# ==========================================================
#  Field categorization
# ==========================================================
def summarize_field_categories(db_fields: List[str]) -> Dict[str, List[str]]:
    cats = {"Morphology": [], "Enzyme": [], "Fermentation": [], "Other": []}
    for f in db_fields:
        n = f.strip()
        l = n.lower()
        if l == "genus":
            continue
        if any(k in l for k in ["gram", "shape", "morphology", "motility", "capsule", "spore"]):
            cats["Morphology"].append(n)
        elif any(k in l for k in ["oxidase", "catalase", "urease", "coagulase",
                                  "lipase", "indole", "citrate", "vp", "mr",
                                  "gelatin", "starch", "nitrate", "h2s", "test"]):
            cats["Enzyme"].append(n)
        elif "fermentation" in l or "utilization" in l:
            cats["Fermentation"].append(n)
        else:
            cats["Other"].append(n)
    return cats


# ==========================================================
#  Utility helpers
# ==========================================================
def _base_name(field: str) -> str:
    return (
        field.lower()
        .replace(" fermentation", "")
        .replace(" utilization", "")
        .replace(" test", "")
        .strip()
    )

def _normalize_analyte_token(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"[.,;:!?\u2013\u2014\-]+$", "", s)
    return s.strip()

def _tokenize_analyte_list(s: str) -> List[str]:
    s = re.sub(r"\s*(?:,|and|or|&|nor)\s*", ",", s.strip(), flags=re.I)
    return [t.strip() for t in s.split(",") if t.strip()]


# ==========================================================
#  Deterministic pattern extractor
# ==========================================================
def extract_patterns_regex(text: str, fields: List[str]) -> Dict[str, str]:
    """General biochemical pattern extractor."""
    out: Dict[str, str] = {}
    t = text.lower()

    # 1️⃣ generic enzyme test positives / negatives
    enzyme_targets = [
        "catalase", "oxidase", "coagulase", "urease", "lipase", "indole",
        "citrate", "vp", "mr", "gelatin", "starch", "nitrate", "h2s"
    ]
    for test in enzyme_targets:
        pos_pat = rf"\b{test}\s*(?:test)?\s*(?:\+|positive|detected)\b"
        neg_pat = rf"\b{test}\s*(?:test)?\s*(?:\-|negative|not\s+detected)\b"
        var_pat = rf"\b{test}\s*(?:test)?\s*weak(?:ly)?\s*positive\b"
        if re.search(pos_pat, t):
            out[f"{test.capitalize()}"] = "Positive"
        elif re.search(neg_pat, t):
            out[f"{test.capitalize()}"] = "Negative"
        elif re.search(var_pat, t):
            out[f"{test.capitalize()}"] = "Variable"

    # 2️⃣ Haemolysis type
    if re.search(r"\b(beta|β)[-\s]?haem", t):
        out["Haemolysis Type"] = "Beta"
    elif re.search(r"\b(alpha|α)[-\s]?haem", t):
        out["Haemolysis Type"] = "Alpha"
    elif re.search(r"\b(gamma|γ)[-\s]?haem", t):
        out["Haemolysis Type"] = "Gamma"

    # 3️⃣ NaCl tolerance
    if re.search(r"\bgrows\s+in\s+[0-9\.]+\s*%?\s*na\s*cl\b", t):
        out["NaCl Tolerance"] = "Positive"
    if re.search(r"\bno\s+growth\s+in\s+[0-9\.]+\s*%?\s*na\s*cl\b", t):
        out["NaCl Tolerance"] = "Negative"

    # 4️⃣ Growth temperature extraction
    for m in re.finditer(r"grows\s+at\s+([0-9]{2,3})\s*°?\s*c", t):
        out["Growth Temperature"] = m.group(1)
    if re.search(r"no\s+growth\s+at\s+([0-9]{2,3})", t):
        # optional secondary temp field if needed
        out["No Growth Temperature"] = re.findall(r"no\s+growth\s+at\s+([0-9]{2,3})", t)[0]

    return out


def extract_fermentations_regex(text: str, fermentation_fields: List[str]) -> Dict[str, str]:
    """Extract fermentation/utilization patterns, shorthand, ONPG, NaCl."""
    out: Dict[str, str] = {}
    t = text.lower()

    # base → field map
    base_to_field = {}
    for f in fermentation_fields:
        base_to_field.setdefault(_base_name(f), set()).add(f)

    def set_val(base: str, val: str):
        if base in base_to_field:
            for field in base_to_field[base]:
                out[field] = val

    # 1️⃣ positive after "ferments"/"utilizes"
    for m in re.finditer(r"(?:ferments|utilizes)\s+([a-z0-9\.\-%\s,/&]+)", t):
        span = re.split(r"\bbut\s+not\b", m.group(1))[0]
        for a in _tokenize_analyte_list(span):
            set_val(_normalize_analyte_token(a), "Positive")

    # 2️⃣ negatives via common phrases
    neg_pats = [
        r"(?:does\s+not|doesn't)\s+(?:ferment|utilize)\s+([a-z0-9\.\-%\s,/&]+)",
        r"cannot\s+(?:ferment|utilize)\s+([a-z0-9\.\-%\s,/&]+)",
        r"unable\s+to\s+(?:ferment|utilize)\s+([a-z0-9\.\-%\s,/&]+)",
        r"non[-\s]?fermenter\s+(?:for|of)?\s+([a-z0-9\.\-%\s,/&]+)",
    ]
    for pat in neg_pats:
        for m in re.finditer(pat, t):
            for a in _tokenize_analyte_list(m.group(1)):
                set_val(_normalize_analyte_token(a), "Negative")

    # 3️⃣ negatives after "but not X, Y or Z / nor Z"
    for m in re.finditer(
        r"(?:ferments|utilizes)[^\.]*?\bbut\s+not\s+([a-z0-9\.\-%\s,/&\sandornor]+)", t
    ):
        for a in _tokenize_analyte_list(m.group(1)):
            set_val(_normalize_analyte_token(a), "Negative")

    # 4️⃣ shorthand  lactose -, rhamnose +
    for m in re.finditer(r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b", t):
        a, sign = m.group(1), m.group(2)
        set_val(_normalize_analyte_token(a), "Positive" if sign == "+" else "Negative")

    # 5️⃣ non-lactose fermenting / fermenter
    for m in re.finditer(r"\bnon[-\s]?([a-z0-9\-]+)\s+ferment(?:er|ing)?\b", t):
        set_val(_normalize_analyte_token(m.group(1)), "Negative")

    # 6️⃣ ONPG
    if re.search(r"\bonpg\s*(?:test)?\s*(\+|positive)\b", t):
        out["ONPG Test"] = "Positive"
    elif re.search(r"\bonpg\s*(?:test)?\s*(\-|negative)\b", t):
        out["ONPG Test"] = "Negative"

    return out


# ==========================================================
#  Prompt builders
# ==========================================================
def build_prompt(user_text: str, cats: Dict[str, List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    morph = ", ".join(cats["Morphology"][:10])
    enz = ", ".join(cats["Enzyme"][:10])
    ferm = ", ".join(cats["Fermentation"][:10])
    other = ", ".join(cats["Other"][:10])
    system = (
        "You parse microbiology observations into structured results. "
        "Focus on morphology, enzyme, and growth traits; fermentation handled separately. "
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


# ==========================================================
#  MAIN ENTRY POINT
# ==========================================================
def parse_input_free_text(
    user_text: str,
    prior_facts: Dict | None = None,
    db_fields: List[str] | None = None,
) -> Dict:
    if not user_text.strip():
        return {}

    db_fields = [f for f in (db_fields or []) if f.lower() != "genus"]
    cats = summarize_field_categories(db_fields)

    # --- Step 1 : LLM parsing
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

    # --- Step 2 : regex enrichment
    parsed.update(extract_fermentations_regex(user_text, cats["Fermentation"]))
    parsed.update(extract_patterns_regex(user_text, db_fields))

    return parsed
