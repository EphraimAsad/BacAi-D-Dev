# parser_llm.py — Step 1.6 l (Expanded regex coverage & schema-aligned)

import os, json, re
from typing import Dict, List
from parser_basic import parse_input_free_text as fallback_parser


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

    # Canonical test names
    add("mr", "Methyl Red")
    add("methyl red", "Methyl Red")
    add("vp", "VP")
    add("voges proskauer", "VP")
    add("h2s", "H2S")
    add("dnase", "Dnase")
    add("gelatin", "Gelatin Hydrolysis")
    add("gelatin liquefaction", "Gelatin Hydrolysis")
    add("lipase", "Lipase Test")
    add("lipase test", "Lipase Test")
    add("onpg", "ONPG")
    add("onpg test", "ONPG")
    add("nacl tolerance", "NaCl Tolerant (>=6%)")
    add("nacl tolerant", "NaCl Tolerant (>=6%)")
    add("nacl", "NaCl Tolerant (>=6%)")
    add("nitrate", "Nitrate Reduction")
    add("nitrate reduction", "Nitrate Reduction")
    add("lysine decarboxylase", "Lysine Decarboxylase")
    add("ornithine decarboxylase", "Ornitihine Decarboxylase")   # matches your sheet spelling
    add("ornitihine decarboxylase", "Ornitihine Decarboxylase")
    add("arginine dihydrolase", "Arginine dihydrolase")
    add("coagulase", "Coagulase")
    add("citrate", "Citrate")
    add("urease", "Urease")
    add("indole", "Indole")
    add("oxidase", "Oxidase")
    add("catalase", "Catalase")
    add("motility", "Motility")
    add("capsule", "Capsule")
    add("spore formation", "Spore Formation")
    add("haemolysis", "Haemolysis")
    add("haemolysis type", "Haemolysis Type")
    add("growth temperature", "Growth Temperature")
    add("media grown on", "Media Grown On")
    add("oxygen requirement", "Oxygen Requirement")

    # Fermentation bases (e.g., "rhamnose" → "Rhamnose Fermentation")
    for f in normalize_columns(db_fields):
        if f.lower().endswith(" fermentation"):
            base = f[:-12].strip().lower()
            alias[base] = f

    return alias


# ──────────────────────────────────────────────────────────────────────────────
# Text utilities
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_token(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"[.,;:!?\u2013\u2014\-]+$", "", s)
    return s.strip()


def _tokenize_list(s: str) -> List[str]:
    """Split by ',', 'and', 'or', '&', 'nor'."""
    s = re.sub(r"\s*(?:,|and|or|&|nor)\s*", ",", s.strip(), flags=re.I)
    return [t.strip() for t in s.split(",") if t.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Regex enrichment: fermentations + ONPG + NaCl
# ──────────────────────────────────────────────────────────────────────────────
def extract_fermentations_regex(text: str, db_fields: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    t = text.lower()
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)

    ferm_fields = [f for f in fields if f.lower().endswith(" fermentation")]
    base_to_field = {f[:-12].strip().lower(): f for f in ferm_fields}

    def set_field_by_base(base: str, val: str):
        b = _normalize_token(base)
        if b in base_to_field:
            out[base_to_field[b]] = val
        elif b in alias and alias[b] in fields:
            out[alias[b]] = val

    # POSITIVE lists ("ferments"/"utilizes")
    for m in re.finditer(r"(?:ferments|utilizes)\s+([a-z0-9\.\-%\s,/&]+)", t):
        span = re.split(r"\bbut\s+not\b", m.group(1))[0]
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
        for m in re.finditer(pat, t):
            for a in _tokenize_list(m.group(1)):
                set_field_by_base(a, "Negative")

    # NEGATIVE after "but not X, Y or Z / nor Z"
    for m in re.finditer(r"(?:ferments|utilizes)[^\.]*?\bbut\s+not\s+([a-z0-9\.\-%\s,/&\sandornor]+)", t):
        for a in _tokenize_list(m.group(1)):
            set_field_by_base(a, "Negative")

    # Shorthand "lactose -" / "rhamnose +"
    for m in re.finditer(r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b", t):
        a, sign = m.group(1), m.group(2)
        set_field_by_base(a, "Positive" if sign == "+" else "Negative")

    # ONPG
    if re.search(r"\bonpg\s*(?:test)?\s*(\+|positive)\b", t):
        if "onpg" in alias and alias["onpg"] in fields:
            out[alias["onpg"]] = "Positive"
    elif re.search(r"\bonpg\s*(?:test)?\s*(\-|negative)\b", t):
        if "onpg" in alias and alias["onpg"] in fields:
            out[alias["onpg"]] = "Negative"

    # NaCl tolerance phrases (in / up to … % NaCl)
    if re.search(r"\b(tolerant|grows|growth)\s+(?:in|up\s+to)\s+[0-9\.]+\s*%?\s*na\s*cl\b", t):
        if "nacl tolerant" in alias and alias["nacl tolerant"] in fields:
            out[alias["nacl tolerant"]] = "Positive"
    if re.search(r"\bno\s+growth\s+(?:in|at)\s+[0-9\.]+\s*%?\s*na\s*cl\b", t):
        if "nacl tolerant" in alias and alias["nacl tolerant"] in fields:
            out[alias["nacl tolerant"]] = "Negative"

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Regex enrichment: enzyme/other tests, capsule, haemolysis, growth temp
# ──────────────────────────────────────────────────────────────────────────────
def extract_biochem_regex(text: str, db_fields: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    t = text.lower()
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)

    def set_field(key_like: str, val: str):
        target = alias.get(key_like.lower(), key_like)
        if target in fields:
            out[target] = val

    # Generic biochemical (+ / - / weakly positive)
    generic_targets = [
        "catalase","oxidase","coagulase","urease","lipase","indole",
        "citrate","vp","methyl red","gelatin","dnase","nitrate reduction","nitrate","h2s","esculin hydrolysis"
    ]
    for test in generic_targets:
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:\+|positive|detected)\b", t):
            set_field(test, "Positive")
        elif re.search(rf"\b{test}\s*(?:test)?\s*(?:\-|negative|not\s+detected)\b", t):
            set_field(test, "Negative")
        elif re.search(rf"\b{test}\s*(?:test)?\s*weak(?:ly)?\s*positive\b", t):
            set_field(test, "Variable")

    # Phrase variants
    if re.search(r"\breduces\s+nitrate\b", t):
        set_field("nitrate", "Positive")
    if re.search(r"\bdoes\s+not\s+reduce\s+nitrate\b", t):
        set_field("nitrate", "Negative")
    if re.search(r"\bproduces\s+h\s*2\s*s\b", t):
        set_field("h2s", "Positive")
    if re.search(r"\bh\s*2\s*s\s+negative\b", t):
        set_field("h2s", "Negative")
    if re.search(r"\bgelatin\s+liquefaction\s+(?:\+|positive)\b", t):
        set_field("gelatin", "Positive")
    if re.search(r"\besculin\s+hydrolysis\s+(?:\+|positive)\b", t):
        set_field("esculin hydrolysis", "Positive")

    # Capsule
    if re.search(r"\b(capsulated|encapsulated)\b", t):
        set_field("capsule", "Positive")
    if re.search(r"\bnon[-\s]?capsulated\b", t):
        set_field("capsule", "Negative")

    # Haemolysis type → exact column
    if re.search(r"\b(beta|β)[-\s]?haem", t):
        set_field("haemolysis type", "Beta")
    elif re.search(r"\b(alpha|α)[-\s]?haem", t):
        set_field("haemolysis type", "Alpha")
    elif re.search(r"\b(gamma|γ)[-\s]?haem", t):
        set_field("haemolysis type", "Gamma")

    # Growth Temperature:
    # - If "grows at X °C" appears, set numeric X.
    # - If "no growth at Y °C" appears, DO NOT set the column (we keep numeric semantics).
    for m in re.finditer(r"\bgrows\s+at\s+([0-9]{2,3})\s*°?\s*c", t):
        temp_num = m.group(1)
        set_field("growth temperature", temp_num)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders (LLM is optional; regex handles most)
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
# Normalization to your schema (drop unknowns, bridge haemolysis)
# ──────────────────────────────────────────────────────────────────────────────
def normalize_to_schema(parsed: Dict[str, str], db_fields: List[str]) -> Dict[str, str]:
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)
    out: Dict[str, str] = {}

    for k, v in (parsed or {}).items():
        kk = k.strip()
        key_l = kk.lower()
        target = None

        if kk in fields:
            target = kk
        elif key_l in alias:
            target = alias[key_l]

        if target in fields and v not in (None, "", "Unknown"):
            out[target] = v

    # Haemolysis Type → Haemolysis: Positive
    ht = alias.get("haemolysis type")
    h = alias.get("haemolysis")
    if ht in out and out[ht] not in ("", "Unknown", None) and h in fields:
        out[h] = "Positive"

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

    # Step 1: (optional) LLM parse — we still try, but regex will enrich heavily
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

    # Step 3: Normalize to your schema & suppress non-existent fields
    normalized = normalize_to_schema(merged, db_fields)
    return normalized
