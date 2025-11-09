import os, json, re
from typing import Dict, List
from parser_basic import parse_input_free_text as fallback_parser


# -------------------------------------------------
# Utility: categorize fields
# -------------------------------------------------
def summarize_field_categories(db_fields: List[str]) -> Dict[str, List[str]]:
    cats = {"Morphology": [], "Enzyme": [], "Fermentation": [], "Other": []}
    for f in db_fields:
        n = f.strip()
        l = n.lower()
        if l == "genus":
            continue
        if any(k in l for k in ["gram","shape","morphology","motility","capsule","spore"]):
            cats["Morphology"].append(n)
        elif any(k in l for k in ["oxidase","catalase","urease","coagulase","lipase","test"]):
            cats["Enzyme"].append(n)
        elif "fermentation" in l or "utilization" in l:
            cats["Fermentation"].append(n)
        else:
            cats["Other"].append(n)
    return cats


# -------------------------------------------------
# Deterministic fermentation pattern extractor
# -------------------------------------------------
def extract_fermentations_regex(text: str, fermentation_fields: List[str]) -> Dict[str, str]:
    """
    Find patterns like 'ferments glucose', 'does not ferment lactose', etc.,
    and map them to known fermentation fields.
    """
    result = {}
    t = text.lower()

    # Positive patterns
    pos = re.findall(r"(?:ferments|utilizes)\s+([a-z0-9\-]+)", t)
    # Negative patterns
    neg = re.findall(r"(?:does\s+not\s+ferment|non\-fermenter\s+for|unable\s+to\s+ferment|cannot\s+ferment|does\s+not\s+utilize)\s+([a-z0-9\-]+)", t)

    for field in fermentation_fields:
        base = field.lower().replace(" fermentation","").replace(" utilization","").strip()
        if any(base in p for p in pos):
            result[field] = "Positive"
        elif any(base in n for n in neg):
            result[field] = "Negative"

    return result


# -------------------------------------------------
# Main parser
# -------------------------------------------------
def parse_input_free_text(user_text: str,
                          prior_facts: Dict | None = None,
                          db_fields: List[str] | None = None) -> Dict:
    if not user_text.strip():
        return {}

    db_fields = [f for f in (db_fields or []) if f.lower() != "genus"]
    cats = summarize_field_categories(db_fields)

    # ---- 1) run LLM parser (fallback on basic)
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
                response_format={"type": "json_object"}
            )
            parsed = json.loads(resp.choices[0].message.content)
        else:
            import ollama
            prompt = build_prompt_text(user_text, cats, prior_facts)
            out = ollama.chat(model=os.getenv("LOCAL_MODEL", "llama3"),
                              messages=[{"role": "user", "content": prompt}])
            m = re.search(r"\{.*\}", out["message"]["content"], re.S)
            parsed = json.loads(m.group(0)) if m else {}
    except Exception as e:
        print("⚠️ LLM parser failed:", e)
        parsed = fallback_parser(user_text, prior_facts)

    # ---- 2) add regex-based fermentation mapping
    regex_hits = extract_fermentations_regex(user_text, cats["Fermentation"])
    parsed.update(regex_hits)

    return parsed


# -------------------------------------------------
# Prompt builders (simplified for short context)
# -------------------------------------------------
def build_prompt(user_text: str, cats: Dict[str, List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    morph = ", ".join(cats["Morphology"][:10])
    enz = ", ".join(cats["Enzyme"][:10])
    ferm = ", ".join(cats["Fermentation"][:10])
    other = ", ".join(cats["Other"][:10])
    system = (
        "You parse microbiology observations into structured results.\n"
        "Handle morphology, enzyme, and growth traits. Leave carbohydrate fermentations to regex rules.\n"
        "Return JSON; unmentioned fields='Unknown'.\n"
        f"Morphology: {morph}\nEnzyme: {enz}\nFermentation examples: {ferm}\nOther: {other}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Previous facts:\n{prior}\nObservation:\n{user_text}"}
    ]


def build_prompt_text(user_text: str, cats: Dict[str, List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    return (
        "Extract morphology, enzyme and other test results from this observation. "
        "Fermentation reactions will be detected automatically.\n"
        "Return JSON; unmentioned fields='Unknown'.\n\n"
        f"Previous facts:\n{prior}\nObservation:\n{user_text}"
    )
