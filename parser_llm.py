# parser_llm.py — Step 1.6 d (Anchored Fermentation Mapping)
import os, json, re
from typing import Dict, List
from parser_basic import parse_input_free_text as fallback_parser


# -------------------------------------------------
# Schema categorization helpers
# -------------------------------------------------
def categorize_fields(db_fields: List[str]) -> Dict[str, List[str]]:
    """Group field names into categories for clearer prompting."""
    cats = {"Morphology": [], "Enzyme": [], "Fermentation": [], "Other": []}
    for f in db_fields:
        name = f.strip()
        lname = name.lower()
        if lname == "genus":
            continue
        if any(k in lname for k in ["gram", "shape", "morphology", "motility", "capsule", "spore"]):
            cats["Morphology"].append(name)
        elif any(k in lname for k in ["oxidase", "catalase", "urease", "coagulase", "lipase", "test"]):
            cats["Enzyme"].append(name)
        elif "fermentation" in lname or "utilization" in lname:
            cats["Fermentation"].append(name)
        else:
            cats["Other"].append(name)
    return cats


# -------------------------------------------------
# Parser entry-point
# -------------------------------------------------
def parse_input_free_text(user_text: str,
                          prior_facts: Dict | None = None,
                          db_fields: List[str] | None = None) -> Dict:
    if not user_text.strip():
        return {}

    db_fields = [f for f in (db_fields or []) if f and f.lower() != "genus"]
    cats = categorize_fields(db_fields)

    try:
        model_choice = os.getenv("BACTAI_MODEL", "local").lower()
        if model_choice == "gpt":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = build_prompt(user_text, cats, prior_facts)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=prompt,
                temperature=0.25,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)

        else:
            import ollama
            prompt = build_prompt_text(user_text, cats, prior_facts)
            out = ollama.chat(
                model=os.getenv("LOCAL_MODEL", "llama3"),
                messages=[{"role": "user", "content": prompt}],
            )
            text = out["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)) if m else {}

    except Exception as e:
        print("⚠️ LLM parser failed — using fallback:", e)
        return fallback_parser(user_text, prior_facts)


# -------------------------------------------------
# Prompt builders
# -------------------------------------------------
def build_prompt(user_text: str, cats: Dict[str, List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)

    morph = ", ".join(cats["Morphology"])
    enzyme = ", ".join(cats["Enzyme"])
    ferm = ", ".join(cats["Fermentation"])
    other = ", ".join(cats["Other"])

    system = (
        "You are a microbiology parser that extracts laboratory observations "
        "into structured test results. Infer meaning from natural language, abbreviations, "
        "and symbols (+ / - / variable). Follow these mappings:\n"
        "- 'ferments X', 'X positive', '+ for X' → 'X Fermentation': 'Positive'\n"
        "- 'does not ferment X', 'non-fermenter for X', 'X negative', '– for X' → 'X Fermentation': 'Negative'\n"
        "- 'motile' / 'non-motile' → 'Motility'\n"
        "- 'oxidase +', 'catalase –', etc. → Enzyme reactions.\n\n"
        "Return valid JSON with keys exactly matching the field names below. "
        "Values should be 'Positive', 'Negative', 'Variable', a short description, or 'Unknown' "
        "if unmentioned.\n\n"
        "Example output:\n"
        "{\n"
        '  "Gram Stain": "Negative",\n'
        '  "Oxidase": "Positive",\n'
        '  "Glucose Fermentation": "Positive",\n'
        '  "Lactose Fermentation": "Negative"\n'
        "}\n\n"
        f"Morphology fields:\n{morph}\n\n"
        f"Enzyme tests:\n{enzyme}\n\n"
        f"Fermentation / utilization tests:\n{ferm}\n\n"
        f"Other fields:\n{other}\n\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Previous facts:\n{prior}\nUser observation:\n{user_text}"},
    ]


def build_prompt_text(user_text: str, cats: Dict[str, List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)

    morph = ", ".join(cats["Morphology"])
    enzyme = ", ".join(cats["Enzyme"])
    ferm = ", ".join(cats["Fermentation"])
    other = ", ".join(cats["Other"])

    return (
        "Extract microbiology test results from this observation. Use the categories below:\n"
        "- Fermentation: interpret phrases like 'ferments X', 'utilizes X', '+ for X', '– for X'.\n"
        "- Enzyme: oxidase, catalase, urease, etc.\n"
        "- Morphology: Gram stain, shape, motility, colony traits.\n"
        "- Other: growth or tolerance.\n\n"
        "Return valid JSON, one key per field; unmentioned = 'Unknown'.\n\n"
        "Example output:\n"
        "{ 'Gram Stain': 'Negative', 'Oxidase': 'Positive', 'Glucose Fermentation': 'Positive' }\n\n"
        f"Morphology: {morph}\n"
        f"Enzyme: {enzyme}\n"
        f"Fermentation: {ferm}\n"
        f"Other: {other}\n\n"
        f"Previous facts:\n{prior}\nUser observation:\n{user_text}"
    )
