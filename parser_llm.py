# parser_llm.py — Step 1.6 c Alias-Aware Smart Schema
import os, json, re
from typing import Dict, List
from parser_basic import parse_input_free_text as fallback_parser


# -----------------------------------------------------
# Build alias-aware field list
# -----------------------------------------------------
def build_field_aliases(db_fields: List[str]) -> dict:
    """
    Create a mapping of {canonical_field: [aliases]}.
    e.g. 'Rhamnose Fermentation' → ['Rhamnose']
    """
    aliases = {}
    for f in db_fields:
        if not f or f.lower() == "genus":
            continue
        name = f.strip()
        short = re.sub(r"\s*(Fermentation|Test)$", "", name, flags=re.I)
        short = short.replace("(", "").replace(")", "").strip()
        alist = [short]
        if "fermentation" in name.lower():
            alist.append(short + " Fermentation")
        if "test" in name.lower():
            alist.append(short + " Test")
        aliases[name] = list(set(alist))
    return aliases


# -----------------------------------------------------
# Parser entry-point
# -----------------------------------------------------
def parse_input_free_text(user_text: str,
                          prior_facts: Dict | None = None,
                          db_fields: List[str] | None = None) -> Dict:
    if not user_text.strip():
        return {}

    db_fields = [f for f in (db_fields or []) if f and f.lower() != "genus"]
    alias_map = build_field_aliases(db_fields)
    alias_text = "; ".join(
        [f"{k}: {', '.join(v)}" for k, v in alias_map.items()]
    )

    try:
        model_choice = os.getenv("BACTAI_MODEL", "local").lower()
        if model_choice == "gpt":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = build_prompt(user_text, alias_text, prior_facts)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=prompt,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)

        else:
            import ollama
            prompt = build_prompt_text(user_text, alias_text, prior_facts)
            out = ollama.chat(
                model=os.getenv("LOCAL_MODEL", "llama3"),
                messages=[{"role": "user", "content": prompt}],
            )
            text = out["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)) if m else {}

    except Exception as e:
        print("⚠️ LLM parser failed – using fallback:", e)
        return fallback_parser(user_text, prior_facts)


# -----------------------------------------------------
# Prompt builders
# -----------------------------------------------------
def build_prompt(user_text: str, alias_text: str, prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    system = (
        "You are a microbiology parser that extracts biochemical, morphological, and growth test results.\n"
        "Use contextual understanding to map phrases like '+', '-', 'ferments X', or 'non-motile' "
        "to the correct test field.  Use the alias list below to match abbreviated or shortened names.\n\n"
        "Alias list (showing equivalent forms):\n"
        f"{alias_text}\n\n"
        "Return valid JSON where each key is a field name and the value is one of:\n"
        "  'Positive', 'Negative', 'Variable', descriptive text, or 'Unknown'.\n"
        "Unmentioned fields → 'Unknown'.\n\n"
        "Example output:\n"
        "{\n"
        '  "Gram Stain": "Negative",\n'
        '  "Oxidase": "Positive",\n'
        '  "Rhamnose Fermentation": "Positive",\n'
        '  "Lactose Fermentation": "Negative"\n'
        "}\n"
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Previous facts:\n{prior}\nUser observation:\n{user_text}",
        },
    ]


def build_prompt_text(user_text: str, alias_text: str, prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    return (
        "Extract microbiology test results from this observation.\n"
        "Match each described reaction to the correct field using the alias list below.\n"
        "If a reaction or abbreviation (e.g. '+', '–', 'ferments X') appears, interpret accordingly.\n"
        "Return valid JSON; unmentioned fields → 'Unknown'.\n\n"
        f"Alias list:\n{alias_text}\n\n"
        "Example output:\n"
        "{ 'Oxidase': 'Positive', 'Rhamnose Fermentation': 'Positive', 'Lactose Fermentation': 'Negative' }\n\n"
        f"Previous facts:\n{prior}\nUser observation:\n{user_text}"
    )
