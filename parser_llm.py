# parser_llm.py — Step 1.6b (Smart Schema Prompt)
import os, json, re
from typing import Dict, List
from parser_basic import parse_input_free_text as fallback_parser


# --------------------------
# Dynamic schema helpers
# --------------------------
def build_field_list(db_fields: List[str]) -> str:
    """Return a short, comma-separated list of valid field names for the LLM prompt."""
    clean = [f.strip() for f in db_fields if f and f.lower() != "genus"]
    return ", ".join(sorted(clean))


# --------------------------
# Main parser
# --------------------------
def parse_input_free_text(user_text: str,
                          prior_facts: Dict | None = None,
                          db_fields: List[str] | None = None) -> Dict:
    """
    Parse free-text microbiology input using GPT or local Llama.
    Falls back to regex parser if the LLM fails.
    """
    if not user_text.strip():
        return {}

    field_list = build_field_list(db_fields or [])

    try:
        model_choice = os.getenv("BACTAI_MODEL", "local").lower()
        if model_choice == "gpt":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = build_prompt(user_text, field_list, prior_facts)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=prompt,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)

        else:  # local (Ollama / Llama)
            import ollama
            prompt = build_prompt_text(user_text, field_list, prior_facts)
            out = ollama.chat(
                model=os.getenv("LOCAL_MODEL", "llama3"),
                messages=[{"role": "user", "content": prompt}],
            )
            text = out["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)) if m else {}

    except Exception as e:
        print("⚠️ LLM parser failed, using fallback:", e)
        return fallback_parser(user_text, prior_facts)


# --------------------------
# Prompt builders
# --------------------------
def build_prompt(user_text: str, field_list: str, prior_facts=None):
    """
    Structured prompt for GPT models.
    """
    prior = json.dumps(prior_facts or {}, indent=2)
    system = (
        "You are a microbiology parser that extracts biochemical, morphological, and growth test results.\n"
        "You will receive user observations describing an organism.\n"
        "Map any described reactions or traits to the closest matching field name from the list below.\n"
        "Examples of mapping:\n"
        "  - 'Rhamnose positive' → 'Rhamnose Fermentation': 'Positive'\n"
        "  - 'ferments sucrose' → 'Sucrose Fermentation': 'Positive'\n"
        "  - 'does not ferment lactose' → 'Lactose Fermentation': 'Negative'\n"
        "  - 'grows in 6.5% NaCl' → 'NaCl Tolerance': 'Positive'\n"
        "  - 'oxidase +' → 'Oxidase': 'Positive'\n\n"
        "Return valid JSON where each key is a field name and the value is one of:\n"
        "  'Positive', 'Negative', 'Variable', a descriptive string, or 'Unknown'.\n"
        "If a field is not mentioned, set it to 'Unknown'."
    )
    list_section = f"\n\nValid field names:\n{field_list}\n\n"
    msg = [
        {"role": "system", "content": system + list_section},
        {
            "role": "user",
            "content": f"Previous facts: {prior}\nUser observation: {user_text}",
        },
    ]
    return msg


def build_prompt_text(user_text: str, field_list: str, prior_facts=None):
    """
    Equivalent prompt for local Llama models.
    """
    prior = json.dumps(prior_facts or {}, indent=2)
    return (
        "Extract microbiology test results from this observation.\n"
        "Match each described reaction to the closest field name from the list below.\n"
        "If the user mentions a test or abbreviation (e.g. '+', '-', 'ferments X'), map it appropriately.\n"
        "Return strictly valid JSON with one key per field.\n"
        "Unmentioned fields → 'Unknown'.\n\n"
        f"Valid field names:\n{field_list}\n\n"
        f"Previous facts: {prior}\nUser observation: {user_text}"
    )
