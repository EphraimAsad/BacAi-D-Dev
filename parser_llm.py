# parser_llm.py — Dynamic Schema Version (Step 1.6)
import os
import json
import re
from typing import Dict, List

from parser_basic import parse_input_free_text as fallback_parser


def build_dynamic_schema(db_fields: List[str]) -> str:
    """
    Build a JSON schema string dynamically from database columns.
    Each field defaults to 'Unknown' or empty string for the LLM.
    """
    schema_lines = []
    for field in db_fields:
        clean_field = field.strip()
        if not clean_field or clean_field.lower() == "genus":
            continue

        # Suggest general value patterns
        if any(k in clean_field.lower() for k in ["fermentation", "oxidase", "catalase", "coagulase", "urease", "lipase"]):
            schema_lines.append(f'  "{clean_field}": "Positive|Negative|Variable|Unknown"')
        elif any(k in clean_field.lower() for k in ["growth temperature", "temperature"]):
            schema_lines.append(f'  "{clean_field}": "Numeric (°C) or Unknown"')
        elif any(k in clean_field.lower() for k in ["gram", "shape", "morphology"]):
            schema_lines.append(f'  "{clean_field}": "Descriptive text or Unknown"')
        else:
            schema_lines.append(f'  "{clean_field}": "Descriptive or categorical value or Unknown"')

    schema_text = "{\n" + ",\n".join(schema_lines) + "\n}"
    return schema_text


def parse_input_free_text(user_text: str, prior_facts: Dict | None = None, db_fields: List[str] | None = None) -> Dict:
    """
    Parse free-text input using GPT or Local model.
    Falls back to regex parser if LLM fails.
    """
    if not user_text.strip():
        return {}

    # Build schema from database fields (dynamic)
    schema_text = build_dynamic_schema(db_fields or [])

    try:
        if os.getenv("BACTAI_MODEL", "local").lower() == "gpt":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            prompt = build_prompt(user_text, schema_text, prior_facts)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=prompt,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            parsed = json.loads(resp.choices[0].message.content)
            return parsed

        else:
            import ollama
            prompt = build_prompt_text(user_text, schema_text, prior_facts)
            out = ollama.chat(model=os.getenv("LOCAL_MODEL", "llama3"), messages=[{"role": "user", "content": prompt}])
            text = out["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            parsed = json.loads(m.group(0)) if m else {}
            return parsed

    except Exception as e:
        print("⚠️ LLM parser failed, falling back to regex parser:", e)
        return fallback_parser(user_text, prior_facts)


def build_prompt(user_text: str, schema_text: str, prior_facts=None):
    """Build structured JSON prompt for GPT parser."""
    prior = json.dumps(prior_facts or {}, indent=2)
    system = (
        "You are a microbiology parser. Extract laboratory observations (biochemical, morphological, or growth tests) "
        "from the user's input. Use the following schema and return strictly valid JSON. "
        "If a field is not mentioned, set its value to 'Unknown'. "
        "Infer reasonable matches from wording (e.g. 'ferments rhamnose' = 'Rhamnose Fermentation: Positive')."
    )
    schema_instruction = f"\nSchema:\n{schema_text}\n"
    msg = [
        {"role": "system", "content": system + schema_instruction},
        {"role": "user", "content": f"Previous facts: {prior}\nUser observation: {user_text}"}
    ]
    return msg


def build_prompt_text(user_text: str, schema_text: str, prior_facts=None):
    """Prompt variant for local Llama models."""
    prior = json.dumps(prior_facts or {}, indent=2)
    return (
        "Extract microbiology test results from this observation.\n"
        "Return JSON with all the fields below. "
        "If a field is not mentioned, set it to 'Unknown'. "
        "Infer logical equivalents (e.g., 'Rhamnose positive' = 'Rhamnose Fermentation: Positive').\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Previous facts: {prior}\nUser observation: {user_text}"
    )
