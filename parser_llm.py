# parser_llm.py
import os, json, re
from typing import Dict
from parser_basic import parse_input_free_text as fallback_parser

# Choose model via sidebar toggle; the app will set these
USE_MODEL = os.environ.get("BACTAI_MODEL", "local")   # "gpt" or "local"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
LOCAL_MODEL  = os.environ.get("LOCAL_MODEL", "llama3")

def parse_input_free_text(user_text: str, prior_facts: Dict | None = None) -> Dict:
    """
    Try LLM-based parsing first; fall back to regex parser if it fails.
    """
    if not user_text.strip():
        return {}

    try:
        if USE_MODEL.lower() == "gpt":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = build_prompt(user_text, prior_facts)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=prompt,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            parsed = json.loads(resp.choices[0].message.content)
            return parsed

        else:  # local via Ollama
            import ollama
            prompt = build_prompt_text(user_text, prior_facts)
            out = ollama.chat(model=LOCAL_MODEL, messages=[{"role":"user","content":prompt}])
            text = out["message"]["content"]
            # try to extract JSON
            m = re.search(r"\{.*\}", text, re.S)
            parsed = json.loads(m.group(0)) if m else {}
            return parsed

    except Exception as e:
        print("⚠️ LLM parser failed, using fallback:", e)
        return fallback_parser(user_text, prior_facts)


def build_prompt(user_text, prior_facts=None):
    """GPT structured prompt for reliable JSON output."""
    prior = json.dumps(prior_facts or {}, indent=2)
    schema = """
Return JSON with these exact keys and values:
{
 "Gram Stain": "Positive|Negative|Unknown",
 "Shape": "Cocci|Rods|Curved|Short Rods|Variable|Unknown",
 "Oxidase": "Positive|Negative|Unknown",
 "Catalase": "Positive|Negative|Unknown",
 "Motility": "Motile|Non-motile|Unknown",
 "Colony Morphology": "text",
 "Media Grown On": "text",
 "Capsule": "Positive|Negative|Unknown",
 "Spore Formation": "Positive|Negative|Unknown",
 "Oxygen Requirement": "Aerobe|Anaerobe|Facultative Anaerobe|Microaerophile|Unknown",
 "Growth Temperature": "numeric or empty string"
}
"""
    system = (
        "You are a microbiology parser. Extract only test observations mentioned by the user. "
        "Use 'Unknown' if not stated. Return strictly valid JSON, no commentary."
    )
    msg = [
        {"role":"system","content":system + schema},
        {"role":"user","content":f"Previous facts: {prior}\n\nUser observation: {user_text}"}
    ]
    return msg


def build_prompt_text(user_text, prior_facts=None):
    """Plain text prompt for local model (Ollama)."""
    prior = json.dumps(prior_facts or {}, indent=2)
    return (
        "Extract microbiology test results from this observation.\n"
        "Output only JSON with keys and allowed values listed below.\n\n"
        f"{build_prompt(user_text, prior_facts)[0]['content']}"
        f"\nPrevious facts: {prior}\nUser observation: {user_text}"
    )
