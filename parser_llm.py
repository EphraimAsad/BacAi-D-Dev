import os, json, re
from typing import Dict, List
from parser_basic import parse_input_free_text as fallback_parser

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

def parse_input_free_text(user_text:str,
                          prior_facts:Dict|None=None,
                          db_fields:List[str]|None=None)->Dict:
    if not user_text.strip():
        return {}

    db_fields = [f for f in (db_fields or []) if f.lower()!="genus"]
    cats = summarize_field_categories(db_fields)

    try:
        model_choice = os.getenv("BACTAI_MODEL","local").lower()
        if model_choice == "gpt":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            messages = build_prompt(user_text, cats, prior_facts)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                messages=messages,
                temperature=0.2,
                response_format={"type":"json_object"}
            )
            return json.loads(resp.choices[0].message.content)
        else:
            import ollama
            prompt = build_prompt_text(user_text, cats, prior_facts)
            out = ollama.chat(model=os.getenv("LOCAL_MODEL","llama3"),
                              messages=[{"role":"user","content":prompt}])
            m = re.search(r"\{.*\}", out["message"]["content"], re.S)
            return json.loads(m.group(0)) if m else {}
    except Exception as e:
        print("⚠️ Parser failed, fallback:", e)
        return fallback_parser(user_text, prior_facts)

def build_prompt(user_text:str, cats:Dict[str,List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    morph = ", ".join(cats["Morphology"][:10])
    enz = ", ".join(cats["Enzyme"][:10])
    ferm = ", ".join(cats["Fermentation"][:10])
    other = ", ".join(cats["Other"][:10])

    system = (
        "You convert free-text lab observations into structured results.\n"
        "Follow these examples exactly:\n"
        "• 'ferments glucose and sucrose' → {'Glucose Fermentation':'Positive','Sucrose Fermentation':'Positive'}\n"
        "• 'does not ferment lactose or rhamnose' → {'Lactose Fermentation':'Negative','Rhamnose Fermentation':'Negative'}\n"
        "• 'oxidase +' → {'Oxidase':'Positive'}\n"
        "• 'catalase –' → {'Catalase':'Negative'}\n"
        "• 'Gram-negative rod, non-motile' → {'Gram Stain':'Negative','Shape':'Rod','Motility':'Non-motile'}\n\n"
        "Return JSON; any field not mentioned → 'Unknown'.\n"
        "Relevant field names by category:\n"
        f"Morphology: {morph}\nEnzyme: {enz}\nFermentation: {ferm}\nOther: {other}"
    )
    return [
        {"role":"system","content":system},
        {"role":"user","content":f"Previous facts:\n{prior}\nObservation:\n{user_text}"}
    ]

def build_prompt_text(user_text:str, cats:Dict[str,List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    morph = ", ".join(cats["Morphology"][:10])
    enz = ", ".join(cats["Enzyme"][:10])
    ferm = ", ".join(cats["Fermentation"][:10])
    other = ", ".join(cats["Other"][:10])
    return (
        "Parse the following microbiology description into JSON results.\n"
        "Examples:\n"
        "  'ferments glucose' -> {'Glucose Fermentation':'Positive'}\n"
        "  'does not ferment lactose' -> {'Lactose Fermentation':'Negative'}\n"
        "  'oxidase +' -> {'Oxidase':'Positive'}\n"
        "  'Gram-positive cocci' -> {'Gram Stain':'Positive','Shape':'Cocci'}\n"
        "Return JSON; unmentioned fields='Unknown'.\n\n"
        f"Morphology: {morph}\nEnzyme: {enz}\nFermentation: {ferm}\nOther: {other}\n\n"
        f"Previous facts:\n{prior}\nObservation:\n{user_text}"
    )
