# parser_basic.py
import re

# Minimal schema defaults (only keys your engine expects; add more as needed)
DEFAULT_FIELDS = {
    "Gram Stain": "Unknown",
    "Shape": "Unknown",
    "Oxidase": "Unknown",
    "Catalase": "Unknown",
    "Motility": "Unknown",
    "Colony Morphology": "Unknown",
    "Media Grown On": "Unknown",
    "Capsule": "Unknown",
    "Spore Formation": "Unknown",
    "Oxygen Requirement": "Unknown",
    "Growth Temperature": "",
    # sugars & others will remain Unknown unless mentioned
}

_POS_PAT = r"(positive|\+)"
_NEG_PAT = r"(negative|-)"

def _has(t, pat):
    return re.search(pat, t) is not None

def parse_input_free_text(text: str, prior_facts: dict | None = None) -> dict:
    """
    Turn a free text description into the dict your engine expects.
    Later we can replace this with an LLM call; keep the same signature.
    """
    t = (text or "").lower()
    facts = DEFAULT_FIELDS.copy()
    if prior_facts:
        facts.update(prior_facts)

    # Gram
    if "gram-" in t or "gram negative" in t:
        facts["Gram Stain"] = "Negative"
    if "gram+" in t or "gram positive" in t:
        facts["Gram Stain"] = "Positive"

    # Shape
    if re.search(r"\bcocci?\b", t):
        facts["Shape"] = "Cocci"
    if re.search(r"\brods?\b|\brot\b", t):
        facts["Shape"] = "Rods"
    if "curved" in t:
        facts["Shape"] = "Curved"
    if "short rod" in t:
        facts["Shape"] = "Short Rods"

    # Oxidase / Catalase
    if "oxidase" in t:
        facts["Oxidase"] = "Positive" if _has(t, r"oxidase\s+" + _POS_PAT) else ("Negative" if _has(t, r"oxidase\s+" + _NEG_PAT) else facts["Oxidase"])
    if "catalase" in t:
        facts["Catalase"] = "Positive" if _has(t, r"catalase\s+" + _POS_PAT) else ("Negative" if _has(t, r"catalase\s+" + _NEG_PAT) else facts["Catalase"])

    # Motility
    if "non-motile" in t or "non motile" in t:
        facts["Motility"] = "Non-motile"
    elif "motile" in t:
        facts["Motility"] = "Motile"

    # Capsule / Spore
    if "capsule positive" in t: facts["Capsule"] = "Positive"
    if "capsule negative" in t: facts["Capsule"] = "Negative"
    if "spore forming" in t: facts["Spore Formation"] = "Positive"
    if "non-spore" in t or "non spore" in t: facts["Spore Formation"] = "Negative"

    # Media
    if "macconkey" in t: facts["Media Grown On"] = "MacConkey Agar"
    if "cetrimide" in t: facts["Media Grown On"] = "Cetrimide Agar"

    # Colony morphology (very light heuristics)
    cm = []
    if "non-lactose" in t or "non lactose" in t or "lactose -" in t:
        cm.append("Non-lactose fermenter")
    if "small" in t: cm.append("Small")
    if "yellow" in t: cm.append("Yellow")
    if "mucoid" in t: cm.append("Mucoid")
    if cm:
        facts["Colony Morphology"] = "; ".join(sorted(set(cm)))

    # Oxygen
    if "aerobe" in t: facts["Oxygen Requirement"] = "Aerobe"
    if "anaerobe" in t: facts["Oxygen Requirement"] = "Anaerobe"
    if "facultative" in t: facts["Oxygen Requirement"] = "Facultative Anaerobe"
    if "microaerophil" in t: facts["Oxygen Requirement"] = "Microaerophile"

    # Growth Temperature (single number e.g., "grows at 42")
    m = re.search(r"\b(\d{2})\s?°?c\b", t)
    if m:
        facts["Growth Temperature"] = m.group(1)

    # Leave all other fields as "Unknown" or "" (engine handles Unknown gracefully)
    return facts
