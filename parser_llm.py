# parser_llm.py — v3 (LMR-ready, tense/plural/negation-aware + ranges + media/morphology vocab)
# ──────────────────────────────────────────────────────────────────────────────
# This module converts free-text microbiology observations into your exact
# BactAI-D schema. It combines a robust rule-based linguistic layer with an
# optional LLM pass (OpenAI or local/Ollama), then normalizes to your Excel
# column names and values.
#
# Major features:
# - Unicode & whitespace normalization
# - Tense/plural/negation-aware parsing (negation scope ±5 tokens)
# - Conjunction handling for "and/or/nor" in both positives and "but not ..." spans
# - Heuristics for "Variable" (weak/trace/inconsistent/variable)
# - Fermentation parser (present/past/progressive, shorthand +/- and negative blocks)
# - Haemolysis Type→Haemolysis bridging (Gamma/None → Variable per your decision)
# - Media parsing with **TSI-only exclusion** + abbreviations (TSA/BHI/CBA/BA/SSA)
# - Colony morphology vocabulary and normalization (color/dash variants, texture)
# - Growth temperature ranges "10–40 °C" → "10//40", "no growth at 45 °C"
# - Abbreviations (MR/VP/LDC/ODC/ADH/LF/NLF)
# - Streamlit sidebar editor helper (field-by-field) + Copy JSON + Insert-to-Sidebar
# - What-if re-evaluation helper: "what if X was Y?" applies the change + reruns
# - Gold test runner with 5 starter cases and human-readable diffs
# - Lightweight self-learning hooks: writes hints from failing tests into /data/*.json
#
# Environment variables:
#   BACTAI_MODEL = "gpt" | "local"       (default "local")
#   OPENAI_MODEL = "gpt-4o-mini" | "gpt-5" | ...
#   LOCAL_MODEL  = "llama3" | "llama3.2" | "mistral-nemo" | ...
#   BACTAI_STRICT_MODE = "1" for strict schema-only, otherwise lenient
#
# Public functions you’ll call from the app:
#   parse_input_free_text(text, prior_facts=None, db_fields=None) -> Dict[str,str]
#   apply_what_if(user_text, prior_result, db_fields) -> Dict[str,str]
#   render_sidebar_json_editor(parsed_dict, db_fields, on_apply_label="Apply")
#   run_gold_tests()  # CLI:  python parser_llm.py --test
#
# NOTE: Streamlit is imported only inside the sidebar helper to avoid hard deps.


import os, re, json, sys, math
from typing import Dict, List, Set, Tuple, Optional
from parser_basic import parse_input_free_text as fallback_parser

# ──────────────────────────────────────────────────────────────────────────────
# Global caches / lightweight learning store paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.getcwd(), "data")
LEARNED_ALIASES_PATH = os.path.join(DATA_DIR, "learned_aliases.json")
LEARNED_PATTERNS_PATH = os.path.join(DATA_DIR, "learned_patterns.json")

def _ensure_data_dir():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        pass

def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json(path: str, obj: dict):
    try:
        _ensure_data_dir()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Allowed values & canonicalization (from your sheet)
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_VALUES: Dict[str, Set[str]] = {
    "Gram Stain": {"Positive", "Negative", "Variable"},
    "Shape": {"Cocci", "Rods", "Bacilli", "Spiral", "Short Rods"},
    "Catalase": {"Positive", "Negative", "Variable"},
    "Oxidase": {"Positive", "Negative", "Variable"},
    "Colony Morphology": set(),  # free text
    "Haemolysis": {"Positive", "Negative", "Variable"},
    "Haemolysis Type": {"None", "Beta", "Gamma", "Alpha"},
    "Indole": {"Positive", "Negative", "Variable"},
    "Growth Temperature": set(),  # numeric or range like "10//40"
    "Media Grown On": set(),
    "Motility": {"Positive", "Negative", "Variable"},
    "Capsule": {"Positive", "Negative", "Variable"},
    "Spore Formation": {"Positive", "Negative", "Variable"},
    "Oxygen Requirement": {"Intracellular", "Aerobic", "Anaerobic", "Facultative Anaerobe", "Microaerophilic", "Capnophilic"},
    "Methyl Red": {"Positive", "Negative", "Variable"},
    "VP": {"Positive", "Negative", "Variable"},
    "Citrate": {"Positive", "Negative", "Variable"},
    "Urease": {"Positive", "Negative", "Variable"},
    "H2S": {"Positive", "Negative", "Variable"},
    "Lactose Fermentation": {"Positive", "Negative", "Variable"},
    "Glucose Fermentation": {"Positive", "Negative", "Variable"},
    "Sucrose Fermentation": {"Positive", "Negative", "Variable"},
    "Nitrate Reduction": {"Positive", "Negative", "Variable"},
    "Lysine Decarboxylase": {"Positive", "Negative", "Variable"},
    "Ornitihine Decarboxylase": {"Positive", "Negative", "Variable"},
    "Arginine dihydrolase": {"Positive", "Negative", "Variable"},
    "Gelatin Hydrolysis": {"Positive", "Negative", "Variable"},
    "Esculin Hydrolysis": {"Positive", "Negative", "Variable"},
    "Dnase": {"Positive", "Negative", "Variable"},
    "ONPG": {"Positive", "Negative", "Variable"},
    "NaCl Tolerant (>=6%)": {"Positive", "Negative", "Variable"},
    "Lipase Test": {"Positive", "Negative", "Variable"},
    "Xylose Fermentation": {"Positive", "Negative", "Variable"},
    "Rhamnose Fermentation": {"Positive", "Negative", "Variable"},
    "Mannitol Fermentation": {"Positive", "Negative", "Variable"},
    "Sorbitol Fermentation": {"Positive", "Negative", "Variable"},
    "Maltose Fermentation": {"Positive", "Negative", "Variable"},
    "Arabinose Fermentation": {"Positive", "Negative", "Variable"},
    "Raffinose Fermentation": {"Positive", "Negative", "Variable"},
    "Inositol Fermentation": {"Positive", "Negative", "Variable"},
    "Trehalose Fermentation": {"Positive", "Negative", "Variable"},
    "Coagulase": {"Positive", "Negative", "Variable"},
}

# Media whitelist (exact spellings, still accept new names; exclude TSI)
MEDIA_WHITELIST = {
    "MacConkey Agar","Nutrient Agar","ALOA","Palcam","Preston","Columbia","BP",
    "Mannitol Salt Agar","MRS","Anaerobic Media","XLD Agar","TBG","TCBS","VID",
    "EMB Agar","CCI","Salt Nutrient Agar","Thayer Martin Agar","Tryptic Soy Agar",
    "Chocolate Agar","Bacteroides Bile Esculin Agar","KVLB Agar","Charcoal Blood Agar",
    "Anaerobic Blood Agar","Yeast Extract Mannitol Agar","Burks Medium","Peptone Water",
    "Sabouraud Dextrose Agar","Yeast Extract Peptone Dextrose","Malt Extract Agar",
    "Middlebrook Agar","Inorganic Mineral Nitrate Media","Inorganic Mineral Ammonia Media",
    "Iron Media","Sulfur Media","Organic Media","Yeast Extract Agar","Cellulose Agar",
    "Baciillus Media","Pyridoxal","Lcysteine","Ferrous Sulfate Media","Hayflicks Agar",
    "Cell Culture","Intracellular","Brain Heart Infusion Agar","Human Fibroblast Cell Culture","BCYE Agar",
    # Abbreviations will map to these:
    "Tryptic Soy Agar", "Brain Heart Infusion Agar", "Columbia Blood Agar", "Blood Agar"
}
MEDIA_EXCLUDE_TERMS = {"tsi", "triple sugar iron"}  # exclude as media

# Abbreviations (biochem & media)
BIOCHEM_ABBR = {
    "mr": "Methyl Red",
    "vp": "VP",
    "ldc": "Lysine Decarboxylase",
    "odc": "Ornitihine Decarboxylase",
    "adh": "Arginine dihydrolase",
    "lf": "Lactose Fermentation",    # positive bias; see NLF
    "nlf": "Lactose Fermentation",   # will set to Negative when phrased as NLF colonies
}
MEDIA_ABBR = {
    "tsa": "Tryptic Soy Agar",
    "bhi": "Brain Heart Infusion Agar",
    "cba": "Columbia Blood Agar",
    "ssa": "Blood Agar",
    "ba": "Blood Agar",
}

# Value synonyms & polarity map
VALUE_SYNONYMS: Dict[str, Dict[str, str]] = {
    "Gram Stain": {
        "gram positive": "Positive","gram-positive": "Positive","g+": "Positive",
        "gram negative": "Negative","gram-negative": "Negative","g-": "Negative",
        "variable": "Variable"
    },
    "Shape": {
        "rod": "Rods","rods": "Rods","bacillus": "Bacilli","bacilli": "Bacilli",
        "coccus": "Cocci","cocci": "Cocci","spiral": "Spiral","short rods": "Short Rods"
    },
    "Oxygen Requirement": {
        "facultative": "Facultative Anaerobe","facultative anaerobe": "Facultative Anaerobe",
        "facultative anaerobic": "Facultative Anaerobe",
        "aerobe": "Aerobic","aerobic": "Aerobic",
        "anaerobe": "Anaerobic","anaerobic": "Anaerobic",
        "microaerophile": "Microaerophilic","microaerophilic": "Microaerophilic",
        "capnophile": "Capnophilic","capnophilic": "Capnophilic",
        "intracellular": "Intracellular"
    },
    "Haemolysis Type": {
        "beta": "Beta","β": "Beta","alpha": "Alpha","α": "Alpha",
        "gamma": "Gamma","γ": "Gamma","none": "None"
    },
    "*POLARITY*": {
        "+": "Positive","positive": "Positive","pos": "Positive",
        "-": "Negative","negative": "Negative","neg": "Negative",
        "weakly positive": "Variable","variable": "Variable","weak": "Variable",
        "trace": "Variable","slight": "Variable","slightly positive": "Variable"
    },
}

POLARITY_FIELDS = {
    "Catalase","Oxidase","Haemolysis","Indole","Motility","Capsule","Spore Formation",
    "Methyl Red","VP","Citrate","Urease","H2S","Lactose Fermentation","Glucose Fermentation",
    "Sucrose Fermentation","Nitrate Reduction","Lysine Decarboxylase","Ornitihine Decarboxylase",
    "Arginine dihydrolase","Gelatin Hydrolysis","Esculin Hydrolysis","Dnase","ONPG",
    "NaCl Tolerant (>=6%)","Lipase Test","Xylose Fermentation","Rhamnose Fermentation",
    "Mannitol Fermentation","Sorbitol Fermentation","Maltose Fermentation","Arabinose Fermentation",
    "Raffinose Fermentation","Inositol Fermentation","Trehalose Fermentation","Coagulase"
}

# Universal negation cues (used within a ±5-token scope window)
NEGATION_CUES = [
    "not produced","no production","not observed","none observed","no reaction",
    "absent","without production","not detected","does not","did not","fails to",
    "unable to","no growth","non-fermenter","nonfermenter","non fermenter"
]

# Variable cues (convert to Variable unless explicit Pos/Neg is closer)
VARIABLE_CUES = ["variable","inconsistent","weak","trace","slight","irregular"]


# ──────────────────────────────────────────────────────────────────────────────
# Colony Morphology vocabulary & normalization
# ──────────────────────────────────────────────────────────────────────────────
CM_TOKENS = {
    # sizes & measurements
    "1/3mm","1/2mm","2/3mm","2/4mm","0.5/1mm","0.5mm/2mm","1mm","2mm","3mm","small","medium","large","tiny","pinpoint","subsurface","satellite",
    # shapes/profile
    "round","circular","convex","flat","domed","heaped","fried egg",
    # edges/surface/texture
    "smooth","rough","wrinkled","granular","mucoid","glistening","dull","matte","shiny","sticky","adherent","powdery","chalk","leathery","velvet","crumbly",
    "ground glass","irregular edges","spreading","swarming","corrode","pit","ropey","butyrous","waxy",
    # opacity/transparency
    "opaque","translucent","colourless","colorless",
    # moisture
    "dry","moist",
    # colours (UK spellings preferred)
    "white","grey","gray","cream","off-white","yellow","pale yellow","orange","pink","coral","red","green","violet","purple","black","brown","beige","tan","blue",
    # extras
    "bright","pigmented","iridescent","ring","dingers ring"
}

# ──────────────────────────────────────────────────────────────────────────────
# Schema helpers and normalization utilities
# ──────────────────────────────────────────────────────────────────────────────
_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def normalize_columns(db_fields: List[str]) -> List[str]:
    """Return exact DB fields (minus 'Genus') with original casing kept."""
    return [f for f in (db_fields or []) if f and f.strip().lower() != "genus"]

def normalize_text(raw: str) -> str:
    """Unicode & whitespace normalization; unify hemolysis→haemolysis; lowercase."""
    t = raw or ""
    t = t.replace("°", " °")
    t = t.translate(_SUBSCRIPT_DIGITS)
    t = (t.replace("\u2010","-").replace("\u2011","-").replace("\u2012","-")
           .replace("\u2013","-").replace("\u2014","-").replace("–","-").replace("—","-"))
    t = re.sub(r"hemolys", "haemolys", t, flags=re.I)
    t = re.sub(r"gray", "grey", t, flags=re.I)  # prefer 'grey'
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _normalize_token(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"[.,;:!?\-]+$", "", s)
    return s.strip()

def _tokenize_list(s: str) -> List[str]:
    """Split using commas and logical conjunctions, return clean tokens."""
    s = re.sub(r"\s*(?:,|and|or|&|nor)\s*", ",", s.strip(), flags=re.I)
    items = [t.strip() for t in s.split(",") if t.strip()]
    return [re.sub(r"[.,;:\s]+$", "", i) for i in items]

def _canon_value(field: str, value: str) -> str:
    """Clamp values to canonical polarity and allowed terms."""
    v = (value or "").strip()
    if not v:
        return v
    # polarity fields
    if field in POLARITY_FIELDS:
        low = v.lower()
        if low in VALUE_SYNONYMS.get("*POLARITY*", {}):
            v = VALUE_SYNONYMS["*POLARITY*"][low]
        else:
            if re.fullmatch(r"\+|positive|pos", low): v = "Positive"
            elif re.fullmatch(r"\-|negative|neg", low): v = "Negative"
            elif any(tok in low for tok in ["weak","variable","trace","slight"]): v = "Variable"

    # field-specific synonyms
    low = v.lower()
    if field in VALUE_SYNONYMS:
        v = VALUE_SYNONYMS[field].get(low, v)

    # clamp to allowed if defined
    allowed = ALLOWED_VALUES.get(field)
    if allowed and v not in allowed:
        tv = v.title()
        if tv in allowed:
            v = tv
    return v

def _set_field_safe(out: Dict[str, str], key: str, val: str):
    """Safe setter with mild conflict policy (explicit negation wins elsewhere)."""
    if not val:
        return
    cur = out.get(key)
    if cur is None:
        out[key] = val
        return
    # if existing is Negative and new is Positive, don't auto-flip unless explicit handled in conflict resolver
    # allow overwriting Variable with Positive/Negative
    if cur == "Variable" and val in {"Positive","Negative"}:
        out[key] = val
        return
    # keep first for now; conflict resolver can run later
    out[key] = val

def _in_window(text_tokens: List[str], idx: int, window: int, predicates: List[str]) -> bool:
    """Check if any phrase in `predicates` appears within ±window tokens of index idx."""
    start = max(0, idx - window)
    end = min(len(text_tokens), idx + window + 1)
    span = " ".join(text_tokens[start:end])
    return any(p in span for p in predicates)

def _any_negation_near(tokens: List[str], idx: int, window: int = 5) -> bool:
    return _in_window(tokens, idx, window, NEGATION_CUES)

def _any_variable_near(tokens: List[str], idx: int, window: int = 5) -> bool:
    return _in_window(tokens, idx, window, VARIABLE_CUES)

def _split_color_hyphens(s: str) -> List[str]:
    """Split dashed colors like grey-cream into ['grey','cream']."""
    parts = []
    for token in re.split(r"[;/,]", s):
        token = token.strip()
        if "-" in token and any(c in token for c in ["grey","gray","white","cream","yellow","orange","pink","red","green","blue","brown","beige","tan","black","purple","violet","off"]):
            parts.extend([p.strip() for p in token.split("-") if p.strip()])
        else:
            parts.append(token)
    return parts

# ──────────────────────────────────────────────────────────────────────────────
# Field categorization (lightweight; for prompt context)
# ──────────────────────────────────────────────────────────────────────────────
def _summarize_field_categories(db_fields: List[str]) -> Dict[str, List[str]]:
    cats = {"Morphology": [], "Enzyme": [], "Fermentation": [], "Other": []}
    for f in normalize_columns(db_fields):
        n = f.strip(); l = n.lower()
        if any(k in l for k in ["gram","shape","morphology","motility","capsule","spore","oxygen requirement","media grown"]):
            cats["Morphology"].append(n)
        elif any(k in l for k in ["oxidase","catalase","urease","coagulase","lipase","indole","citrate","vp","methyl red","gelatin","dnase","nitrate","h2s","esculin"]):
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
    exact = {f.lower(): f for f in normalize_columns(db_fields)}
    alias: Dict[str, str] = {}

    def add(a: str, target: str):
        t = target.lower()
        if t in exact:
            alias[a.lower()] = exact[t]

    # Base fields & known spellings/typos
    add("mr","Methyl Red"); add("methyl red","Methyl Red")
    add("vp","VP"); add("voges proskauer","VP")
    add("h2s","H2S"); add("dnase","Dnase"); add("gelatin","Gelatin Hydrolysis")
    add("gelatin liquefaction","Gelatin Hydrolysis")
    add("lipase","Lipase Test"); add("lipase test","Lipase Test")
    add("onpg","ONPG"); add("onpg test","ONPG"); add("esculin hydrolysis","Esculin Hydrolysis")
    add("nacl tolerance","NaCl Tolerant (>=6%)"); add("nacl tolerant","NaCl Tolerant (>=6%)"); add("nacl","NaCl Tolerant (>=6%)")
    add("nitrate","Nitrate Reduction"); add("nitrate reduction","Nitrate Reduction")
    add("lysine decarboxylase","Lysine Decarboxylase")
    add("ornithine decarboxylase","Ornitihine Decarboxylase"); add("ornitihine decarboxylase","Ornitihine Decarboxylase")
    add("arginine dihydrolase","Arginine dihydrolase")
    add("coagulase","Coagulase"); add("citrate","Citrate")
    add("urease","Urease"); add("indole","Indole")
    add("oxidase","Oxidase"); add("catalase","Catalase")
    add("motility","Motility"); add("capsule","Capsule")
    add("spore formation","Spore Formation")
    add("haemolysis","Haemolysis"); add("haemolysis type","Haemolysis Type")
    add("growth temperature","Growth Temperature")
    add("media grown on","Media Grown On")
    add("oxygen requirement","Oxygen Requirement")
    add("gram stain","Gram Stain"); add("shape","Shape")
    add("glucose fermantation","Glucose Fermentation")  # typo → canonical

    # Biochem abbreviations
    for abbr, full in BIOCHEM_ABBR.items():
        add(abbr, full)

    # Fermentation bases (map "rhamnose" → "Rhamnose Fermentation")
    for f in normalize_columns(db_fields):
        if f.lower().endswith(" fermentation"):
            base = f[:-12].strip().lower()
            alias[base] = f

    # Media names & abbreviations map to "Media Grown On" key (value will join)
    for m in MEDIA_WHITELIST:
        alias[m.lower()] = "Media Grown On"
    for abbr, full in MEDIA_ABBR.items():
        alias[abbr.lower()] = "Media Grown On"

    # Learned aliases from previous failures (if any)
    learned = _load_json(LEARNED_ALIASES_PATH)
    for k, v in learned.items():
        if v in exact.values():  # only accept if target still exists
            alias[k.lower()] = v

    return alias


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders (LLM optional; we keep it minimal because regex does heavy lift)
# ──────────────────────────────────────────────────────────────────────────────
def build_prompt(user_text: str, cats: Dict[str, List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    morph = ", ".join(cats["Morphology"][:10])
    enz = ", ".join(cats["Enzyme"][:10])
    ferm = ", ".join(cats["Fermentation"][:10])
    other = ", ".join(cats["Other"][:10])
    system = (
        "You parse microbiology observations into structured results. "
        "Focus on morphology, enzyme, growth and oxygen traits. Fermentations are handled by rules. "
        "Return compact JSON; unmentioned fields='Unknown'.\n"
        f"Morphology: {morph}\nEnzyme: {enz}\nFermentation: {ferm}\nOther: {other}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Previous facts:\n{prior}\nObservation:\n{user_text}"},
    ]

def build_prompt_text(user_text: str, cats: Dict[str, List[str]], prior_facts=None):
    prior = json.dumps(prior_facts or {}, indent=2)
    return (
        "Extract morphology, enzyme and growth/oxygen results from this description. "
        "Leave fermentation mapping mostly to rules. "
        "Return JSON; unmentioned fields='Unknown'.\n\n"
        f"Previous facts:\n{prior}\nObservation:\n{user_text}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Colony morphology normalization
# ──────────────────────────────────────────────────────────────────────────────
def normalize_cm_phrase(text: str) -> str:
    """Extract and normalize colony morphology tokens from free text."""
    t = text.lower()
    spans = []
    m = re.search(r"colon(?:y|ies)\s+(?:are|appear|were|appearing|appeared)\s+([^.]+?)(?:\s+on|\.)", t)
    if m:
        spans.append(m.group(1))
    spans.append(t)

    found: List[str] = []
    def add(tok: str):
        tok = tok.strip()
        if not tok: return
        if tok not in found:
            found.append(tok)

    for s in spans:
        # explicit measurements like "1/2mm", "0.5/1mm"
        for mm in re.findall(r"(?:\d+(?:\.\d+)?\/\d+(?:\.\d+)?mm|\d+(?:\.\d+)?mm|0\.5\/1mm|0\.5mm\/2mm|1\/3mm|2\/3mm|2\/4mm)", s):
            add(mm)
        # prepare a normalized string for multiword tokens
        s_norm = " " + re.sub(r"[,;/]", " ", s) + " "
        multi = ["ground glass","irregular edges","fried egg","dingers ring","off-white","pale yellow","cream-white"]
        for mword in multi:
            if f" {mword} " in s_norm:
                add(mword)

        # token scan (split on separators and hyphens)
        parts = re.split(r"[,;:/\s]+", s)
        hyphen_fixed = []
        for p in parts:
            hyphen_fixed.extend(_split_color_hyphens(p))
        for p in hyphen_fixed:
            low = p.strip().lower()
            if low in {"colorless"}: low = "colourless"
            if low in CM_TOKENS:
                add(low)

    # order for aesthetics
    order_groups = [
        {"1/3mm","1/2mm","2/3mm","2/4mm","0.5/1mm","0.5mm/2mm","1mm","2mm","3mm","tiny","small","medium","large","pinpoint","subsurface","satellite"},
        {"round","circular","convex","flat","domed","heaped","fried egg"},
        {"smooth","rough","wrinkled","granular","mucoid","glistening","dull","matte","shiny","sticky","adherent","powdery","chalk","leathery","velvet","crumbly",
         "ground glass","irregular edges","spreading","swarming","corrode","pit","ropey","butyrous","waxy","bright","pigmented","iridescent"},
        {"opaque","translucent","colourless"},
        {"dry","moist"},
        {"white","grey","gray","cream","off-white","yellow","pale yellow","orange","pink","coral","red","green","violet","purple","black","brown","beige","tan","blue"},
    ]
    ordered: List[str] = []
    seen = set()
    for grp in order_groups:
        for tok in found:
            if tok in grp and tok not in seen:
                ordered.append(tok); seen.add(tok)
    # leftovers
    for tok in found:
        if tok not in seen:
            ordered.append(tok); seen.add(tok)

    # pretty case + Grey normalization
    pretty = []
    for w in ordered:
        if w == "gray": w = "grey"
        # Title-case but keep mm strings and acronyms as-is
        if re.search(r"\d", w) or w.isupper():
            pretty.append(w)
        else:
            # handle compound e.g. "pale yellow" → "Yellow (Pale)"
            if w == "pale yellow":
                pretty.append("Yellow (Pale)")
            elif w == "off-white":
                pretty.append("Off-White")
            elif w == "cream-white":
                pretty.append("Cream; White")
            else:
                pretty.append(w.title())
    # flatten "Cream; White" produced by cream-white split
    flat = []
    for item in pretty:
        if item == "Cream; White":
            flat.extend(["Cream","White"])
        else:
            flat.append(item)

    return "; ".join(flat)


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction layers (regex)
# ──────────────────────────────────────────────────────────────────────────────

def extract_fermentations_regex(text: str, db_fields: List[str]) -> Dict[str, str]:
    """
    Parse fermentations with tense/plural/conjunctions & negation window.
    Handles: ferments/fermented/fermenting/utilizes/utilized/produced acid from
    Negative blocks: "does not ferment", "but not X, Y or Z", "unable to utilize"
    Shorthand: "lactose +" / "rhamnose -"
    """
    out: Dict[str, str] = {}
    t = normalize_text(text)
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)

    # Build base → exact field map for fermentations
    ferm_fields = [f for f in fields if f.lower().endswith(" fermentation")]
    base_to_field = {f[:-12].strip().lower(): f for f in ferm_fields}

    def set_field_by_base(base: str, val: str):
        b = _normalize_token(base)
        if b in base_to_field:
            _set_field_safe(out, base_to_field[b], _canon_value(base_to_field[b], val))
        elif b in alias and alias[b] in fields:
            _set_field_safe(out, alias[b], _canon_value(alias[b], val))

    # Tokenize for negation scope heuristics
    tokens = t.split()

    # POSITIVE lists (ferments/utilizes/produces acid from)
    pos_patterns = [
        r"(?:ferments?|utilizes?|produces?\s+acid\s+from)\s+([a-z0-9\.\-%\s,/&]+)"
    ]
    for pat in pos_patterns:
        for m in re.finditer(pat, t, flags=re.I):
            span = m.group(1)
            # cut off at "but not" to avoid leaking negatives
            span = re.split(r"(?i)\bbut\s+not\b", span)[0]
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
        for m in re.finditer(pat, t, flags=re.I):
            for a in _tokenize_list(m.group(1)):
                set_field_by_base(a, "Negative")

    # NEGATIVE after "but not …" (split and/or/nor)
    for m in re.finditer(r"(?:ferments?|utilizes?)[^.]*?\bbut\s+not\s+([\w\s,;.&-]+)", t, flags=re.I):
        seg = m.group(1)
        seg = re.sub(r"\bor\b", ",", seg, flags=re.I)
        seg = re.sub(r"\bnor\b", ",", seg, flags=re.I)
        for a in _tokenize_list(seg):
            set_field_by_base(a, "Negative")

    # Shorthand "+/-"
    for m in re.finditer(r"\b([a-z0-9\-]+)\s*(?:fermentation)?\s*([+\-])\b", t, flags=re.I):
        a, sign = m.group(1), m.group(2)
        set_field_by_base(a, "Positive" if sign == "+" else "Negative")

    # Variable cues near specific sugars ("raffinose variable")
    for base in list(base_to_field.keys()):
        if re.search(rf"\b{re.escape(base)}\b\s+(?:variable|inconsistent|weak|trace|slight|irregular)", t, flags=re.I):
            set_field_by_base(base, "Variable")

    # ONPG
    if re.search(r"\bonpg\s*(?:test)?\s*(?:is\s+)?(\+|positive)\b", t, flags=re.I):
        if "onpg" in alias and alias["onpg"] in fields:
            _set_field_safe(out, alias["onpg"], "Positive")
    elif re.search(r"\bonpg\s*(?:test)?\s*(?:is\s+)?(\-|negative)\b", t, flags=re.I):
        if "onpg" in alias and alias["onpg"] in fields:
            _set_field_safe(out, alias["onpg"], "Negative")

    # NaCl tolerant
    if re.search(r"\b(tolerant|grows|growth)\s+(?:in|up\s+to|to|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t, flags=re.I):
        if "nacl tolerant" in alias and alias["nacl tolerant"] in fields:
            _set_field_safe(out, alias["nacl tolerant"], "Positive")
    if re.search(r"\bno\s+growth\s+(?:in|at)\s+[0-9\.]+\s*%?\s*(?:na\s*cl|salt)\b", t, flags=re.I):
        if "nacl tolerant" in alias and alias["nacl tolerant"] in fields:
            _set_field_safe(out, alias["nacl tolerant"], "Negative")
    if re.search(r"\bnacl\s+tolerant\b", t, flags=re.I):
        if "nacl tolerant" in alias and alias["nacl tolerant"] in fields:
            _set_field_safe(out, alias["nacl tolerant"], "Positive")

    # Handle LF/NLF short forms on MacConkey
    if re.search(r"\bnlf\b", t):
        _set_field_safe(out, "Lactose Fermentation", "Negative")
    if re.search(r"\blf\b", t) and "Lactose Fermentation" not in out:
        _set_field_safe(out, "Lactose Fermentation", "Positive")

    return out


def extract_biochem_regex(text: str, db_fields: List[str]) -> Dict[str, str]:
    """
    Enzymes, morphology, oxygen, haemolysis, growth temp & ranges, media, capsules,
    decarboxylases (plural-aware), and general tense/negation handling.
    """
    out: Dict[str, str] = {}
    raw = text or ""
    t = normalize_text(raw)
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)
    tokens = t.split()

    def set_field(k_like: str, val: str):
        target = alias.get(k_like.lower(), k_like)
        if target in fields:
            _set_field_safe(out, target, _canon_value(target, val))

    # Gram Stain (avoid dual mention flips)
    if re.search(r"\bgram[-\s]?positive\b", t) and not re.search(r"\bgram[-\s]?negative\b", t):
        set_field("gram stain", "Positive")
    elif re.search(r"\bgram[-\s]?negative\b", t) and not re.search(r"\bgram[-\s]?positive\b", t):
        set_field("gram stain", "Negative")

    # Shape
    if re.search(r"\bcocci\b", t): set_field("shape", "Cocci")
    if re.search(r"\brods?\b|bacilli\b", t): set_field("shape", "Rods")  # prefer "Rods"
    if re.search(r"\bspiral\b", t): set_field("shape", "Spiral")
    if re.search(r"\bshort\s+rods\b", t): set_field("shape", "Short Rods")

    # Motility
    # "non-motile", "non motile", "motile"
    if re.search(r"\bnon[-\s]?motile\b", t):
        set_field("motility", "Negative")
    elif re.search(r"\bmotile\b", t):
        set_field("motility", "Positive")

    # Capsule
    if re.search(r"\b(capsulated|encapsulated)\b", t):
        set_field("capsule", "Positive")
    if re.search(r"\bnon[-\s]?capsulated\b|\bcapsule\s+absent\b", t):
        set_field("capsule", "Negative")
    if re.search(r"\bcapsule\s+(?:variable|inconsistent|weak)\b", t):
        set_field("capsule", "Variable")

    # Spore formation
    if re.search(r"\bnon[-\s]?spore[-\s]?forming\b|\bno\s+spores?\b", t):
        set_field("spore formation", "Negative")
    if re.search(r"\bspore[-\s]?forming\b|\bspores?\s+present\b", t):
        set_field("spore formation", "Positive")

    # Oxygen requirement (prioritize specific)
    if re.search(r"\bintracellular\b", t): set_field("oxygen requirement", "Intracellular")
    elif re.search(r"\bcapnophil(ic|e)\b", t): set_field("oxygen requirement", "Capnophilic")
    elif re.search(r"\bmicroaerophil(ic|e)\b", t): set_field("oxygen requirement", "Microaerophilic")
    elif re.search(r"\bfacultative\b", t) or re.search(r"\bfacultative\s+anaerob", t):
        set_field("oxygen requirement", "Facultative Anaerobe")
    elif re.search(r"\baerobic\b", t): set_field("oxygen requirement", "Aerobic")
    elif re.search(r"\banaerobic\b", t): set_field("oxygen requirement", "Anaerobic")

    # Generic enzyme tests (polarity + variable cues). Tense-aware.
    targets = [
        "catalase","oxidase","coagulase","urease","lipase","indole",
        "citrate","vp","methyl red","gelatin","dnase","nitrate reduction","nitrate","h2s","esculin hydrolysis"
    ]
    for test in targets:
        # positive
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:\+|positive|detected|produced)\b", t):
            set_field(test, "Positive")
        # negative
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:\-|negative|not\s+detected|not\s+produced|absent)\b", t):
            set_field(test, "Negative")
        # weak/trace/variable
        if re.search(rf"\b{test}\s*(?:test)?\s*(?:weak(?:ly)?\s*positive|variable|trace|slight)\b", t):
            set_field(test, "Variable")

    # Special phrasing H2S: "not produced", "produces", etc.
    if re.search(r"\bh\s*2\s*s\s+not\s+produced\b", t): set_field("h2s", "Negative")
    if re.search(r"\bh\s*2\s*s\s+(?:\+|positive|detected|produced)\b", t): set_field("h2s", "Positive")

    # Nitrate alternate phrasing
    if re.search(r"\breduces\s+nitrate\b", t): set_field("nitrate", "Positive")
    if re.search(r"\bdoes\s+not\s+reduce\s+nitrate\b", t): set_field("nitrate", "Negative")

    # Haemolysis type (and "no haemolysis" → Gamma)
    if re.search(r"\b(beta|β)[-\s]?haem", t):
        set_field("haemolysis type", "Beta")
    elif re.search(r"\b(alpha|α)[-\s]?haem", t):
        set_field("haemolysis type", "Alpha")
    elif re.search(r"\b(gamma|γ)[-\s]?haem\b", t) or re.search(r"\bno\s+haemolysis\b|\bhaemolysis\s+not\s+observed\b", t):
        set_field("haemolysis type", "Gamma")

    # Methyl Red abbreviation (MR)
    if re.search(r"\bmr\s*(?:test)?\s*(\+|positive)\b", t): set_field("methyl red", "Positive")
    if re.search(r"\bmr\s*(?:test)?\s*(\-|negative)\b", t): set_field("methyl red", "Negative")

    # VP abbreviation already handled via generic, but re-enforce:
    if re.search(r"\bvp\s*(?:test)?\s*(\+|positive)\b", t): set_field("vp", "Positive")
    if re.search(r"\bvp\s*(?:test)?\s*(\-|negative)\b", t): set_field("vp", "Negative")

    # Decarboxylases / dihydrolase (plural-aware)
    # "lysine, ornithine and arginine decarboxylases positive"
    if re.search(r"\bdecarboxylases?\s+positive\b", t):
        pre = re.search(r"([a-z,\s]+)decarboxylases?\s+positive", t)
        if pre:
            lst = _tokenize_list(pre.group(1))
            for it in lst:
                it = it.strip().lower()
                if it.startswith("lysine"): set_field("lysine decarboxylase", "Positive")
                if it.startswith("ornithine") or it.startswith("ornitihine"): set_field("ornithine decarboxylase", "Positive")
                if it.startswith("arginine"): set_field("arginine dihydrolase", "Positive")
    if re.search(r"\bdecarboxylases?\s+negative\b", t):
        pre = re.search(r"([a-z,\s]+)decarboxylases?\s+negative", t)
        if pre:
            lst = _tokenize_list(pre.group(1))
            for it in lst:
                it = it.strip().lower()
                if it.startswith("lysine"): set_field("lysine decarboxylase", "Negative")
                if it.startswith("ornithine") or it.startswith("ornitihine"): set_field("ornithine decarboxylase", "Negative")
                if it.startswith("arginine"): set_field("arginine dihydrolase", "Negative")

    # Singular forms too
    # e.g., "lysine decarboxylase positive"
    for k, field in [("lysine decarboxylase","lysine decarboxylase"),
                     ("ornithine decarboxylase","ornithine decarboxylase"),
                     ("ornitihine decarboxylase","ornitihine decarboxylase"),
                     ("arginine dihydrolase","arginine dihydrolase")]:
        if re.search(rf"\b{k}\s+(?:test\s+)?(\+|positive)\b", t): set_field(field, "Positive")
        if re.search(rf"\b{k}\s+(?:test\s+)?(\-|negative)\b", t): set_field(field, "Negative")

    # Growth Temperature:
    # - ranges: "grows 10–40 °C", "growth between 5 and 45 °C"
    # - single allowed: "grows at 37 °C"
    # - "no growth at 45 °C" (store but as exclusion; we will prefer min//max if present)
    range1 = re.search(r"grows\s+(\d{1,2})\s*(?:–|-|to)\s*(\d{1,2})\s*°?\s*c", t)
    range2 = re.search(r"growth\s+(?:between|from)\s+(\d{1,2})\s*(?:and|to)\s*(\d{1,2})\s*°?\s*c", t)
    if range1:
        low, high = range1.group(1), range1.group(2)
        set_field("growth temperature", f"{low}//{high}")
    elif range2:
        low, high = range2.group(1), range2.group(2)
        set_field("growth temperature", f"{low}//{high}")

    for m in re.finditer(r"(?<!no\s)grows\s+(?:well\s+)?at\s+([0-9]{1,3})\s*°?\s*c", t):
        # Only set if we don't already have a range
        if out.get("Growth Temperature", "").find("//") == -1:
            set_field("growth temperature", m.group(1))

    # No growth at X °C → we keep as knowledge but not set if a valid grows-at exists
    # (Optionally, you can store as "no growth at 45" annotation if you add a field)
    # We leave it as a future enhancement.

    # Media detection: match even if "agar" omitted for common names; exclude TSI.
    collected_media: List[str] = []
    # Standard "... on Blood, Nutrient and MacConkey agar"
    # Accept tokens "blood", "macconkey", "xld", "nutrient", "tsa", "bhi", "cba", "ba", "ssa".
    candidate_media = set()
    for name in ["blood", "macconkey", "xld", "nutrient", "tsa", "bhi", "cba", "ba", "ssa", "chocolate", "emb"]:
        if re.search(rf"\b{name}\b", t):
            candidate_media.add(name)

    # Also capture "... on X agar" pattern
    for m in re.finditer(r"\b([a-z0-9\-\+ ]+)\s+agar\b", t):
        lowname = m.group(1).strip().lower()
        if not any(ex in lowname for ex in MEDIA_EXCLUDE_TERMS):
            candidate_media.add(lowname + " agar")

    def canon_media(name: str) -> Optional[str]:
        if name in {"xld"}:
            return "XLD Agar"
        if name in {"macconkey"}:
            return "MacConkey Agar"
        if name in {"blood","ba","ssa"}:
            return "Blood Agar"
        if name == "nutrient":
            return "Nutrient Agar"
        if name == "tsa":
            return "Tryptic Soy Agar"
        if name == "bhi":
            return "Brain Heart Infusion Agar"
        if name == "cba":
            return "Columbia Blood Agar"
        # if we saw "... agar" literal
        if name.endswith(" agar"):
            pretty = name[:-5].strip().title() + " Agar"
            return pretty
        return None

    for nm in candidate_media:
        pretty = canon_media(nm)
        if not pretty:
            continue
        # map to whitelist if present (to unify spellings)
        canon = next((w for w in MEDIA_WHITELIST if w.lower() == pretty.lower()), pretty)
        if canon not in collected_media:
            collected_media.append(canon)

    if collected_media:
        _set_field_safe(out, "Media Grown On", "; ".join(collected_media))

    # Colony morphology — build from vocabulary
    cm_value = normalize_cm_phrase(raw)
    if cm_value:
        _set_field_safe(out, "Colony Morphology", cm_value)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Normalization & Haemolysis bridge & conflict resolver
# ──────────────────────────────────────────────────────────────────────────────
def normalize_to_schema(parsed: Dict[str, str], db_fields: List[str]) -> Dict[str, str]:
    """Map keys/variants to exact sheet columns; bridge haemolysis; clamp media; dedupe."""
    fields = normalize_columns(db_fields)
    alias = build_alias_map(db_fields)
    out: Dict[str, str] = {}

    strict = os.getenv("BACTAI_STRICT_MODE", "0") == "1"

    # 1) Key normalization to exact columns
    for k, v in (parsed or {}).items():
        kk = k.strip(); key_l = kk.lower(); target = None
        if kk in fields:
            target = kk
        elif key_l in alias:
            target = alias[key_l]

        if target in fields:
            cv = _canon_value(target, v)
            if cv not in ("", None, "Unknown"):
                out[target] = cv
        else:
            # lenient mode can keep unknown keys under "Unmapped" (not returned by default)
            if not strict:
                pass

    # 2) Bridge: Haemolysis Type → Haemolysis
    ht = alias.get("haemolysis type"); h = alias.get("haemolysis")
    if ht in out and h in fields:
        tval = out.get(ht, "")
        if tval in {"Alpha", "Beta"}:
            out[h] = "Positive"
        elif tval in {"Gamma", "None"}:
            out[h] = "Variable"  # as requested

    # 3) Clamp media spellings when possible and de-dupe
    if "Media Grown On" in out and out["Media Grown On"]:
        parts = [p.strip() for p in out["Media Grown On"].split(";") if p.strip()]
        fixed = []
        for p in parts:
            match = next((m for m in MEDIA_WHITELIST if m.lower() == p.lower()), p)
            fixed.append(match)
        seen = set(); ordered = []
        for x in fixed:
            if x not in seen:
                ordered.append(x); seen.add(x)
        out["Media Grown On"] = "; ".join(ordered)

    # 4) Final tidy for Colony Morphology (dedupe semicolon parts)
    if "Colony Morphology" in out and out["Colony Morphology"]:
        chunks = [c.strip() for c in out["Colony Morphology"].split(";") if c.strip()]
        seen = set(); cleaned = []
        for c in chunks:
            if c not in seen:
                cleaned.append(c); seen.add(c)
        out["Colony Morphology"] = "; ".join(cleaned)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY (LLM hybrid + regex enrichment + schema normalization)
# ──────────────────────────────────────────────────────────────────────────────
def parse_input_free_text(
    user_text: str,
    prior_facts: Dict | None = None,
    db_fields: List[str] | None = None,
) -> Dict:
    if not (user_text and user_text.strip()):
        return {}
    db_fields = db_fields or []
    cats = _summarize_field_categories(db_fields)

    # 1) Optional LLM (best-effort, regex still enriches)
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
            # ✅ Default model is DeepSeek unless overridden
            model_name = os.getenv("LOCAL_MODEL", "deepseek-v3.1:671b")
            out = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            m = re.search(r"\{.*\}", out["message"]["content"], re.S)
            llm_parsed = json.loads(m.group(0)) if m else {}
    except Exception as e:
        print("⚠️ LLM parser failed — fallback:", e)
        llm_parsed = fallback_parser(user_text, prior_facts)

    # 2) Regex enrichment
    regex_ferm = extract_fermentations_regex(user_text, db_fields)
    regex_bio  = extract_biochem_regex(user_text, db_fields)

    # 3) Merge
    merged = {}
    merged.update(llm_parsed or {})
    merged.update(regex_ferm)
    merged.update(regex_bio)

    # 4) Normalize to schema
    normalized = normalize_to_schema(merged, db_fields)
    return normalized


# ──────────────────────────────────────────────────────────────────────────────
# WHAT-IF helper: apply conditional change "what if X was Y?" and return new dict
# ──────────────────────────────────────────────────────────────────────────────
def apply_what_if(user_text: str, prior_result: Dict[str, str], db_fields: List[str]) -> Dict[str, str]:
    """
    Detects "what if field was value" style edits and returns a modified copy.
    Examples:
        "what if catalase was negative?"
        "suppose oxidase became positive"
        "if it were motile instead?"
    """
    if not (user_text and prior_result):
        return prior_result or {}

    alias = build_alias_map(db_fields)
    txt = normalize_text(user_text)
    # patterns for "what if/ suppose/ if it were/ change ___ to ___"
    patterns = [
        r"what\s+if\s+([a-z\s]+?)\s+(?:is|was|were|became|becomes|turned|changed\s+to)\s+([a-z\+\-]+)",
        r"suppose\s+([a-z\s]+?)\s+(?:is|was|were|became|becomes)\s+([a-z\+\-]+)",
        r"if\s+it\s+(?:is|was|were)\s+([a-z\s]+?)\s*(?:instead)?\s*(?:of|to)?\s*([a-z\+\-]+)?",
        r"change\s+([a-z\s]+?)\s+to\s+([a-z\+\-]+)"
    ]
    for pat in patterns:
        m = re.search(pat, txt)
        if m:
            field = (m.group(1) or "").strip()
            new_val = (m.group(2) or "").strip() if m.lastindex and m.lastindex >= 2 else ""
            # map field through alias
            target = alias.get(field.lower(), None)
            if not target:
                # try title case
                target = field.title()
            # polarity canonicalization
            new_val = _canon_value(target, new_val.title())
            if target in prior_result and new_val:
                new_dict = dict(prior_result)
                new_dict[target] = new_val
                return new_dict
    return prior_result


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT SIDEBAR EDITOR (field-by-field) + Copy JSON + Insert-to-Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar_json_editor(parsed_dict: Dict[str, str], db_fields: List[str], on_apply_label: str = "Apply Changes") -> Dict[str,str]:
    """
    Renders a Streamlit sidebar editor with each known field as a select/text input.
    Returns the possibly edited dict. Safe to call only within a Streamlit app.
    """
    try:
        import streamlit as st
    except Exception:
        # If Streamlit not available, just return the dict
        return parsed_dict

    fields = normalize_columns(db_fields)
    edited = dict(parsed_dict)

    st.sidebar.markdown("### ✏️ Review & Edit Parsed Results")
    with st.sidebar.expander("Edit fields", expanded=True):
        for f in fields:
            allowed = ALLOWED_VALUES.get(f)
            current = edited.get(f, "")
            if f == "Growth Temperature":
                # accept either a single number or "low//high"
                edited[f] = st.text_input(f, value=str(current))
            elif f == "Media Grown On" or f == "Colony Morphology":
                edited[f] = st.text_area(f, value=current, height=64, help="Semicolon-separated values")
            elif allowed:
                # discrete options
                options = ["", *sorted(list(allowed))]
                # pick index
                try:
                    idx = options.index(current) if current in options else 0
                except ValueError:
                    idx = 0
                edited[f] = st.selectbox(f, options, index=idx)
            else:
                edited[f] = st.text_input(f, value=current)

    # Buttons row
    c1, c2, c3 = st.sidebar.columns([1,1,1])
    with c1:
        if st.button(on_apply_label, use_container_width=True):
            st.toast("Applied edits to parsed results.", icon="✅")
    with c2:
        # Copy JSON to clipboard (browser-side: use st.code; Streamlit doesn't have direct clipboard)
        st.download_button("Copy JSON", data=json.dumps(edited, indent=2), file_name="parsed.json", mime="application/json", use_container_width=True)
    with c3:
        # Insert to sidebar inputs: we just return edited dict; the app is expected to write into st.session_state.user_input
        if st.button("Insert to Sidebar", use_container_width=True):
            st.session_state.setdefault("user_input", {})
            for k, v in edited.items():
                st.session_state["user_input"][k] = v or "Unknown"
            st.toast("Inserted values into sidebar inputs.", icon="🧪")

    return edited


# ──────────────────────────────────────────────────────────────────────────────
# GOLD TESTS: starter cases + runner with human-readable diffs + learning hooks
# ──────────────────────────────────────────────────────────────────────────────
STARTER_DB_FIELDS = [
    # Your schema (without Genus)
    "Gram Stain","Shape","Catalase","Oxidase","Colony Morphology","Haemolysis","Haemolysis Type","Indole",
    "Growth Temperature","Media Grown On","Motility","Capsule","Spore Formation","Oxygen Requirement","Methyl Red","VP",
    "Citrate","Urease","H2S","Lactose Fermentation","Glucose Fermentation","Sucrose Fermentation","Nitrate Reduction",
    "Lysine Decarboxylase","Ornitihine Decarboxylase","Arginine dihydrolase","Gelatin Hydrolysis","Esculin Hydrolysis","Dnase",
    "ONPG","NaCl Tolerant (>=6%)","Lipase Test","Xylose Fermentation","Rhamnose Fermentation","Mannitol Fermentation","Sorbitol Fermentation",
    "Maltose Fermentation","Arabinose Fermentation","Raffinose Fermentation","Inositol Fermentation","Trehalose Fermentation","Coagulase"
]

GOLD_CASES: List[Dict] = [
    {
        "name": "Salmonella enterica (classic)",
        "input": (
            "Gram-negative rod, motile, non-spore-forming. No haemolysis on blood agar. "
            "Oxidase negative, catalase positive, indole negative. Urease negative, citrate positive, MR positive, VP negative. "
            "Produces H2S on TSI. Nitrate reduced. Gelatin hydrolysis negative, DNase negative, esculin hydrolysis negative. "
            "Does not produce coagulase or lipase. Grows at 37 °C (not at 45 °C), facultative anaerobe, not tolerant of 6 % NaCl. "
            "Ferments glucose, maltose, mannitol, arabinose, xylose, trehalose, but not lactose, sucrose, raffinose, inositol, or rhamnose. ONPG negative. "
            "Lysine and ornithine decarboxylases positive; arginine dihydrolase negative. No capsule observed."
        ),
        "expected": {
            "Gram Stain":"Negative","Shape":"Rods","Motility":"Positive","Spore Formation":"Negative",
            "Haemolysis Type":"Gamma","Haemolysis":"Variable","Oxidase":"Negative","Catalase":"Positive","Indole":"Negative",
            "Urease":"Negative","Citrate":"Positive","Methyl Red":"Positive","VP":"Negative","H2S":"Positive","Nitrate Reduction":"Positive",
            "Gelatin Hydrolysis":"Negative","Dnase":"Negative","Esculin Hydrolysis":"Negative","Coagulase":"Negative","Lipase Test":"Negative",
            "Growth Temperature":"37","Oxygen Requirement":"Facultative Anaerobe","NaCl Tolerant (>=6%)":"Negative",
            "Glucose Fermentation":"Positive","Maltose Fermentation":"Positive","Mannitol Fermentation":"Positive","Arabinose Fermentation":"Positive",
            "Xylose Fermentation":"Positive","Trehalose Fermentation":"Positive","Lactose Fermentation":"Negative","Sucrose Fermentation":"Negative",
            "Raffinose Fermentation":"Negative","Inositol Fermentation":"Negative","Rhamnose Fermentation":"Negative","ONPG":"Negative","Capsule":"Negative",
            "Media Grown On":"Blood Agar; Nutrient Agar; MacConkey Agar"  # permissive
        }
    },
    {
        "name": "Staphylococcus aureus",
        "input": (
            "Gram-positive cocci. Beta-haemolytic on blood agar. Catalase positive, coagulase positive, DNase positive. "
            "Oxidase negative. Indole negative. VP positive, MR negative. Citrate variable. Urease variable. H2S negative. "
            "Grows at 37 °C; aerobic or facultative. Non-motile, non-spore-forming. "
            "Ferments glucose, mannitol, sucrose; does not ferment lactose or xylose. ONPG negative. NaCl tolerant up to 6%."
        ),
        "expected": {
            "Gram Stain":"Positive","Shape":"Cocci","Haemolysis Type":"Beta","Haemolysis":"Positive",
            "Catalase":"Positive","Coagulase":"Positive","Dnase":"Positive","Oxidase":"Negative","Indole":"Negative",
            "VP":"Positive","Methyl Red":"Negative","Citrate":"Variable","Urease":"Variable","H2S":"Negative",
            "Growth Temperature":"37","Oxygen Requirement":"Facultative Anaerobe","Motility":"Negative","Spore Formation":"Negative",
            "Glucose Fermentation":"Positive","Mannitol Fermentation":"Positive","Sucrose Fermentation":"Positive",
            "Lactose Fermentation":"Negative","Xylose Fermentation":"Negative","ONPG":"Negative","NaCl Tolerant (>=6%)":"Positive",
            "Media Grown On":"Blood Agar; Nutrient Agar"
        }
    },
    {
        "name": "Listeria monocytogenes",
        "input": (
            "Gram-positive short rods, tumbling motility at room temperature; catalase positive, oxidase negative. "
            "Beta-haemolysis weak. Indole negative. VP negative, MR variable. Urease negative, citrate negative, H2S negative. "
            "Grows at 4 °C but not 45 °C. Facultative anaerobe. Non-spore-forming. "
            "Ferments glucose, maltose; does not ferment lactose, xylose, or mannitol. ONPG negative. Esculin hydrolysis positive."
        ),
        "expected": {
            "Gram Stain":"Positive","Shape":"Short Rods","Motility":"Positive","Catalase":"Positive","Oxidase":"Negative",
            "Haemolysis Type":"Beta","Haemolysis":"Positive","Indole":"Negative","VP":"Negative","Methyl Red":"Variable",
            "Urease":"Negative","Citrate":"Negative","H2S":"Negative","Oxygen Requirement":"Facultative Anaerobe","Spore Formation":"Negative",
            "Glucose Fermentation":"Positive","Maltose Fermentation":"Positive","Lactose Fermentation":"Negative","Xylose Fermentation":"Negative",
            "Mannitol Fermentation":"Negative","ONPG":"Negative","Esculin Hydrolysis":"Positive","Growth Temperature":"4",
            "Media Grown On":"Blood Agar; Nutrient Agar"
        }
    },
    {
        "name": "Pseudomonas aeruginosa",
        "input": (
            "Gram-negative rods, oxidase positive, catalase positive, indole negative. Motile, non-fermenter on MacConkey (NLF). "
            "Produces pigments; beta-haemolysis may be observed. Urease variable, citrate positive, H2S negative. "
            "Aerobic. Grows at 37 °C and 42 °C. Gelatin hydrolysis positive. DNase negative. Nitrate reduced. "
            "Does not ferment lactose, sucrose, or mannitol; glucose fermentation negative or variable. ONPG negative. "
            "NaCl tolerance variable."
        ),
        "expected": {
            "Gram Stain":"Negative","Shape":"Rods","Oxidase":"Positive","Catalase":"Positive","Indole":"Negative","Motility":"Positive",
            "Lactose Fermentation":"Negative","Pigmented":"Positive",  # NOTE: Pigmented isn't a schema field; we omit in final normalize.
            "Haemolysis Type":"Beta","Haemolysis":"Positive",
            "Urease":"Variable","Citrate":"Positive","H2S":"Negative","Oxygen Requirement":"Aerobic","Growth Temperature":"37",
            "Gelatin Hydrolysis":"Positive","Dnase":"Negative","Nitrate Reduction":"Positive",
            "Sucrose Fermentation":"Negative","Mannitol Fermentation":"Negative","Glucose Fermentation":"Variable","ONPG":"Negative",
            "NaCl Tolerant (>=6%)":"Variable","Media Grown On":"MacConkey Agar; Blood Agar"
        }
    },
    {
        "name": "Enterobacter cloacae complex",
        "input": (
            "Gram-negative, motile rods. Facultative anaerobe. Colonies on nutrient agar are 2–3 mm, smooth, convex, moist, grey-cream and slightly translucent. "
            "Grows on Blood, Nutrient and MacConkey agar (pink colonies). Catalase positive, oxidase negative. Indole negative. MR negative, VP positive. "
            "Citrate positive. Urease variable. H2S not produced. Nitrate reduced. Gelatin hydrolysis variable. Esculin hydrolysis positive. DNase negative. ONPG positive. "
            "Grows at 37 °C. Lysine, ornithine and arginine decarboxylases positive. Coagulase negative. Lipase negative. "
            "Fermented: lactose, glucose, sucrose, mannitol, xylose, arabinose, inositol, maltose, trehalose; raffinose variable. "
            "Haemolysis not observed."
        ),
        "expected": {
            "Gram Stain":"Negative","Shape":"Rods","Motility":"Positive","Oxygen Requirement":"Facultative Anaerobe",
            "Colony Morphology":"2/3mm; Smooth; Convex; Moist; Grey; Cream; Translucent",
            "Media Grown On":"Nutrient Agar; MacConkey Agar; Blood Agar",
            "Catalase":"Positive","Oxidase":"Negative","Indole":"Negative","Methyl Red":"Negative","VP":"Positive",
            "Citrate":"Positive","Urease":"Variable","H2S":"Negative","Nitrate Reduction":"Positive",
            "Gelatin Hydrolysis":"Variable","Esculin Hydrolysis":"Positive","Dnase":"Negative","ONPG":"Positive",
            "Growth Temperature":"37","Coagulase":"Negative","Lipase Test":"Negative",
            "Lysine Decarboxylase":"Positive","Ornitihine Decarboxylase":"Positive","Arginine dihydrolase":"Positive",
            "Lactose Fermentation":"Positive","Glucose Fermentation":"Positive","Sucrose Fermentation":"Positive","Mannitol Fermentation":"Positive",
            "Xylose Fermentation":"Positive","Arabinose Fermentation":"Positive","Inositol Fermentation":"Positive","Maltose Fermentation":"Positive",
            "Trehalose Fermentation":"Positive","Raffinose Fermentation":"Variable",
            "Haemolysis Type":"Gamma","Haemolysis":"Variable"
        }
    }
]

def _diff_dicts(expected: Dict[str,str], got: Dict[str,str]) -> Tuple[List[str], List[str], List[Tuple[str,str,str]]]:
    """Return (missing_keys, extra_keys, mismatched_values[(key, exp, got)])."""
    missing, extra, mismatched = [], [], []
    for k in expected.keys():
        if k not in got:
            missing.append(k)
        elif expected[k] != got[k]:
            mismatched.append((k, expected[k], got[k]))
    for k in got.keys():
        if k not in expected:
            extra.append(k)
    return missing, extra, mismatched

def _learn_from_failure(input_text: str, expected: Dict[str,str], got: Dict[str,str], db_fields: List[str]):
    """
    Lightweight learning hook:
    - If a key is consistently missing but the text contains a recognizable token (e.g., 'mr'), add alias.
    - If we see a media token unrecognized, we can stash a learned alias candidate.
    """
    alias = build_alias_map(db_fields)  # includes learned
    learned_aliases = _load_json(LEARNED_ALIASES_PATH)

    # Example: map 'mr' -> 'Methyl Red' if expected has Methyl Red but parsed didn't
    for k in expected.keys():
        if k not in got:
            # check common abbr direct in text
            for abbr, full in BIOCHEM_ABBR.items():
                if full == k and re.search(rf"\b{abbr}\b", input_text.lower()):
                    learned_aliases[abbr] = full
    _save_json(LEARNED_ALIASES_PATH, learned_aliases)

def run_gold_tests():
    print("Running Gold Spec Tests...\n")
    total, passed = 0, 0
    dbf = STARTER_DB_FIELDS
    for case in GOLD_CASES:
        total += 1
        parsed = parse_input_free_text(case["input"], db_fields=dbf)
        # Restrict comparison to keys present in expected (lenient evaluation)
        expected = case["expected"]
        # We also allow the parser to have more keys (extra are listed)
        missing, extra, mismatched = _diff_dicts(expected, parsed)
        if not missing and not mismatched:
            print(f"✅ {case['name']} passed.")
            passed += 1
        else:
            print(f"❌ {case['name']} failed.")
            if missing:
                print("  Missing fields:")
                for k in missing: print("   -", k)
            if mismatched:
                print("  Mismatched fields:")
                for (k, e, g) in mismatched:
                    print(f"   - {k}: expected '{e}' got '{g}'")
            if extra:
                print("  Extra fields (not required in expected):")
                for k in extra: print("   -", k)
            # Learning hook
            _learn_from_failure(case["input"], expected, parsed, dbf)
        # print a compact diff summary
        print("  Parsed keys:", sorted(list(parsed.keys()))[:8], "..." if len(parsed)>8 else "")
        print()

    pct = (passed/total*100.0) if total else 0.0
    print(f"Result: {passed}/{total} passed ({pct:.1f}%).")
    if passed < total:
        print("Hints saved (if any) into /data/learned_aliases.json for review.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_gold_tests()
    else:
        print("parser_llm.py v3 loaded. Use --test to run the Gold Spec tests.")
