import json
from parser_llm import parse_input_free_text

# --- FIELD SCHEMA ---
db_fields = [
    "Gram Stain", "Shape", "Motility", "Oxidase", "Catalase", "Indole", "Urease",
    "Citrate", "Methyl Red", "VP", "DNase", "Gelatin Hydrolysis", "Esculin Hydrolysis",
    "Nitrate Reduction", "H2S", "Oxygen Requirement", "Growth Temperature",
    "Media Grown On", "Colony Morphology", "Haemolysis", "Haemolysis Type", "Coagulase",
    "Lysine Decarboxylase", "Ornithine Decarboxylase", "Arginine dihydrolase", "ONPG",
    "NaCl Tolerant (>=6%)", "Lipase Test", "Lactose Fermentation", "Glucose Fermentation",
    "Sucrose Fermentation", "Maltose Fermentation", "Mannitol Fermentation", "Xylose Fermentation",
    "Arabinose Fermentation", "Rhamnose Fermentation", "Raffinose Fermentation",
    "Inositol Fermentation", "Trehalose Fermentation"
]

# --- LOAD GOLD TESTS ---
try:
    with open("gold_tests.json", "r", encoding="utf-8") as f:
        tests = json.load(f)
except FileNotFoundError:
    print("⚠️ gold_tests.json not found. Make sure it's in the project root.")
    exit()

# --- RUN TESTS ---
total, passed = 0, 0

for case in tests:
    name = case.get("name", f"Case_{total+1}")
    input_text = case.get("input", "")
    expected = case.get("expected", {})

    print(f"\n🧫 {name}")
    try:
        parsed = parse_input_free_text(input_text, db_fields=db_fields)
    except Exception as e:
        print(f"⚠️ Parser failed on input: {e}")
        continue

    mismatched = []
    for key, exp_val in expected.items():
        val = parsed.get(key, "")
        if str(val).strip() != str(exp_val).strip():
            mismatched.append((key, val, exp_val))

    if not mismatched:
        print("✅ Passed all checks.")
        passed += 1
    else:
        print("❌ Mismatches:")
        for k, val, exp in mismatched:
            print(f"   • {k}: got {val!r}, expected {exp!r}")

    total += 1

print(f"\n🧩 Gold Test Summary: {passed}/{total} tests passed.")
