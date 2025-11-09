import json
from parser_llm import parse_input_free_text

# Load database field list from your Excel schema, or manually define minimal set
db_fields = list([
    "Gram Stain","Shape","Motility","Oxidase","Catalase","Indole","Urease","Citrate","Methyl Red","VP",
    "DNase","Gelatin Hydrolysis","Esculin Hydrolysis","Nitrate Reduction","H2S","Oxygen Requirement",
    "Growth Temperature","Media Grown On","Colony Morphology","Haemolysis","Haemolysis Type","Coagulase",
    "Lysine Decarboxylase","Ornitihine Decarboxylase","Arginine dihydrolase","ONPG",
    "NaCl Tolerant (>=6%)","Lipase Test","Lactose Fermentation","Glucose Fermentation","Sucrose Fermentation",
    "Maltose Fermentation","Mannitol Fermentation","Xylose Fermentation","Arabinose Fermentation","Rhamnose Fermentation",
    "Raffinose Fermentation","Inositol Fermentation","Trehalose Fermentation"
])

with open("gold_tests.json", "r", encoding="utf-8") as f:
    tests = json.load(f)

total, passed = 0, 0

for case in tests:
    print(f"\n🧫 {case['name']}")
    parsed = parse_input_free_text(case["input"], db_fields=db_fields)
    expected = case["expected"]
    mismatched = []
    for key, exp_val in expected.items():
        val = parsed.get(key)
        if val != exp_val:
            mismatched.append((key, val, exp_val))
    if not mismatched:
        print("✅ Passed all checks.")
        passed += 1
    else:
        print("❌ Mismatches:")
        for k, val, exp in mismatched:
            print(f"   • {k}: got {val!r}, expected {exp!r}")
    total += 1

print(f"\n✅ {passed}/{total} tests passed.")
