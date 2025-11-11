# run_gold_tests.py — unified Gold Tests runner (LLM-first + Basic fallback)
# - Filters expected fields to current DB schema
# - Logs failures to parser_feedback.json
# - Triggers analyze_feedback_and_learn + auto_update_parser_regex for BOTH parsers
# - Optional: auto-commit to GitHub if env/secrets are present

import os
import sys
import json
from datetime import datetime

# --- Parsers ---
from parser_llm import (
    parse_input_free_text as parse_llm,
    analyze_feedback_and_learn as llm_learn,
    auto_update_parser_regex as llm_autopatch,
)
from parser_basic import (
    parse_input_free_text as parse_basic,
    analyze_feedback_and_learn as basic_learn,
    auto_update_parser_regex as basic_autopatch,
)

FEEDBACK_PATH = "parser_feedback.json"
MEMORY_PATH   = "parser_memory.json"
GOLD_PATH     = "gold_tests.json"

# --- Optional repo sync ---
def auto_git_commit():
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    repo  = os.getenv("GITHUB_REPO")  # e.g. "EphraimAsad/BacAi-D-Dev"
    email = os.getenv("GITHUB_EMAIL", os.getenv("GIT_USER_EMAIL", "bot@bactaid.local"))
    name  = os.getenv("GITHUB_NAME", os.getenv("GIT_USER_NAME", "BactAI-D AutoLearner"))
    branch= os.getenv("GIT_BRANCH","main")

    if not token or not repo:
        print("ℹ️ Skipping git push (GITHUB_TOKEN/GH_TOKEN or GITHUB_REPO missing).")
        return

    import subprocess
    try:
        subprocess.run(["git","config","--global","user.email", email], check=True)
        subprocess.run(["git","config","--global","user.name",  name ], check=True)
        # Only push if there are actual changes
        status = subprocess.run(["git","status","--porcelain"], capture_output=True, text=True)
        if not status.stdout.strip():
            print("ℹ️ No local changes to commit.")
            return
        subprocess.run(["git","add","parser_llm.py","parser_basic.py","parser_memory.json","parser_feedback.json"], check=False)
        msg = f"🤖 Auto-learned regex update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git","commit","-m", msg], check=False)
        remote_url = f"https://{token}@github.com/{repo}.git"
        subprocess.run(["git","push", remote_url, f"HEAD:{branch}"], check=True)
        print("✅ Pushed learning updates to GitHub.")
    except Exception as e:
        print(f"⚠️ Auto-commit failed: {e}")

def load_gold():
    try:
        with open(GOLD_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ gold_tests.json not found in repo root.")
        sys.exit(1)

def save_feedback(feedback):
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2, ensure_ascii=False)

def main():
    # You can edit this schema list or import directly from your live Excel at runtime.
    db_fields = [
        "Gram Stain","Shape","Motility","Oxidase","Catalase","Indole","Urease",
        "Citrate","Methyl Red","VP","Dnase","Gelatin Hydrolysis","Esculin Hydrolysis",
        "Nitrate Reduction","H2S","Oxygen Requirement","Growth Temperature",
        "Media Grown On","Colony Morphology","Haemolysis","Haemolysis Type","Coagulase",
        "Lysine Decarboxylase","Ornitihine Decarboxylase","Arginine dihydrolase","ONPG",
        "NaCl Tolerant (>=6%)","Lipase Test","Lactose Fermentation","Glucose Fermentation",
        "Sucrose Fermentation","Maltose Fermentation","Mannitol Fermentation","Xylose Fermentation",
        "Arabinose Fermentation","Rhamnose Fermentation","Raffinose Fermentation",
        "Inositol Fermentation","Trehalose Fermentation"
    ]

    tests = load_gold()
    total = 0
    passed = 0
    feedback = []

    print("\n▶️ Running Gold Spec Tests (LLM-first; Basic fallback)\n")

    for case in tests:
        total += 1
        name = case.get("name", f"Case_{total}")
        text = case.get("input", "")
        expected_raw = case.get("expected", {})

        # Only validate keys that are in the CURRENT DB schema
        expected = {k: v for k, v in expected_raw.items() if k in db_fields}

        # Try LLM first, then fallback to Basic regex if anything goes wrong
        backend = "LLM"
        try:
            parsed = parse_llm(text, prior_facts={}, db_fields=db_fields)
        except Exception:
            backend = "Basic"
            parsed = parse_basic(text, prior_facts={}, db_fields=db_fields)

        # Compare
        mismatches = []
        for k, exp in expected.items():
            got = parsed.get(k, "")
            if str(got) != str(exp):
                mismatches.append({"field": k, "expected": exp, "got": got})

        if not mismatches:
            print(f"✅ {name} — Passed ({backend})")
            passed += 1
        else:
            print(f"❌ {name} — Mismatches ({backend})")
            for m in mismatches:
                print(f"   - {m['field']}: expected '{m['expected']}', got '{m['got']}'")
            # Log for learning
            feedback.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "name": name,
                "text": text,
                "errors": mismatches
            })

    print(f"\n🧩 Summary: {passed}/{total} passed")

    # Persist feedback → trigger learning → auto-patch both parsers
    save_feedback(feedback)
    if feedback:
        print("\n🧠 Learning from failures + updating regex rules…")
        # LLM parser learn + patch
        llm_learn(FEEDBACK_PATH, MEMORY_PATH)
        llm_autopatch(MEMORY_PATH, "parser_llm.py")
        # Basic parser learn + patch
        basic_learn()
        basic_autopatch()

        # Optional: push changes if CI token is present
        if (os.getenv("ENABLE_AUTO_COMMIT","false").lower() == "true") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"):
            auto_git_commit()
    else:
        print("🎉 No failures → no learning updates needed.")

if __name__ == "__main__":
    main()