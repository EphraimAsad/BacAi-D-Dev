#!/usr/bin/env python3
"""
gold_tests_runner.py — robust runner for BactAI-D parser gold tests + self-learning

- Ensures imports resolve by pinning repo root to sys.path
- Uses absolute paths based on this file’s directory
- Runs LLM-first parser tests with Basic (regex) fallback
- Writes feedback + memory to repo root
- Applies auto-patches to parser_llm.py and parser_basic.py
- Optional Git push if GH_TOKEN / GITHUB_REPO provided
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
import importlib

# ──────────────────────────────────────────────────────────────────────────────
# Resolve repo root + ensure on sys.path
# ──────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE  # put this file in repo root; else adjust to HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLD_TESTS_PATH = REPO_ROOT / "gold_tests.json"
FEEDBACK_PATH   = REPO_ROOT / "parser_feedback.json"
MEMORY_PATH     = REPO_ROOT / "parser_memory.json"
PARSER_LLM_PATH = REPO_ROOT / "parser_llm.py"
PARSER_BASIC_PATH = REPO_ROOT / "parser_basic.py"

def log_env():
    print("── gold_tests_runner diagnostics ──")
    print(f"cwd: {Path.cwd()}")
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"gold_tests.json exists: {GOLD_TESTS_PATH.exists()}")
    print(f"parser_llm.py exists:  {PARSER_LLM_PATH.exists()}")
    print(f"parser_basic.py exists:{PARSER_BASIC_PATH.exists()}")
    print(f"sys.path[0]: {sys.path[0]}")
    print("repo root listing:", [p.name for p in REPO_ROOT.iterdir()])
    print("───────────────────────────────────")

log_env()
importlib.invalidate_caches()

# ──────────────────────────────────────────────────────────────────────────────
# Safe imports with diagnostics
# ──────────────────────────────────────────────────────────────────────────────
try:
    from parser_llm import (
        parse_input_free_text as parse_llm_input_free_text,
        enable_self_learning_autopatch as llm_autopatch,
    )
    print("✅ Imported parser_llm successfully.")
except Exception as e:
    print(f"❌ Failed to import parser_llm: {e!r}")
    print("Tip: ensure parser_llm.py is at repo root and not a folder named 'parser_llm/'.")
    raise

try:
    from parser_basic import (
        parse_input_free_text as parse_basic_input_free_text,
        enable_self_learning_autopatch as basic_autopatch,
    )
    print("✅ Imported parser_basic successfully.")
except Exception as e:
    print(f"❌ Failed to import parser_basic: {e!r}")
    raise

# ──────────────────────────────────────────────────────────────────────────────
# Load DB schema dynamically (optional but recommended)
# ──────────────────────────────────────────────────────────────────────────────
# If you prefer a static schema, you can hardcode your list instead.
try:
    import pandas as pd
    # Prefer data/bacteria_db.xlsx, fallback bacteria_db.xlsx
    db_path = REPO_ROOT / "data" / "bacteria_db.xlsx"
    if not db_path.exists():
        alt = REPO_ROOT / "bacteria_db.xlsx"
        db_path = alt if alt.exists() else db_path
    if db_path.exists():
        df = pd.read_excel(db_path)
        df.columns = [str(c).strip() for c in df.columns]
        db_fields = [c for c in df.columns if c.lower() != "genus"]
        print(f"📚 Loaded DB fields ({len(db_fields)}): {', '.join(db_fields)}")
    else:
        print("⚠️ No database found, falling back to conservative static schema.")
        db_fields = [
            "Gram Stain","Shape","Motility","Oxidase","Catalase","Indole","Urease",
            "Citrate","Methyl Red","VP","DNase","Gelatin Hydrolysis","Esculin Hydrolysis",
            "Nitrate Reduction","H2S","Oxygen Requirement","Growth Temperature",
            "Media Grown On","Colony Morphology","Haemolysis","Haemolysis Type","Coagulase",
            "Lysine Decarboxylase","Ornithine Decarboxylase","Arginine dihydrolase","ONPG",
            "NaCl Tolerant (>=6%)","Lipase Test","Lactose Fermentation","Glucose Fermentation",
            "Sucrose Fermentation","Maltose Fermentation","Mannitol Fermentation","Xylose Fermentation",
            "Arabinose Fermentation","Rhamnose Fermentation","Raffinose Fermentation",
            "Inositol Fermentation","Trehalose Fermentation"
        ]
except Exception as e:
    print(f"⚠️ Could not load DB schema dynamically: {e!r}")
    db_fields = [
        "Gram Stain","Shape","Motility","Oxidase","Catalase","Indole","Urease",
        "Citrate","Methyl Red","VP","DNase","Gelatin Hydrolysis","Esculin Hydrolysis",
        "Nitrate Reduction","H2S","Oxygen Requirement","Growth Temperature",
        "Media Grown On","Colony Morphology","Haemolysis","Haemolysis Type","Coagulase",
        "Lysine Decarboxylase","Ornithine Decarboxylase","Arginine dihydrolase","ONPG",
        "NaCl Tolerant (>=6%)","Lipase Test","Lactose Fermentation","Glucose Fermentation",
        "Sucrose Fermentation","Maltose Fermentation","Mannitol Fermentation","Xylose Fermentation",
        "Arabinose Fermentation","Rhamnose Fermentation","Raffinose Fermentation",
        "Inositol Fermentation","Trehalose Fermentation"
    ]

# ──────────────────────────────────────────────────────────────────────────────
# Load tests
# ──────────────────────────────────────────────────────────────────────────────
if not GOLD_TESTS_PATH.exists():
    print("⚠️ gold_tests.json not found in repo root.")
    sys.exit(1)

with open(GOLD_TESTS_PATH, "r", encoding="utf-8") as f:
    tests = json.load(f)

total, passed = 0, 0
feedback = []

print(f"🧪 Running {len(tests)} gold tests...")
for case in tests:
    total += 1
    name = case.get("name", f"Case_{total}")
    text = case.get("input", "")
    expected = case.get("expected", {})

    print(f"\n— {name}")
    # LLM-first with regex fallback
    try:
        parsed = parse_llm_input_free_text(text, prior_facts={}, db_fields=db_fields)
        backend = "LLM"
    except Exception as e:
        print(f"  LLM parse failed ({e!r}); falling back to regex.")
        parsed = parse_basic_input_free_text(text, prior_facts={}, db_fields=db_fields)
        backend = "Regex"

    mismatches = []
    for k, vexp in expected.items():
        if k not in db_fields:
            # Ignore expectations for fields not present in schema
            continue
        vgot = parsed.get(k, "")
        if str(vgot) != str(vexp):
            mismatches.append({"field": k, "got": vgot, "expected": vexp})

    if not mismatches:
        print(f"  ✅ Passed ({backend})")
        passed += 1
    else:
        print(f"  ❌ Failed ({backend})")
        for m in mismatches:
            print(f"    - {m['field']}: got {m['got']!r}, expected {m['expected']!r}")
        feedback.append({"name": name, "text": text, "errors": mismatches})

# Save feedback for learning (repo root)
with open(FEEDBACK_PATH, "w", encoding="utf-8") as fb:
    json.dump(feedback, fb, indent=2, ensure_ascii=False)

print(f"\n🧩 Gold Test Summary: {passed}/{total} passed.")

# ──────────────────────────────────────────────────────────────────────────────
# Apply learning + auto-patch (both parsers)
# ──────────────────────────────────────────────────────────────────────────────
try:
    # LLM parser learner
    llm_autopatch(run_tests=False, db_fields=db_fields)
    print("✅ Applied LLM parser self-learning/auto-patch.")
except Exception as e:
    print(f"⚠️ LLM autopatch failed: {e!r}")

try:
    # Regex parser learner
    basic_autopatch(run_tests=False, db_fields=db_fields)
    print("✅ Applied regex parser self-learning/auto-patch.")
except Exception as e:
    print(f"⚠️ Regex autopatch failed: {e!r}")

# ──────────────────────────────────────────────────────────────────────────────
# Optional Git auto-commit (same envs as app)
# ──────────────────────────────────────────────────────────────────────────────
def try_git_commit():
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    repo  = os.getenv("GITHUB_REPO")
    branch = os.getenv("GIT_BRANCH", "main")
    email = os.getenv("GITHUB_EMAIL", "bot@bactaid.local")
    name  = os.getenv("GITHUB_NAME", "BactAI-D AutoLearner")

    if not token or not repo:
        print("ℹ️ Skipping Git push (missing GH_TOKEN/GITHUB_REPO).")
        return

    try:
        # Ensure repo and remote exist
        if not (REPO_ROOT / ".git").exists():
            print("🧩 Initializing git repo for push...")
            subprocess.run(["git", "init"], cwd=str(REPO_ROOT), check=False)
            subprocess.run(["git", "remote", "add", "origin", f"https://{token}@github.com/{repo}.git"],
                           cwd=str(REPO_ROOT), check=False)
            subprocess.run(["git", "fetch", "origin", branch], cwd=str(REPO_ROOT), check=False)
            subprocess.run(["git", "checkout", "-B", branch], cwd=str(REPO_ROOT), check=False)

        subprocess.run(["git", "config", "--global", "user.email", email], check=False)
        subprocess.run(["git", "config", "--global", "user.name", name], check=False)

        status = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT),
                                capture_output=True, text=True)
        if not status.stdout.strip():
            print("ℹ️ No changes to commit.")
            return

        subprocess.run(["git", "add",
                        "parser_llm.py", "parser_basic.py",
                        "parser_memory.json", "parser_feedback.json"],
                       cwd=str(REPO_ROOT), check=False)

        msg = f"🤖 Gold tests + auto-learn — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", msg], cwd=str(REPO_ROOT), check=False)
        subprocess.run(["git", "push", "origin", f"HEAD:{branch}"], cwd=str(REPO_ROOT), check=True)
        print("✅ Pushed learning updates to GitHub.")
    except Exception as e:
        print(f"⚠️ Git push failed: {e!r}")

try_git_commit()
