# app_chat.py — LLM-first chat (Ollama Cloud) with Basic regex fallback
# - Git auto-commit enabled (runtime .git init for Streamlit Cloud)
# - Gold Spec Tests trigger self-learning + regex patch
# - Fully compatible with updated parser_llm.py (DeepSeek v3.1:671b)

import os
import re
import json
from datetime import datetime
import subprocess
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# EARLY GIT INIT — required for Streamlit Cloud to push back to GitHub
# ──────────────────────────────────────────────────────────────────────────────
def ensure_git_repo():
    """Initializes a git repo in the current directory if none exists."""
    gh_token = os.getenv("GH_TOKEN")
    gh_repo = os.getenv("GITHUB_REPO")
    branch = os.getenv("GIT_BRANCH", "main")

    if not gh_token or not gh_repo:
        print("⚠️ Missing GH_TOKEN or GITHUB_REPO; skipping repo init.")
        return

    if not os.path.exists(".git"):
        print("🧩 Initializing git repo for Streamlit Cloud...")
        subprocess.run(["git", "init"], check=False)
        subprocess.run(
            ["git", "remote", "add", "origin",
             f"https://{gh_token}@github.com/{gh_repo}.git"],
            check=False,
        )
        subprocess.run(["git", "fetch", "origin", branch], check=False)
        subprocess.run(["git", "checkout", "-B", branch], check=False)
        print("✅ Git repo initialized and connected to remote.")

ensure_git_repo()

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
from engine import BacteriaIdentifier
# LLM-first parser (Ollama Cloud)
from parser_llm import parse_input_free_text as parse_llm_input_free_text
# Deterministic fallback parser
from parser_basic import parse_input_free_text as parse_basic_input_free_text
# Self-learning and autopatch
from parser_llm import enable_self_learning_autopatch
enable_self_learning_autopatch(run_tests=False)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BactAI-D — Language Reasoning (Chat)", layout="wide")
DEFAULT_LOCAL_MODEL = os.getenv("LOCAL_MODEL", "deepseek-v3.1:671b")

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path, last_modified):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

primary_path = os.path.join("data", "bacteria_db.xlsx")
fallback_path = os.path.join("bacteria_db.xlsx")
data_path = primary_path if os.path.exists(primary_path) else fallback_path

try:
    last_modified = os.path.getmtime(data_path)
except FileNotFoundError:
    st.error(f"Database not found at '{primary_path}' or '{fallback_path}'.")
    st.stop()

db = load_data(data_path, last_modified)
eng = BacteriaIdentifier(db)
db_fields = [c for c in db.columns if c.strip().lower() != "genus"]

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧫 BactAI-D — Language Reasoning (Chat)")
st.caption(f"📅 Database last updated: {datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("""
Describe your findings in plain English (e.g.,
_'Gram negative rod, oxidase positive, motile, non-lactose fermenter on MacConkey.'_
). I’ll parse it, run the BactAI-D engine, and explain the result.
""")
st.markdown("""
**How to read the scores**

- **Confidence** – Based **only on the tests you entered**.
- **True Confidence (All Tests)** – The same score scaled by **all possible fields in the database**.
""")

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if "facts" not in st.session_state:
    st.session_state.facts = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "gold_results" not in st.session_state:
    st.session_state.gold_results = None
if "gold_summary" not in st.session_state:
    st.session_state.gold_summary = None
if "active_parser" not in st.session_state:
    st.session_state.active_parser = f"LLM (Ollama: {DEFAULT_LOCAL_MODEL})"

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Runtime")
st.sidebar.text_input(
    "Active Parser",
    value=st.session_state.active_parser,
    disabled=True,
    help="The parser currently used by the app."
)

with st.sidebar.expander("🧬 Supported Tests (database fields)", expanded=False):
    st.write(", ".join(sorted(db_fields)))

with st.sidebar.expander("🧪 Parsed Fields (from chat)", expanded=True):
    if st.session_state.facts:
        st.json(st.session_state.facts)
    else:
        st.caption("No parsed fields yet. Send a message to begin.")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset conversation"):
    for key in ["facts", "history", "last_results", "gold_results", "gold_summary"]:
        st.session_state[key] = {} if key == "facts" else None
    st.session_state.active_parser = f"LLM (Ollama: {DEFAULT_LOCAL_MODEL})"
    st.rerun()
# ──────────────────────────────────────────────────────────────────────────────
# GOLD SPEC TESTS — runs from gold_tests.json and triggers self-learning
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar.expander("🧪 Gold Spec Tests (Parser Validation)", expanded=False):
    if st.button("▶️ Run Gold Spec Tests & Self-Learn"):
        try:
            # Load test cases from file
            with open("gold_tests.json", "r", encoding="utf-8") as f:
                tests = json.load(f)

            results = []
            feedback = []
            passed = 0

            for case in tests:
                text = case["input"]
                expected = case.get("expected", {})

                # Try LLM first, then fallback to Basic
                used_backend = None
                try:
                    parsed = parse_llm_input_free_text(
                        text,
                        prior_facts={},
                        db_fields=db_fields
                    )
                    used_backend = f"LLM (Ollama: {DEFAULT_LOCAL_MODEL})"
                except Exception:
                    parsed = parse_basic_input_free_text(
                        text,
                        prior_facts={},
                        db_fields=db_fields
                    )
                    used_backend = "Basic (regex)"

                # Compare only expected keys (gold standard)
                mismatched = []
                for key, exp_val in expected.items():
                    got_val = parsed.get(key)
                    if got_val != exp_val:
                        mismatched.append({"field": key, "got": got_val, "expected": exp_val})

                if mismatched:
                    results.append({
                        "name": case.get("name", "Unnamed Case"),
                        "status": "❌",
                        "mismatches": mismatched,
                        "parsed": parsed,
                        "expected": expected,
                        "backend": used_backend
                    })
                    feedback.append({"name": case.get("name", "Unnamed Case"), "text": text, "errors": mismatched})
                else:
                    results.append({
                        "name": case.get("name", "Unnamed Case"),
                        "status": "✅",
                        "mismatches": [],
                        "parsed": parsed,
                        "expected": expected,
                        "backend": used_backend
                    })
                    passed += 1

            # Save feedback for learning
            with open("parser_feedback.json", "w", encoding="utf-8") as fb:
                json.dump(feedback, fb, indent=2)

            # Learning + auto-regex patch in parser_llm.py
            from parser_llm import analyze_feedback_and_learn, auto_update_parser_regex
            analyze_feedback_and_learn("parser_feedback.json", "parser_memory.json")
            auto_update_parser_regex("parser_memory.json", "parser_llm.py")

            st.session_state.gold_results = results
            st.session_state.gold_summary = (passed, len(tests))
            st.success(f"Gold Spec Tests complete ({passed}/{len(tests)} passed).")

        except FileNotFoundError:
            st.error("⚠️ gold_tests.json not found. Add it to the repo root.")
        except Exception as e:
            st.error(f"Error running Gold Test Suite: {e}")

    # Clear learning memory
    if st.button("🧹 Clear Learning Memory"):
        try:
            if os.path.exists("parser_feedback.json"):
                os.remove("parser_feedback.json")
                st.success("Cleared feedback memory.")
            else:
                st.info("No feedback file found.")
        except Exception as e:
            st.error(f"Could not clear memory: {e}")

# Display Gold Test results in main area (when present)
if st.session_state.gold_results is not None:
    passed, total = st.session_state.gold_summary or (0, 0)
    st.markdown("## 🧪 Gold Test Results (Chat)")
    st.markdown(f"**Summary:** {passed}/{total} passed")
    for r in st.session_state.gold_results:
        if r["status"] == "✅":
            st.markdown(f"**{r['name']}** — ✅ Passed all checks · _{r.get('backend','')}_")
        else:
            st.markdown(f"**{r['name']}** — ❌ Mismatches · _{r.get('backend','')}_")
            for m in r["mismatches"]:
                st.markdown(f"- **{m['field']}**: got `{m['got']}`, expected `{m['expected']}`")
            with st.expander("Show parsed vs expected"):
                st.code(json.dumps({"parsed": r["parsed"], "expected": r["expected"]}, indent=2), language="json")

st.sidebar.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# CHAT HISTORY
# ──────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ──────────────────────────────────────────────────────────────────────────────
# CHAT INPUT → Parse (LLM first, fallback to Basic), identify, reply
# ──────────────────────────────────────────────────────────────────────────────
def parse_with_fallback(user_text: str, prior_facts: dict, db_fields: list) -> tuple[dict, str]:
    """
    Try LLM (Ollama Cloud) first. If it fails in any way, fall back to Basic regex.
    Returns: (parsed_dict, backend_label)
    """
    # Attempt LLM parse
    try:
        parsed = parse_llm_input_free_text(
            user_text,
            prior_facts=prior_facts,
            db_fields=db_fields
        )
        return parsed, f"LLM (Ollama: {DEFAULT_LOCAL_MODEL})"
    except Exception:
        # Fallback to deterministic parser
        parsed = parse_basic_input_free_text(
            user_text,
            prior_facts=prior_facts,
            db_fields=db_fields
        )
        return parsed, "Basic (regex)"

user_msg = st.chat_input("Tell me your observations…")
if user_msg:
    st.session_state.history.append({"role": "user", "content": user_msg})
    st.chat_message("user").markdown(user_msg)

    # Parse (LLM first, fallback to Basic)
    parsed, backend_label = parse_with_fallback(user_msg, st.session_state.facts, db_fields)
    # Update the visible indicator to reflect what we actually used this turn
    st.session_state.active_parser = backend_label

    # Identify
    results = eng.identify(parsed)

    if not results:
        reply = (
            "I couldn't find a strong match with the current information. "
            "Try adding more descriptive test results (e.g., ONPG, NaCl tolerance, haemolysis type, colony colour/size, media, etc.)."
        )
    else:
        top = results[0]
        top3 = results[:3]
        ranked_str = ", ".join([f"**{r.genus}** ({r.confidence_percent()}%)" for r in top3])
        reasoning = top.reasoning_paragraph(ranked_results=results)
        next_tests = top.reasoning_factors.get("next_tests", "")
        reply_lines = [
            f"**Top match:** {top.genus} — {top.confidence_percent()}% (true: {top.true_confidence()}%)",
            f"**Other candidates:** {ranked_str}",
            f"**Why:** {reasoning}",
            f"_Parsed by: {backend_label}_",
        ]
        if next_tests:
            reply_lines.append(f"**Next tests to differentiate:** {next_tests}")
        if top.extra_notes:
            reply_lines.append(f"**Notes:** {top.extra_notes}")
        reply = "\n\n".join(reply_lines)

    # Update session memory
    st.session_state.facts.update({k: v for k, v in parsed.items() if v and v != "Unknown"})
    st.session_state.last_results = results
    st.session_state.history.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)

# ──────────────────────────────────────────────────────────────────────────────
# 🔄 AUTO-GIT COMMIT (Full integration for Streamlit Cloud)
# ──────────────────────────────────────────────────────────────────────────────
def auto_git_commit():
    """Automatically commit parser learning updates and push to GitHub."""
    import subprocess
    from datetime import datetime

    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    branch = os.getenv("GIT_BRANCH", "main")
    email = os.getenv("GIT_USER_EMAIL", os.getenv("GITHUB_EMAIL", "bot@bactaid.local"))
    name = os.getenv("GIT_USER_NAME", os.getenv("GITHUB_NAME", "BactAI-D AutoCommit"))

    if not token or not repo:
        print("⚠️ Missing GitHub credentials (GH_TOKEN/GITHUB_REPO). Skipping auto-commit.")
        return

    try:
        # Configure Git identity & safety
        subprocess.run(["git", "config", "--global", "user.email", email], check=False)
        subprocess.run(["git", "config", "--global", "user.name", name], check=False)
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", os.getcwd()], check=False)

        # Stage relevant files (learning-related & core engine)
        files_to_add = [
            "parser_llm.py",
            "parser_basic.py",
            "parser_memory.json",
            "parser_feedback.json",
            "engine.py",
            "gold_tests.json"
        ]
        for f in files_to_add:
            if os.path.exists(f):
                subprocess.run(["git", "add", f], check=False)

        # Check for staged changes
        status = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True)
        if not status.stdout.strip():
            print("ℹ️ No staged changes to commit.")
            return

        msg = f"🤖 Auto-learn update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", msg], check=False)

        remote_url = f"https://{token}@github.com/{repo}.git"
        subprocess.run(["git", "push", remote_url, f"HEAD:{branch}"], check=True)
        print("✅ Auto-commit and push completed successfully.")
    except Exception as e:
        print(f"⚠️ Auto-commit failed: {e}")


# Auto-run commit after gold test learning or feedback changes
if os.path.exists("parser_memory.json") or os.path.exists("parser_feedback.json"):
    try:
        auto_git_commit()
        st.sidebar.success("✅ Auto-commit to GitHub completed.")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Auto-commit skipped: {e}")
