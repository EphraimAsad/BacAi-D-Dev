# app_chat.py — Step 1.6 (Dynamic Schema + Gold Spec Tests)
import os
import re
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from engine import BacteriaIdentifier
from parser_llm import parse_input_free_text  # Dynamic schema + feedback loop

# ---- PAGE CONFIG ----
st.set_page_config(page_title="BactAI-D — Language Reasoning (Chat)", layout="wide")

# ---- LOAD DATA ----
@st.cache_data
def load_data(path, last_modified):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

# Resolve data path
primary_path = os.path.join("data", "bacteria_db.xlsx")
fallback_path = os.path.join("bacteria_db.xlsx")
data_path = primary_path if os.path.exists(primary_path) else fallback_path

try:
    last_modified = os.path.getmtime(data_path)
except FileNotFoundError:
    st.error(f"Database file not found at '{primary_path}' or '{fallback_path}'.")
    st.stop()

db = load_data(data_path, last_modified)
eng = BacteriaIdentifier(db)
db_fields = [c for c in db.columns if c.lower() != "genus"]

# ---- HEADER ----
st.title("🧫 BactAI-D — Language Reasoning (Chat)")
st.caption(f"📅 Database last updated: {datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown(
    "Describe your findings in plain English (e.g., "
    "_'Gram negative rod, oxidase positive, motile, non-lactose fermenter on MacConkey.'_). "
    "I’ll parse it, run the BactAI-D engine, and explain the result."
)
st.markdown(
    """
**How to read the scores**

- **Confidence** – Based **only on the tests you entered**.  
- **True Confidence (All Tests)** – The same score scaled by **all possible fields in the database**.
"""
)

# ---- SESSION STATE ----
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

# ---- SIDEBAR: PARSER SETTINGS ----
st.sidebar.markdown("### 🧠 Parser Settings")
parser_choice = st.sidebar.radio("Choose Parser Model", ["Local Llama (via Ollama)", "GPT (Cloud)"], index=0)
os.environ["BACTAI_MODEL"] = "gpt" if "GPT" in parser_choice else "local"
os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
os.environ["LOCAL_MODEL"] = "llama3"

with st.sidebar.expander("🧩 Active Models (info)", expanded=False):
    st.text_input("OpenAI model", value=os.environ["OPENAI_MODEL"], disabled=True)
    st.text_input("Local model (Ollama)", value=os.environ["LOCAL_MODEL"], disabled=True)

st.sidebar.markdown("---")

# ---- SIDEBAR: PARSED FIELDS ----
with st.sidebar.expander("🧪 Parsed Fields (from conversation)", expanded=True):
    if st.session_state.facts:
        st.json(st.session_state.facts)
    else:
        st.caption("No parsed fields yet. Send a message to begin.")

# ---- SIDEBAR: SUPPORTED TESTS ----
with st.sidebar.expander("🧬 Supported Tests (current database fields)", expanded=False):
    st.write(", ".join(sorted(db_fields)))

# ---- SIDEBAR: RESET & GOLD TESTS ----
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset conversation"):
    st.session_state.facts = {}
    st.session_state.history = []
    st.session_state.last_results = None
    st.session_state.gold_results = None
    st.session_state.gold_summary = None
    st.rerun()

# 🧪 GOLD SPEC TESTS
with st.sidebar.expander("🧪 Gold Spec Tests (Parser Validation)", expanded=False):
    # Run test suite
    if st.button("▶️ Run Gold Spec Tests"):
        try:
            with open("gold_tests.json", "r", encoding="utf-8") as f:
                tests = json.load(f)

            results = []
            feedback = []
            passed = 0

            for case in tests:
                parsed = parse_input_free_text(case["input"], db_fields=db_fields)
                expected = case.get("expected", {})
                mismatched = []
                for key, exp_val in expected.items():
                    got_val = parsed.get(key)
                    if got_val != exp_val:
                        mismatched.append({"field": key, "got": got_val, "expected": exp_val})

                if mismatched:
                    results.append({"name": case["name"], "status": "❌", "mismatches": mismatched, "parsed": parsed, "expected": expected})
                    feedback.append({"name": case["name"], "text": case["input"], "errors": mismatched})
                else:
                    results.append({"name": case["name"], "status": "✅", "mismatches": [], "parsed": parsed, "expected": expected})
                    passed += 1

            # Save feedback file
            with open("parser_feedback.json", "w", encoding="utf-8") as fb:
                json.dump(feedback, fb, indent=2)

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

# ---- DISPLAY GOLD TEST RESULTS ----
if st.session_state.gold_results is not None:
    passed, total = st.session_state.gold_summary or (0, 0)
    st.markdown("## 🧪 Gold Test Results (Chat)")
    st.markdown(f"**Summary:** {passed}/{total} passed")
    for r in st.session_state.gold_results:
        if r["status"] == "✅":
            st.markdown(f"**{r['name']}** — ✅ Passed all checks")
        else:
            st.markdown(f"**{r['name']}** — ❌ Mismatches:")
            for m in r["mismatches"]:
                st.markdown(f"- **{m['field']}**: got `{m['got']}`, expected `{m['expected']}`")
            with st.expander("Show parsed vs expected"):
                st.code(json.dumps({"parsed": r["parsed"], "expected": r["expected"]}, indent=2), language="json")

st.sidebar.markdown("---")

# ---- CHAT HISTORY ----
for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ---- CHAT INPUT ----
user_msg = st.chat_input("Tell me your observations…")
if user_msg:
    st.session_state.history.append({"role": "user", "content": user_msg})
    st.chat_message("user").markdown(user_msg)

    # ⬇️ Parse free text using dynamic schema + learning feedback
    parsed = parse_input_free_text(user_msg, prior_facts=st.session_state.facts, db_fields=db_fields)
    results = eng.identify(parsed)

    if not results:
        reply = (
            "I couldn't find a good match with the current information. "
            "Try adding more descriptive test results or mention colony colour, growth, etc."
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

# ---- FOOTER ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain (Eph)</b></div>", unsafe_allow_html=True)
