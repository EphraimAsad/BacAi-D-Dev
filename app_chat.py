# app_chat.py — Step 1.6 (Dynamic Schema)
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from engine import BacteriaIdentifier
from parser_llm import parse_input_free_text  # Now dynamic schema

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

- **Confidence** – Based **only on the tests you entered**. It reflects matches across the fields you actually provided.  
- **True Confidence (All Tests)** – The same internal score scaled by **all possible fields in the database**.
"""
)

# ---- SESSION STATE ----
if "facts" not in st.session_state:
    st.session_state.facts = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None

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

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset conversation"):
    st.session_state.facts = {}
    st.session_state.history = []
    st.session_state.last_results = None
    st.rerun()

# ---- CHAT HISTORY ----
for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ---- CHAT INPUT ----
user_msg = st.chat_input("Tell me your observations…")
if user_msg:
    st.session_state.history.append({"role": "user", "content": user_msg})
    st.chat_message("user").markdown(user_msg)

    # ⬇️ NEW: Pass db_fields dynamically to parser
    parsed = parse_input_free_text(user_msg, prior_facts=st.session_state.facts, db_fields=db_fields)

    results = eng.identify(parsed)

    if not results:
        reply = (
            "I couldn't find a good match with the current information. "
            "Try adding a few more traits or tests (see **Supported Tests** in the sidebar)."
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

    st.session_state.facts.update({k: v for k, v in parsed.items() if v and v != "Unknown"})
    st.session_state.last_results = results
    st.session_state.history.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)

# ---- FOOTER ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain (Eph)</b></div>", unsafe_allow_html=True)
