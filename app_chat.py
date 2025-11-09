# app_chat.py
import os
from datetime import datetime
import pandas as pd
import streamlit as st

from engine import BacteriaIdentifier
from parser_basic import parse_input_free_text

# ---- CONFIG ----
st.set_page_config(page_title="BactAI-D — Chat", layout="wide")

# ---- LOAD DATA (same logic as your app.py) ----
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
    st.error(f"Database file not found at '{primary_path}' or '{fallback_path}'.")
    st.stop()

db = load_data(data_path, last_modified)
eng = BacteriaIdentifier(db)

st.sidebar.caption(f"📅 DB last updated: {datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')}")
st.title("🧫 BactAI-D — Language Reasoning (Step 1: Chat)")

st.markdown(
    "Describe your findings in plain English (e.g., "
    "_'Gram negative rod, oxidase positive, motile, non-lactose fermenter on MacConkey.'_). "
    "I’ll parse it, run the BactAI-D engine, and explain the result."
)

# ---- Session memory ----
if "facts" not in st.session_state:
    st.session_state.facts = {}
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"role":"user|assistant", "content": str}
if "last_results" not in st.session_state:
    st.session_state.last_results = None

# Right-side inspector
with st.sidebar.expander("🧪 Parsed Fields (editable soon)", expanded=True):
    if st.session_state.facts:
        st.json(st.session_state.facts)
    else:
        st.caption("No parsed fields yet. Send a message to begin.")
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset conversation"):
    st.session_state.facts = {}
    st.session_state.history = []
    st.session_state.last_results = None
    st.rerun()

# Render previous messages
for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
prompt = st.chat_input("Tell me your observations…")
if prompt:
    # 1) Show user message
    st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # 2) Parse to engine fields (merge with prior facts so conversation is additive)
    parsed = parse_input_free_text(prompt, prior_facts=st.session_state.facts)

    # 3) Run engine
    results = eng.identify(parsed)

    if not results:
        reply = "I couldn't find a good match with the current information. Try adding a few more traits or tests."
    else:
        # Top-3 overview
        top = results[0]
        top3 = results[:3]

        # Build a compact, friendly response using your engine's info
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

    # 4) Update memory and show reply
    st.session_state.facts.update({k: v for k, v in parsed.items() if v and v != 'Unknown'})
    st.session_state.last_results = results
    st.session_state.history.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)
