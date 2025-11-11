# app.py — BactAI-D Main Interface (Aligned with self-learning parsers)
# ---------------------------------------------------------------------
import streamlit as st
import pandas as pd
import re
import os
import json
import subprocess
from fpdf import FPDF
from datetime import datetime

# Core imports
from engine import BacteriaIdentifier
from parser_llm import parse_input_free_text as parse_llm_input_free_text, enable_self_learning_autopatch
from parser_basic import enable_self_learning_autopatch as enable_regex_autopatch

# ──────────────────────────────────────────────────────────────────────────────
# GIT INITIALIZATION (Streamlit Cloud Fix)
# ──────────────────────────────────────────────────────────────────────────────
def ensure_git_repo():
    """Initializes a git repo on Streamlit Cloud if none exists."""
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
            ["git", "remote", "add", "origin", f"https://{gh_token}@github.com/{gh_repo}.git"],
            check=False,
        )
        subprocess.run(["git", "fetch", "origin", branch], check=False)
        subprocess.run(["git", "checkout", "-B", branch], check=False)
        print("✅ Git repo initialized and connected to remote.")

ensure_git_repo()
enable_self_learning_autopatch(run_tests=False)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BactAI-D Assistant", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
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
    st.error(f"Database file not found at '{primary_path}' or '{fallback_path}'.")
    st.stop()

db = load_data(data_path, last_modified)
eng = BacteriaIdentifier(db)

st.sidebar.caption(f"📅 Database last updated: {datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')}")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧫 BactAI-D: Intelligent Bacteria Identification Assistant")
st.markdown("Use the sidebar to input your biochemical and morphological results.")

# ──────────────────────────────────────────────────────────────────────────────
# FIELD GROUPS
# ──────────────────────────────────────────────────────────────────────────────
MORPH_FIELDS = ["Gram Stain", "Shape", "Colony Morphology", "Media Grown On", "Motility", "Capsule", "Spore Formation"]
ENZYME_FIELDS = ["Catalase", "Oxidase", "Coagulase", "Lipase Test"]
SUGAR_FIELDS = [
    "Glucose Fermentation", "Lactose Fermentation", "Sucrose Fermentation", "Maltose Fermentation",
    "Mannitol Fermentation", "Sorbitol Fermentation", "Xylose Fermentation", "Rhamnose Fermentation",
    "Arabinose Fermentation", "Raffinose Fermentation", "Trehalose Fermentation", "Inositol Fermentation"
]

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if "user_input" not in st.session_state:
    st.session_state.user_input = {}
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "gold_results" not in st.session_state:
    st.session_state.gold_results = None
if "gold_summary" not in st.session_state:
    st.session_state.gold_summary = None

# ──────────────────────────────────────────────────────────────────────────────
# RESET TRIGGER
# ──────────────────────────────────────────────────────────────────────────────
if "reset_trigger" in st.session_state and st.session_state["reset_trigger"]:
    for key in list(st.session_state.user_input.keys()):
        st.session_state.user_input[key] = "Unknown"
    for key in list(st.session_state.keys()):
        if key not in ["user_input", "results", "reset_trigger", "gold_results", "gold_summary"]:
            if isinstance(st.session_state[key], list):
                st.session_state[key] = []
            else:
                st.session_state[key] = "Unknown"
    st.session_state["reset_trigger"] = False
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR INPUT SECTIONS
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style='background-color:#1565C0; padding:12px; border-radius:10px;'>
        <h3 style='text-align:center; color:white; margin:0;'>🔬 Input Test Results</h3>
    </div>
    """,
    unsafe_allow_html=True
)

def get_unique_values(field):
    vals = []
    for v in eng.db[field]:
        parts = re.split(r"[;/]", str(v))
        for p in parts:
            clean = p.strip()
            if clean and clean not in vals:
                vals.append(clean)
    vals.sort()
    return vals

with st.sidebar.expander("🧫 Morphological Tests", expanded=True):
    for field in MORPH_FIELDS:
        if field in ["Shape", "Colony Morphology", "Media Grown On"]:
            options = get_unique_values(field)
            selected = st.multiselect(field, options, default=[], key=field)
            st.session_state.user_input[field] = "; ".join(selected) if selected else "Unknown"
        else:
            st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field)

with st.sidebar.expander("🧪 Enzyme Tests", expanded=False):
    for field in ENZYME_FIELDS:
        st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field)

with st.sidebar.expander("🍬 Carbohydrate Fermentation Tests", expanded=False):
    for field in SUGAR_FIELDS:
        st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field)

with st.sidebar.expander("🧬 Other Tests", expanded=False):
    for field in db.columns:
        if field in ["Genus"] + MORPH_FIELDS + ENZYME_FIELDS + SUGAR_FIELDS:
            continue
        if field == "Haemolysis Type":
            options = get_unique_values(field)
            selected = st.multiselect(field, options, default=[], key=field)
            st.session_state.user_input[field] = "; ".join(selected) if selected else "Unknown"
        elif field == "Oxygen Requirement":
            options = get_unique_values(field)
            st.session_state.user_input[field] = st.selectbox(field, ["Unknown"] + options, index=0, key=field)
        elif field == "Growth Temperature":
            st.session_state.user_input[field] = st.text_input(field + " (°C)", "", key=field)
        else:
            st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field)

if st.sidebar.button("🔄 Reset All Inputs"):
    st.session_state["reset_trigger"] = True
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# GOLD SPEC TESTS (SELF-LEARNING)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar.expander("🧪 Gold Spec Tests", expanded=False):
    if st.button("▶️ Run Gold Spec Tests & Self-Learn"):
        with st.spinner("Running Gold Spec Tests and analyzing feedback..."):
            try:
                enable_self_learning_autopatch(run_tests=True, db_fields=[c for c in db.columns if c.lower() != "genus"])
                enable_regex_autopatch(run_tests=True, db_fields=[c for c in db.columns if c.lower() != "genus"])
                st.success("✅ Gold Spec Tests completed and learning applied.")
            except Exception as e:
                st.error(f"Gold Test Learning failed: {e}")

    if st.button("🧹 Clear Learning Memory"):
        for f in ["parser_feedback.json", "parser_memory.json"]:
            if os.path.exists(f):
                os.remove(f)
        st.success("Cleared parser learning memory.")

# ──────────────────────────────────────────────────────────────────────────────
# IDENTIFICATION
# ──────────────────────────────────────────────────────────────────────────────
if st.sidebar.button("🔍 Identify"):
    with st.spinner("Analyzing results..."):
        results = eng.identify(st.session_state.user_input)
        if not results:
            st.error("No matches found.")
        else:
            results = pd.DataFrame(
                [
                    [
                        r.genus,
                        f"{r.confidence_percent()}%",
                        f"{r.true_confidence()}%",
                        r.reasoning_paragraph(results),
                        r.reasoning_factors.get("next_tests", ""),
                        r.extra_notes
                    ]
                    for r in results
                ],
                columns=["Genus", "Confidence", "True Confidence (All Tests)", "Reasoning", "Next Tests", "Extra Notes"],
            )
            st.session_state.results = results

if not st.session_state.results.empty:
    st.info("Percentages based on entered tests. True confidence reflects all fields.")
    for _, row in st.session_state.results.iterrows():
        confidence_value = int(row["Confidence"].replace("%", ""))
        confidence_color = "🟢" if confidence_value >= 75 else "🟡" if confidence_value >= 50 else "🔴"
        header = f"**{row['Genus']}** — {confidence_color} {row['Confidence']}"
        with st.expander(header):
            st.markdown(f"**Reasoning:** {row['Reasoning']}")
            st.markdown(f"**Next Tests:** {row['Next Tests']}")
            st.markdown(f"**True Confidence:** {row['True Confidence (All Tests)']}")
            if row["Extra Notes"]:
                st.markdown(f"**Notes:** {row['Extra Notes']}")

# ──────────────────────────────────────────────────────────────────────────────
# PDF EXPORT
# ──────────────────────────────────────────────────────────────────────────────
def export_pdf(results_df, user_input):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "BactAI-D Identification Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Entered Results:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for k, v in user_input.items():
        pdf.multi_cell(0, 6, f"- {k}: {v}")
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Predictions:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for _, row in results_df.iterrows():
        pdf.multi_cell(0, 6, f"- {row['Genus']} — {row['Confidence']} (True: {row['True Confidence (All Tests)']})")
        pdf.multi_cell(0, 6, f"  Reasoning: {row['Reasoning']}")
        pdf.ln(2)
    pdf.output("BactAI-d_Report.pdf")
    return "BactAI-d_Report.pdf"

if not st.session_state.results.empty:
    if st.button("📄 Export Results to PDF"):
        pdf_path = export_pdf(st.session_state.results, st.session_state.user_input)
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name="BactAI-d_Report.pdf")

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain (Eph)</b></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# AUTO-GIT COMMIT (Streamlit Cloud Safe)
# ──────────────────────────────────────────────────────────────────────────────
def auto_git_commit():
    """Pushes learned updates safely back to GitHub."""
    gh_token = os.getenv("GH_TOKEN")
    gh_repo = os.getenv("GITHUB_REPO")
    branch = os.getenv("GIT_BRANCH", "main")
    email = os.getenv("GIT_USER_EMAIL", "bot@bactaid.local")
    name = os.getenv("GIT_USER_NAME", "BactAI-D AutoLearner")

    if not gh_token or not gh_repo:
        print("⚠️ GitHub credentials not found; skipping auto-commit.")
        return

    try:
        subprocess.run(["git", "config", "--global", "user.email", email], check=False)
        subprocess.run(["git", "config", "--global", "user.name", name], check=False)
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", os.getcwd()], check=False)

        remote_url = f"https://{gh_token}@github.com/{gh_repo}.git"
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=False)

        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not result.stdout.strip():
            print("ℹ️ No local changes to commit — skipping Git push.")
            return

        subprocess.run([
            "git", "add",
            "parser_llm.py", "parser_basic.py",
            "parser_memory.json", "parser_feedback.json"
        ], check=False)

        commit_msg = f"🤖 Auto-learn update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=False)
        subprocess.run(["git", "push", "origin", f"HEAD:{branch}"], check=False)
        print("✅ Auto-commit and push completed.")
    except Exception as e:
        print(f"⚠️ Auto-commit failed: {e}")

auto_git_commit()
st.sidebar.success("✅ Learning cycle complete & synced with GitHub.")