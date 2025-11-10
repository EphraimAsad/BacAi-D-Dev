import streamlit as st
import pandas as pd
import re
import os
import json
from fpdf import FPDF
from datetime import datetime

from engine import BacteriaIdentifier
from parser_llm import parse_input_free_text  # <-- for Gold Spec Tests

# --- CONFIG ---
st.set_page_config(page_title="BactAI-D Assistant", layout="wide")

# --- LOAD DATA with auto-reload when the file changes ---
@st.cache_data
def load_data(path, last_modified):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

# Resolve path (prefer ./data/bacteria_db.xlsx, fallback to ./bacteria_db.xlsx)
primary_path = os.path.join("data", "bacteria_db.xlsx")
fallback_path = os.path.join("bacteria_db.xlsx")
data_path = primary_path if os.path.exists(primary_path) else fallback_path

# Get last modified time (used as cache key so cache invalidates on change)
try:
    last_modified = os.path.getmtime(data_path)
except FileNotFoundError:
    st.error(f"Database file not found at '{primary_path}' or '{fallback_path}'.")
    st.stop()

db = load_data(data_path, last_modified)
eng = BacteriaIdentifier(db)

# Optional: show when the DB was last updated
st.sidebar.caption(f"📅 Database last updated: {datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')}")

# --- PAGE HEADER ---
st.title("🧫 BactAI-D: Intelligent Bacteria Identification Assistant")
st.markdown("Use the sidebar to input your biochemical and morphological results.")

# --- FIELD GROUPS ---
MORPH_FIELDS = ["Gram Stain", "Shape", "Colony Morphology", "Media Grown On", "Motility", "Capsule", "Spore Formation"]
ENZYME_FIELDS = ["Catalase", "Oxidase", "Coagulase", "Lipase Test"]
SUGAR_FIELDS = [
    "Glucose Fermentation", "Lactose Fermentation", "Sucrose Fermentation", "Maltose Fermentation",
    "Mannitol Fermentation", "Sorbitol Fermentation", "Xylose Fermentation", "Rhamnose Fermentation",
    "Arabinose Fermentation", "Raffinose Fermentation", "Trehalose Fermentation", "Inositol Fermentation"
]

# --- SESSION STATE ---
if "user_input" not in st.session_state:
    st.session_state.user_input = {}
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "gold_results" not in st.session_state:
    st.session_state.gold_results = None
if "gold_summary" not in st.session_state:
    st.session_state.gold_summary = None

# --- RESET TRIGGER HANDLER (before widgets are created) ---
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

# --- SIDEBAR HEADER ---
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

# --- SIDEBAR INPUTS ---
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

# --- RESET BUTTON ---
if st.sidebar.button("🔄 Reset All Inputs"):
    st.session_state["reset_trigger"] = True
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# 🧪 GOLD SPEC TESTS — sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar.expander("🧪 Gold Spec Tests", expanded=False):
    # Run tests
    if st.button("▶️ Run Gold Spec Tests"):
        try:
            # Load gold tests file
            with open("gold_tests.json", "r", encoding="utf-8") as f:
                tests = json.load(f)

            # DB fields to clamp outputs to your schema
            db_fields = [c for c in db.columns if c.strip().lower() != "genus"]

            results = []
            feedback = []
            passed = 0

            for case in tests:
                parsed = parse_input_free_text(case["input"], db_fields=db_fields)
                expected = case.get("expected", {})
                mismatched = []

                # Compare only expected keys (gold standard)
                for key, exp_val in expected.items():
                    got_val = parsed.get(key)
                    if got_val != exp_val:
                        mismatched.append({"field": key, "got": got_val, "expected": exp_val})

                if mismatched:
                    results.append({"name": case.get("name", "Unnamed Case"), "status": "❌", "mismatches": mismatched, "parsed": parsed, "expected": expected})
                    feedback.append({"name": case.get("name", "Unnamed Case"), "text": case["input"], "errors": mismatched})
                else:
                    results.append({"name": case.get("name", "Unnamed Case"), "status": "✅", "mismatches": [], "parsed": parsed, "expected": expected})
                    passed += 1

            # Save feedback for the parser’s learning loop
            with open("parser_feedback.json", "w", encoding="utf-8") as fb:
                json.dump(feedback, fb, indent=2)
            from parser_llm import analyze_feedback_and_learn
            analyze_feedback_and_learn("parser_feedback.json", "parser_memory.json")
            from parser_llm import auto_update_parser_regex
            auto_update_parser_regex("parser_memory.json", "parser_llm.py")



            st.session_state.gold_results = results
            st.session_state.gold_summary = (passed, len(tests))
            st.success("Gold Spec Tests finished.")

        except FileNotFoundError:
            st.error("gold_tests.json not found in repo root. Please add it and try again.")
        except Exception as e:
            st.error(f"Gold Test failed to run: {e}")

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

# --- IDENTIFY BUTTON ---
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

# --- DISPLAY RESULTS ---
if not st.session_state.results.empty:
    st.info("Percentages based upon options entered. True confidence percentage shown within each expanded result.")
    for _, row in st.session_state.results.iterrows():
        confidence_value = int(row["Confidence"].replace("%", ""))
        confidence_color = "🟢" if confidence_value >= 75 else "🟡" if confidence_value >= 50 else "🔴"
        header = f"**{row['Genus']}** — {confidence_color} {row['Confidence']}"
        with st.expander(header):
            st.markdown(f"**Reasoning:** {row['Reasoning']}")
            st.markdown(f"**Top 3 Next Tests to Differentiate:** {row['Next Tests']}")
            st.markdown(f"**True Confidence (All Tests):** {row['True Confidence (All Tests)']}")
            if row["Extra Notes"]:
                st.markdown(f"**Notes:** {row['Extra Notes']}")

# --- GOLD TEST RESULTS DISPLAY (main area) ---
if st.session_state.gold_results is not None:
    passed, total = st.session_state.gold_summary or (0, 0)
    st.markdown("## 🧪 Gold Test Results")
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

# --- PDF EXPORT ---
def export_pdf(results_df, user_input):
    def safe_text(text):
        """Convert text to Latin-1 safe characters."""
        text = str(text).replace("•", "-").replace("—", "-").replace("–", "-")
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "BactAI-d Identification Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Entered Test Results:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for k, v in user_input.items():
        pdf.multi_cell(0, 6, safe_text(f"- {k}: {v}"))

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Top Possible Matches:", ln=True)
    pdf.set_font("Helvetica", "", 10)

    for _, row in results_df.iterrows():
        pdf.multi_cell(0, 7, safe_text(f"- {row['Genus']} — Confidence: {row['Confidence']} (True: {row['True Confidence (All Tests)']})"))
        pdf.multi_cell(0, 6, safe_text(f"  Reasoning: {row['Reasoning']}"))
        if row['Next Tests']:
            pdf.multi_cell(0, 6, safe_text(f"  Next Tests: {row['Next Tests']}"))
        if row['Extra Notes']:
            pdf.multi_cell(0, 6, safe_text(f"  Notes: {row['Extra Notes']}"))
        pdf.ln(3)

    pdf.output("BactAI-d_Report.pdf")
    return "BactAI-d_Report.pdf"

if not st.session_state.results.empty:
    if st.button("📄 Export Results to PDF"):
        pdf_path = export_pdf(st.session_state.results, st.session_state.user_input)
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name="BactAI-d_Report.pdf")

# --- FOOTER ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain</b> | www.linkedin.com/in/zain-asad-1998EPH</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# 🔁 AUTO-GIT COMMIT (Streamlit Cloud safe version)
# ────────────────────────────────────────────────────────────────
import subprocess

def auto_git_commit():
    """Safely commits parser updates to GitHub if changes occurred."""
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    email = os.getenv("GITHUB_EMAIL", "bot@bactaid.local")
    name = os.getenv("GITHUB_NAME", "BactAI-D AutoLearner")

    if not token or not repo:
        print("⚠️ GitHub credentials not found; skipping auto-commit.")
        return

    try:
        # Configure git identity
        subprocess.run(["git", "config", "--global", "user.email", email], check=True)
        subprocess.run(["git", "config", "--global", "user.name", name], check=True)

        # Check for modifications first
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not result.stdout.strip():
            print("ℹ️ No local changes to commit — skipping Git push.")
            return

        # Stage modified learning-related files only
        subprocess.run([
            "git", "add",
            "parser_llm.py", "parser_basic.py",
            "parser_memory.json", "parser_feedback.json"
        ], check=False)

        # Commit with timestamp
        commit_msg = f"🤖 Auto-learned regex update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=False)

        # Push using the token
        remote_url = f"https://{token}@github.com/{repo}.git"
        subprocess.run(["git", "push", remote_url, "HEAD:main"], check=True)

        print("✅ Successfully committed & pushed updates to GitHub.")
    except Exception as e:
        print(f"⚠️ Auto-commit failed: {e}")

# Call after learning updates
auto_git_commit()


