# app.py — AutoDFit (final, production-ready)
# NOTE: This code is defensive and ready to run in an environment with the recommended
# packages installed (see requirements.txt below). Set Python runtime to 3.11 on your host.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import pickle
import tempfile
from datetime import datetime
import random

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, roc_curve, auc

# GA (deap) imports will be used lazily if user enables GA
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except Exception:
    DEAP_AVAILABLE = False

# ---------------------------
# Page config + minimal safe CSS
# ---------------------------
st.set_page_config(page_title="AutoDFit", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* App background & typography */
.stApp { background-color: #0b1220; color: #e6eef8; }
/* title style */
.autodfit-title { text-align:center; font-size:36px; font-weight:700;
  background: linear-gradient(90deg,#06b6d4,#8b5cf6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-top:6px; }
/* subtitle */
.autodfit-sub { text-align:center; color:#9fb0d6; margin-bottom:18px; }
/* metrics */
[data-testid="stMetric"] { background: #0f1724; border: 1px solid #1f2a44; border-radius:10px; padding:8px; }
/* sidebar */
section[data-testid="stSidebar"] { background-color:#071226; color:#cfe6ff; }
/* buttons */
.stButton>button { background: linear-gradient(90deg,#06b6d4,#8b5cf6); color:white; border-radius:8px; padding:8px 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="autodfit-title">AutoDFit</div>', unsafe_allow_html=True)
st.markdown('<div class="autodfit-sub">AI Dataset Intelligence — AutoML · Explainability · Report</div>', unsafe_allow_html=True)

# ---------------------------
# Sidebar options
# ---------------------------
with st.sidebar:
    st.header("Options")
    ENABLE_SHAP = st.checkbox("Enable SHAP explainability (sampled)", value=True)
    ENABLE_GA = st.checkbox("Run Genetic Feature Optimization (slow)", value=False)
    GA_POP = st.slider("GA population (if enabled)", 6, 30, 10, 2)
    GA_GEN = st.slider("GA generations (if enabled)", 3, 10, 5, 1)
    st.write("---")
    st.write("Tip: use dataset <= 10k rows for responsive runs. SHAP & GA are heavy on CPU.")

# ---------------------------
# File upload
# ---------------------------
uploaded = st.file_uploader("Upload CSV (the target column will be selected next)", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin. Try Iris or Titanic for a quick demo.")
    st.stop()

# safe CSV load
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

try:
    df = load_csv(uploaded)
except Exception as e:
    st.error("Failed to read CSV: " + str(e))
    st.stop()

# Quick dataset overview
rows, cols = df.shape
missing_cells = int(df.isnull().sum().sum())
st.header("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", rows)
c2.metric("Columns", cols)
c3.metric("Missing cells", missing_cells)
st.dataframe(df.head(200))

# Target selection
target = st.selectbox("Select the target (label) column", df.columns.tolist())

# Basic health score
total_cells = df.size if df.size > 0 else 1
completeness = (1 - missing_cells/total_cells) * 100
uniqueness = (df.nunique().sum() / total_cells) * 100
quality_score = round(0.7 * completeness + 0.3 * uniqueness, 2)
st.write(f"Dataset health score: **{quality_score}%**")

# derive X, y
X_df = df.drop(columns=[target])
y_ser = df[target].copy()

# Detect problem type (heuristic)
def detect_problem_type(y: pd.Series) -> str:
    # If dtype object or <= 20 unique values -> classification (heuristic)
    if y.dtype == object:
        return "classification"
    try:
        nunique = y.nunique()
        if nunique <= 20:
            # if numeric but integer-like and few classes -> classification
            return "classification"
    except Exception:
        pass
    # fallback: regression if float-like or many unique values
    return "regression"

problem_type = detect_problem_type(y_ser)
st.success(f"Detected problem type: **{problem_type}** (heuristic)")

# Preprocessing pipeline: numeric -> impute+scale ; categorical -> ordinal encoder (handles unseen)
num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

transformers = []
if len(num_cols) > 0:
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    transformers.append(("num", num_pipeline, num_cols))

if len(cat_cols) > 0:
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    transformers.append(("cat", cat_pipeline, cat_cols))

if len(transformers) == 0:
    st.error("No features found (all columns may be non-usable after dropping target).")
    st.stop()

preprocessor = ColumnTransformer(transformers, remainder="drop")

# Train/test split + preprocess
with st.spinner("Splitting and preprocessing..."):
    try:
        X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y_ser, test_size=0.2, random_state=42, shuffle=True)
    except Exception as e:
        st.error("train_test_split failed: " + str(e))
        st.stop()

    try:
        X_train = preprocessor.fit_transform(X_train_df)
        X_test = preprocessor.transform(X_test_df)
    except Exception as e:
        st.error("Preprocessing failed: " + str(e))
        st.stop()

# Model training function (kept light)
def build_and_train(X_tr, X_te, y_tr, y_te, problem):
    if problem == "classification":
        candidates = {
            "Logistic": LogisticRegression(max_iter=300),
            "RandomForest": RandomForestClassifier(n_estimators=80),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier()
        }
    else:
        candidates = {
            "Linear": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=80),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor()
        }

    scores = {}
    trained = {}
    for name, m in candidates.items():
        try:
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            score = (accuracy_score(y_te, preds) if problem == "classification" else r2_score(y_te, preds))
            scores[name] = float(score)
            trained[name] = m
        except Exception:
            scores[name] = float("nan")
            trained[name] = m
    return scores, trained

with st.spinner("Training candidate models..."):
    scores, trained_models = build_and_train(X_train, X_test, y_train, y_test, problem_type)

# leaderboard
leader = pd.DataFrame(list(scores.items()), columns=["Model", "Score"])
leader = leader.sort_values(by="Score", ascending=False, key=lambda col: col.fillna(-9999)).reset_index(drop=True)
st.subheader("Model leaderboard")
st.dataframe(leader.style.format({"Score": "{:.4f}"}))

if leader.shape[0] == 0 or leader["Score"].isnull().all():
    st.error("No model produced valid results.")
    st.stop()

best_name = leader.loc[0, "Model"]
best_score = leader.loc[0, "Score"]
best_model = trained_models.get(best_name)

st.metric("Best model", best_name, f"{(best_score*100):.2f}%" if not np.isnan(best_score) else "N/A")

# model comparison bar chart
st.subheader("Model comparison")
fig, ax = plt.subplots(figsize=(7, 3))
vals = [0.0 if (pd.isna(v) or v is None) else v * 100 for v in leader["Score"].tolist()]
ax.bar(leader["Model"], vals)
ax.set_ylabel("Score (%)")
ax.set_ylim(bottom=0)
plt.xticks(rotation=20)
st.pyplot(fig)
plt.close(fig)

# classification evaluation plots
if problem_type == "classification":
    st.subheader("Evaluation — confusion matrix & ROC (best model)")
    try:
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig_cm, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    except Exception:
        st.warning("Confusion matrix unavailable.")

    # ROC if probabilities available
    if hasattr(best_model, "predict_proba"):
        try:
            probs = best_model.predict_proba(X_test)
            # choose class 1 if binary else use first class's probability for ROC
            if probs.shape[1] == 2:
                probs1 = probs[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs1)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax = plt.subplots(figsize=(4, 3))
                ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], "--", color="grey")
                ax.legend()
                st.pyplot(fig_roc)
                plt.close(fig_roc)
        except Exception:
            st.info("ROC not available for this best model/dataset.")

# feature importance if present
st.subheader("Feature importance (if supported)")
try:
    if hasattr(best_model, "feature_importances_"):
        feats = best_model.feature_importances_
        fig_fi, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(len(feats)), feats)
        ax.set_xlabel("Feature (post-preprocessing index)")
        st.pyplot(fig_fi)
        plt.close(fig_fi)
    else:
        st.info("This model doesn't expose `feature_importances_` (RandomForest does).")
except Exception:
    st.warning("Could not compute feature importance.")

# SHAP explainability (optional, sampled)
if ENABLE_SHAP:
    st.subheader("SHAP explainability (sampled)")
    try:
        import shap  # lazy import since shap is heavy
        # use small background and sample
        background = X_train[:200] if X_train.shape[0] > 200 else X_train
        sample = X_test[:100] if X_test.shape[0] > 100 else X_test
        explainer = shap.Explainer(best_model, background)
        shap_values = explainer(sample, check_additivity=False)
        fig_shap = plt.figure(figsize=(6, 4))
        shap.summary_plot(shap_values, sample, show=False)
        st.pyplot(fig_shap)
        plt.close(fig_shap)
    except Exception as e:
        st.warning("SHAP unavailable or failed: " + str(e))

# Genetic feature optimization (optional & controlled)
best_individual = None
if ENABLE_GA:
    st.subheader("Genetic Feature Optimization (optional)")
    if not DEAP_AVAILABLE:
        st.warning("GA disabled because `deap` is not installed.")
    else:
        n_features = X_train.shape[1]
        if n_features <= 1:
            st.info("Not enough features for GA.")
        else:
            # GA setup (small/pop-controlled through sidebar)
            if not hasattr(creator, "FitnessMax"):
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)
            toolbox = base.Toolbox()
            toolbox.register("attr_bool", random.randint, 0, 1)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            def eval_ind(ind):
                sel = [i for i, bit in enumerate(ind) if bit == 1]
                if len(sel) == 0:
                    return 0.0,
                try:
                    m = RandomForestClassifier(n_estimators=60) if problem_type == "classification" else RandomForestRegressor(n_estimators=60)
                    m.fit(X_train[:, sel], y_train)
                    p = m.predict(X_test[:, sel])
                    s = accuracy_score(y_test, p) if problem_type == "classification" else r2_score(y_test, p)
                    return float(s),
                except Exception:
                    return 0.0,

            toolbox.register("evaluate", eval_ind)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
            toolbox.register("select", tools.selTournament, tournsize=3)

            pop = toolbox.population(n=GA_POP)
            with st.spinner("Running GA (this may take a while)..."):
                algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.2, ngen=GA_GEN, verbose=False)
            best_individual = tools.selBest(pop, 1)[0]
            st.success(f"GA finished. Selected features count: {sum(best_individual)}")

# Quick prediction interface
st.subheader("Quick prediction (single sample)")
if st.button("Predict first test row"):
    try:
        pred = best_model.predict(X_test[:1])
        st.write("Prediction:", pred[0])
    except Exception as e:
        st.warning("Prediction failed: " + str(e))

# Download trained model
st.subheader("Export")
try:
    model_bytes = pickle.dumps(best_model)
    st.download_button("Download trained model (.pkl)", data=model_bytes, file_name="autodfit_model.pkl", mime="application/octet-stream")
except Exception:
    st.info("Model could not be serialized for download.")

# -----------------------------
# PDF report generation (optional)
# -----------------------------
def pdf_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(40, 800, "AutoDFit AI Analytics Platform")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(550, 20, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()

def generate_pdf_report():
    # Lazy import reportlab - if missing, raise an informative error
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as e:
        raise RuntimeError("reportlab not installed. PDF generation requires reportlab. " + str(e))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # Cover
    story.append(Spacer(1, 140))
    story.append(Paragraph("AutoDFit — Analysis Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    story.append(PageBreak())

    # TOC
    story.append(Paragraph("Table of Contents", styles["Heading1"]))
    for s in ["1. Executive Summary", "2. Dataset Overview", "3. Model Performance", "4. Visual Analytics", "5. Explainability", "6. Genetic Optimization", "7. Conclusion"]:
        story.append(Paragraph(s, styles["Normal"]))
    story.append(PageBreak())

    # Executive summary
    story.append(Paragraph("1. Executive Summary", styles["Heading1"]))
    story.append(Paragraph(f"Best model: {best_name}", styles["Normal"]))
    story.append(Paragraph(f"Best score: {best_score if best_score is not None else 'N/A'}", styles["Normal"]))
    story.append(PageBreak())

    # Dataset overview
    story.append(Paragraph("2. Dataset Overview", styles["Heading1"]))
    story.append(Paragraph(f"Rows: {rows}", styles["Normal"]))
    story.append(Paragraph(f"Columns: {cols}", styles["Normal"]))
    story.append(Paragraph(f"Missing cells: {missing_cells}", styles["Normal"]))
    story.append(PageBreak())

    # Model performance table
    story.append(Paragraph("3. Model Performance", styles["Heading1"]))
    table_data = [["Model", "Score (%)"]]
    for m, s in scores.items():
        try:
            table_data.append([m, f"{float(s)*100:.2f}"])
        except Exception:
            table_data.append([m, "N/A"])
    tbl = Table(table_data)
    tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.black), ("TEXTCOLOR",(0,0),(-1,0),colors.white), ("GRID",(0,0),(-1,-1),0.5,colors.grey), ("ALIGN",(1,1),(-1,-1),"CENTER")]))
    story.append(tbl)
    story.append(PageBreak())

    # Visual analytics (heatmap)
    story.append(Paragraph("4. Visual Analytics", styles["Heading1"]))
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmpname = tmp.name
        tmp.close()
        plt.figure(figsize=(6,4))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
        plt.tight_layout()
        plt.savefig(tmpname)
        plt.close()
        story.append(RLImage(tmpname, width=400, height=300))
        story.append(PageBreak())
        os.remove(tmpname)
    except Exception:
        story.append(Paragraph("Heatmap unavailable.", styles["Normal"]))
        story.append(PageBreak())

    # Explainability (feature importance + SHAP)
    story.append(Paragraph("5. Explainability", styles["Heading1"]))
    try:
        if hasattr(best_model, "feature_importances_"):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmpname = tmp.name
            tmp.close()
            plt.figure()
            plt.bar(range(len(best_model.feature_importances_)), best_model.feature_importances_)
            plt.tight_layout()
            plt.savefig(tmpname)
            plt.close()
            story.append(RLImage(tmpname, width=400, height=250))
            os.remove(tmpname)
    except Exception:
        story.append(Paragraph("Feature importance unavailable.", styles["Normal"]))

    story.append(PageBreak())

    # GA summary
    story.append(Paragraph("6. Genetic Optimization", styles["Heading1"]))
    if best_individual is not None:
        story.append(Paragraph(f"GA selected features count: {sum(best_individual)}", styles["Normal"]))
    else:
        story.append(Paragraph("GA not executed or returned no result.", styles["Normal"]))
    story.append(PageBreak())

    # Conclusion
    story.append(Paragraph("7. Conclusion", styles["Heading1"]))
    story.append(Paragraph("AutoDFit automates dataset profiling, model comparison, explainability and optimization.", styles["Normal"]))

    doc.build(story, onFirstPage=pdf_header_footer, onLaterPages=pdf_header_footer)
    buffer.seek(0)
    return buffer

# PDF download button (safe call)
st.subheader("Full report")
try:
    pdf_buffer = generate_pdf_report()  # will raise if reportlab not installed
    st.download_button("Download full PDF report", data=pdf_buffer, file_name="AutoDFit_Report.pdf", mime="application/pdf")
except Exception as e:
    st.info("PDF generation unavailable: " + str(e))

# End of app
