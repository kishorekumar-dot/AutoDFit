# app.py — AutoDFit (final, production-ready, DEAP removed; pure-Python GA)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import io
import random
import pickle
import os
import copy
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, roc_curve, auc

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    PageBreak, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------
# Streamlit page config & header
# ---------------------------
st.set_page_config(page_title="AutoDFit", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <div style="text-align:center; margin-top:12px;">
      <svg width="240" height="80" viewBox="0 0 260 90" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="g1" x1="0" x2="1">
            <stop stop-color="#00F5FF" offset="0%"/>
            <stop stop-color="#7B61FF" offset="100%"/>
          </linearGradient>
        </defs>
        <circle cx="52" cy="45" r="20" fill="url(#g1)">
          <animate attributeName="r" values="16;24;16" dur="2s" repeatCount="indefinite"/>
        </circle>
        <text x="95" y="54" font-family="Arial" font-size="30" font-weight="700" fill="#111827">AutoDFit</text>
      </svg>
    </div>
    """,
    unsafe_allow_html=True
)
st.title("AutoDFit — AI Dataset Intelligence Platform")
st.caption("Automated dataset fitness, model comparison, explainability and professional reporting")

# ---------------------------
# Caching helpers
# ---------------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data
def quick_data_stats(df: pd.DataFrame):
    rows, cols = df.shape
    missing = int(df.isnull().sum().sum())
    total = df.size if (df.size) else 1
    completeness = (1 - missing/total) * 100
    uniqueness = (df.nunique().sum()/total) * 100
    quality_score = round(completeness*0.7 + uniqueness*0.3, 2)
    return rows, cols, missing, quality_score

@st.cache_resource
def train_models_cached(X_train, X_test, y_train, y_test, problem):
    if problem == "classification":
        models = {
            "Logistic": LogisticRegression(max_iter=300),
            "RandomForest": RandomForestClassifier(n_estimators=80),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier()
        }
    else:
        models = {
            "Linear": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=80),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor()
        }

    scores = {}
    trained = {}
    for name, m in models.items():
        try:
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            score = accuracy_score(y_test, pred) if problem == "classification" else r2_score(y_test, pred)
            scores[name] = float(score) if score is not None else float("nan")
            trained[name] = m
        except Exception:
            scores[name] = float("nan")
            trained[name] = m
    return scores, trained

# ---------------------------
# Sidebar options
# ---------------------------
with st.sidebar:
    st.header("Options")
    show_shap_checkbox = st.checkbox("Enable SHAP explainability (sampled)", value=True)
    run_ga_checkbox = st.checkbox("Run Genetic Feature Optimization (slow)", value=False)
    ga_pop = st.slider("GA population (if enabled)", min_value=6, max_value=30, value=10, step=2)
    ga_gen = st.slider("GA generations (if enabled)", min_value=3, max_value=8, value=5, step=1)
    st.write("Tip: use small datasets for faster runs (<= 10000 rows)")

# ---------------------------
# File upload
# ---------------------------
uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to start. Example: Iris, Titanic, etc.")
    st.stop()

df = load_csv(uploaded)

# quick stats
rows, cols, missing, quality_score = quick_data_stats(df)
st.header("Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", rows)
c2.metric("Cols", cols)
c3.metric("Missing cells", missing)
st.write(f"Dataset health score: **{quality_score}%**")
st.dataframe(df.head())

# correlation (safe)
st.subheader("Correlation heatmap")
fig_corr = plt.figure(figsize=(6,4))
try:
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
    st.pyplot(fig_corr)
except Exception:
    st.warning("Correlation heatmap could not be computed for this dataset.")
plt.close(fig_corr)

# target selection
target_col = st.selectbox("Select target column", df.columns)
X_df = df.drop(columns=[target_col])
y_ser = df[target_col]

# detect problem type
problem_type = "classification" if (y_ser.dtype == "object" or y_ser.nunique() < 20) else "regression"
st.write(f"Detected problem type: **{problem_type}**")

# create preprocessing pipeline dynamically
num_cols = X_df.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_df.select_dtypes(exclude=np.number).columns.tolist()

transformers = []
if len(num_cols) > 0:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    transformers.append(("num", num_pipeline, num_cols))

if len(cat_cols) > 0:
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    transformers.append(("cat", cat_pipeline, cat_cols))

if len(transformers) == 0:
    st.error("No valid features found in dataset after dropping target. Cannot proceed.")
    st.stop()

preprocessor = ColumnTransformer(transformers, remainder="drop")

# split + preprocess
with st.spinner("Preparing data (split + preprocess)..."):
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y_ser, test_size=0.2, random_state=42)
    try:
        X_train = preprocessor.fit_transform(X_train_df)
        X_test = preprocessor.transform(X_test_df)
    except Exception as e:
        st.error("Preprocessing failed: " + str(e))
        st.stop()

# Train models
with st.spinner("Training models (this may take a moment)..."):
    scores, trained_models = train_models_cached(X_train, X_test, y_train, y_test, problem_type)

# Leaderboard and best model
leader_df = pd.DataFrame(list(scores.items()), columns=["Model", "Score"])
# handle NaNs and sort (descending)
leader_df["Score_filled"] = leader_df["Score"].fillna(-9999)
leader_df = leader_df.sort_values("Score_filled", ascending=False).drop(columns=["Score_filled"])
st.subheader("Model leaderboard")
st.dataframe(leader_df.style.format({"Score": "{:.4f}"}))

if leader_df.shape[0] == 0 or leader_df["Score"].isnull().all():
    st.error("No trained models produced valid scores.")
    st.stop()

best_model_name = leader_df.iloc[0]["Model"]
best_score = leader_df.iloc[0]["Score"]
best_model_obj = trained_models.get(best_model_name)

# show best model metric in metric control
try:
    display_val = f"{float(best_score*100):.2f}%" if best_score is not None and not np.isnan(best_score) else "N/A"
    st.metric("Best model", best_model_name, display_val)
except Exception:
    st.metric("Best model", best_model_name)

# Model comparison chart
st.subheader("Model comparison chart")
fig_bar = plt.figure(figsize=(7,3))
scores_numeric = [s if (s is not None and not (isinstance(s, float) and np.isnan(s))) else 0.0 for s in leader_df["Score"].tolist()]
plt.bar(leader_df["Model"], np.array(scores_numeric) * 100)
plt.ylabel("Score (%)")
plt.xticks(rotation=25)
st.pyplot(fig_bar)
plt.close(fig_bar)

# If classification: confusion and ROC
if problem_type == "classification":
    st.subheader("Evaluation: Confusion matrix & ROC (best model)")
    try:
        preds = best_model_obj.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig_cm = plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    except Exception:
        st.warning("Could not compute confusion matrix.")

    if hasattr(best_model_obj, "predict_proba"):
        try:
            probs = best_model_obj.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            fig_roc = plt.figure(figsize=(4,3))
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], "--")
            plt.legend()
            st.pyplot(fig_roc)
            plt.close(fig_roc)
        except Exception:
            st.info("ROC could not be computed for this model/dataset.")

# Feature importance
st.subheader("Feature importance (if supported by model)")
if hasattr(best_model_obj, "feature_importances_"):
    try:
        feats = best_model_obj.feature_importances_
        fig_feat = plt.figure(figsize=(6,3))
        plt.bar(range(len(feats)), feats)
        plt.xlabel("Feature index (post-preprocessing)")
        st.pyplot(fig_feat)
        plt.close(fig_feat)
    except Exception:
        st.info("Could not display feature importance.")
else:
    st.info("Selected model doesn't expose `feature_importances_` (try RandomForest).")

# SHAP explainability
if show_shap_checkbox:
    st.subheader("SHAP explainability (sampled)")
    try:
        background = X_train[:200] if X_train.shape[0] > 200 else X_train
        sample = X_test[:100] if X_test.shape[0] > 100 else X_test
        explainer = shap.Explainer(best_model_obj, background)
        shap_values = explainer(sample, check_additivity=False)
        fig_shap = plt.figure(figsize=(6,4))
        shap.summary_plot(shap_values, sample, show=False)
        st.pyplot(fig_shap)
        plt.close(fig_shap)
    except Exception as e:
        st.warning("SHAP explanation couldn't run reliably for this model/dataset: " + str(e))

# -----------------------------
# Pure-Python Genetic Algorithm
# -----------------------------
def run_simple_ga(eval_fn, n_features, pop_size=10, gens=5, cx_prob=0.6, mut_prob=0.2, tourn_k=3):
    # init population
    pop = [[1 if random.random() < 0.5 else 0 for _ in range(n_features)] for _ in range(pop_size)]

    def fitness(ind):
        return eval_fn(ind)[0]

    def tournament_select(pop, fitnesses):
        idxs = random.sample(range(len(pop)), k=min(tourn_k, len(pop)))
        best = max(idxs, key=lambda i: fitnesses[i])
        return copy.deepcopy(pop[best])

    def two_point_crossover(a, b):
        if len(a) < 2:
            return copy.deepcopy(a), copy.deepcopy(b)
        i, j = sorted(random.sample(range(len(a)), 2))
        child1 = a[:i] + b[i:j] + a[j:]
        child2 = b[:i] + a[i:j] + b[j:]
        return child1, child2

    def mutate(ind, indpb):
        for i in range(len(ind)):
            if random.random() < indpb:
                ind[i] = 1 - ind[i]

    fitnesses = [fitness(ind) for ind in pop]
    for g in range(gens):
        newpop = []
        while len(newpop) < pop_size:
            parent1 = tournament_select(pop, fitnesses)
            parent2 = tournament_select(pop, fitnesses)
            if random.random() < cx_prob:
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            mutate(child1, mut_prob)
            mutate(child2, mut_prob)
            newpop.append(child1)
            if len(newpop) < pop_size:
                newpop.append(child2)
        pop = newpop
        fitnesses = [fitness(ind) for ind in pop]
    best_idx = int(np.argmax(fitnesses))
    return pop[best_idx], fitnesses[best_idx]

# Genetic Algorithm (optional)
best_individual = None
if run_ga_checkbox:
    st.subheader("Genetic Feature Optimization (sampled / optional)")
    n_features = X_train.shape[1]
    if n_features <= 1:
        st.info("Not enough features for GA.")
    else:

        def eval_ind(ind):
            sel = [i for i, b in enumerate(ind) if b == 1]
            if len(sel) == 0:
                return (0.0,)
            try:
                m = RandomForestClassifier(n_estimators=60) if problem_type == "classification" else RandomForestRegressor(n_estimators=60)
                m.fit(X_train[:, sel], y_train)
                pred = m.predict(X_test[:, sel])
                s = accuracy_score(y_test, pred) if problem_type == "classification" else r2_score(y_test, pred)
                return (float(s),)
            except Exception:
                return (0.0,)

        with st.spinner("Running GA (this may take a moment)..."):
            best_individual, best_score_ga = run_simple_ga(
                eval_fn=eval_ind,
                n_features=n_features,
                pop_size=ga_pop,
                gens=ga_gen,
                cx_prob=0.6,
                mut_prob=0.05,
                tourn_k=3
            )
        st.write("GA selected features count:", int(sum(best_individual)))
        st.write("GA best score:", float(best_score_ga))

# Prediction interface
st.subheader("Quick prediction (sample)")
if st.button("Predict on first test row"):
    try:
        sample_pred = best_model_obj.predict(X_test[:1])
        st.write("Prediction:", sample_pred[0])
    except Exception as e:
        st.warning("Prediction failed: " + str(e))

# Download trained model
try:
    st.download_button("Download trained model (.pkl)",
                       data=pickle.dumps(best_model_obj),
                       file_name="autodfit_model.pkl",
                       mime="application/octet-stream")
except Exception:
    st.info("Model download unavailable for this object.")

# -----------------------------
# PDF report generation
# -----------------------------
def pdf_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(40, 800, "AutoDFit AI Analytics Platform")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(550, 20, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()

def generate_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # Cover
    story.append(Spacer(1, 140))
    story.append(Paragraph("AutoDFit AI Intelligence Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    story.append(PageBreak())

    # TOC
    story.append(Paragraph("Table of Contents", styles["Heading1"]))
    for item in ["1. Executive Summary", "2. Dataset Overview", "3. Model Performance",
                 "4. Visual Analytics", "5. Explainability", "6. Genetic Optimization", "7. Conclusion"]:
        story.append(Paragraph(item, styles["Normal"]))
    story.append(PageBreak())

    # Executive
    story.append(Paragraph("1. Executive Summary", styles["Heading1"]))
    story.append(Paragraph(f"Best model: {best_model_name}", styles["Normal"]))
    story.append(Paragraph(f"Best score: {best_score if best_score is not None else 'N/A'}", styles["Normal"]))
    story.append(PageBreak())

    # Dataset
    story.append(Paragraph("2. Dataset Overview", styles["Heading1"]))
    story.append(Paragraph(f"Rows: {rows}", styles["Normal"]))
    story.append(Paragraph(f"Columns: {cols}", styles["Normal"]))
    story.append(Paragraph(f"Missing cells: {missing}", styles["Normal"]))
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
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.black),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ALIGN", (1,1), (-1,-1), "CENTER")
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # Visual analytics (heatmap)
    story.append(Paragraph("4. Visual Analytics", styles["Heading1"]))
    try:
        plt.figure(figsize=(6,4))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
        plt.tight_layout()
        plt.savefig("tmp_heatmap.png")
        plt.close()
        story.append(RLImage("tmp_heatmap.png", width=400, height=300))
        story.append(PageBreak())
    except Exception:
        story.append(Paragraph("Correlation heatmap unavailable.", styles["Normal"]))
        story.append(PageBreak())

    # Explainability (feature importance + SHAP)
    story.append(Paragraph("5. Explainability (SHAP & Feature importance)", styles["Heading1"]))
    try:
        if hasattr(best_model_obj, "feature_importances_"):
            plt.figure()
            plt.bar(range(len(best_model_obj.feature_importances_)), best_model_obj.feature_importances_)
            plt.tight_layout()
            plt.savefig("tmp_feat.png")
            plt.close()
            story.append(RLImage("tmp_feat.png", width=400, height=250))
    except Exception:
        story.append(Paragraph("Feature importance not available.", styles["Normal"]))

    try:
        background = X_train[:200] if X_train.shape[0] > 200 else X_train
        sample = X_test[:100] if X_test.shape[0] > 100 else X_test
        explainer = shap.Explainer(best_model_obj, background)
        shap_vals = explainer(sample, check_additivity=False)
        plt.figure(figsize=(6,4))
        shap.summary_plot(shap_vals, sample, show=False)
        plt.tight_layout()
        plt.savefig("tmp_shap.png")
        plt.close()
        story.append(RLImage("tmp_shap.png", width=400, height=250))
    except Exception:
        story.append(Paragraph("SHAP plot unavailable.", styles["Normal"]))

    story.append(PageBreak())

    # GA
    story.append(Paragraph("6. Genetic Optimization", styles["Heading1"]))
    if best_individual is not None:
        story.append(Paragraph(f"Selected features count: {sum(best_individual)}", styles["Normal"]))
    else:
        story.append(Paragraph("GA not executed or returned no result.", styles["Normal"]))
    story.append(PageBreak())

    story.append(Paragraph("7. Conclusion", styles["Heading1"]))
    story.append(Paragraph("AutoDFit provides automated dataset evaluation, model comparison, optimization and explainability.", styles["Normal"]))

    # build
    doc.build(story, onFirstPage=pdf_header_footer, onLaterPages=pdf_header_footer)

    # cleanup
    for tmp in ["tmp_heatmap.png", "tmp_feat.png", "tmp_shap.png"]:
        if os.path.exists(tmp):
            os.remove(tmp)

    buffer.seek(0)
    return buffer

# Download PDF
try:
    st.download_button(
        label="📄 Download Full Analysis Report (PDF)",
        data=generate_pdf_report(),
        file_name="AutoDFit_Report.pdf",
        mime="application/pdf"
    )
except Exception as e:
    st.warning("Could not generate PDF: " + str(e))
