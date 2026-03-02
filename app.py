# app.py — AutoDFit v1.0 (Production-ready for Render)
# Features:
# - Tabbed UI: Home, Dataset Intelligence, AutoML, Explainability, Optimization, Report
# - Safe preprocessing (OrdinalEncoder), model training (5 algos), leaderboard
# - Pure-Python GA optimization (optional)
# - SHAP on-demand (button, sampled)
# - Multi-page PDF report generation (ReportLab) with temp images
# - Download trained model
# - Defensive error handling + caching

import os
import io
import tempfile
import pickle
import random
import copy
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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

# -------------------------
# Page config and styling
# -------------------------
st.set_page_config(page_title="AutoDFit", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background-color: #f7f8fb; }
      .title { font-weight:700; }
    </style>
    """, unsafe_allow_html=True
)

# Simple SVG logo (inline)
st.markdown(
    """
    <div style="text-align:center; margin-bottom:8px;">
      <svg width="260" height="80" viewBox="0 0 260 90" xmlns="http://www.w3.org/2000/svg">
        <defs><linearGradient id="g1" x1="0" x2="1"><stop stop-color="#00F5FF" offset="0%"/><stop stop-color="#7B61FF" offset="100%"/></linearGradient></defs>
        <circle cx="52" cy="45" r="20" fill="url(#g1)">
          <animate attributeName="r" values="16;24;16" dur="2s" repeatCount="indefinite"/>
        </circle>
        <text x="95" y="54" font-family="Arial" font-size="30" font-weight="700" fill="#111827">AutoDFit</text>
      </svg>
    </div>
    """, unsafe_allow_html=True
)
st.title("AutoDFit — AI Dataset Intelligence Platform")
st.caption("Automated dataset fitness, model selection, explainability & evolutionary feature optimization")

# -------------------------
# Configurable defaults (via environment for Render)
# -------------------------
GA_DEFAULT_POP = int(os.getenv("GA_DEFAULT_POP", 10))
GA_DEFAULT_GEN = int(os.getenv("GA_DEFAULT_GEN", 3))
SHAP_SAMPLE = int(os.getenv("SHAP_SAMPLE", 100))

# -------------------------
# Caches and helpers
# -------------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_resource
def train_models_cached(X_train, X_test, y_train, y_test, problem):
    # Models chosen to be representative and light enough
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
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = accuracy_score(y_test, pred) if problem == "classification" else r2_score(y_test, pred)
            scores[name] = float(score) if score is not None else float("nan")
            trained[name] = model
        except Exception:
            scores[name] = float("nan")
            trained[name] = model
    return scores, trained

def safe_corr(df):
    try:
        return df.corr(numeric_only=True)
    except Exception:
        return None

# Pure-Python GA (lightweight)
def run_simple_ga(eval_fn, n_features, pop_size=10, gens=3, cx_prob=0.6, mut_prob=0.05, tourn_k=3):
    if n_features <= 1:
        return None, 0.0
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
    for _ in range(gens):
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
    return pop[best_idx], float(fitnesses[best_idx])

# -------------------------
# UI: Tabs skeleton
# -------------------------
tabs = st.tabs(["Home", "Dataset Intelligence", "AutoML Engine", "Explainable AI", "Optimization Lab", "AI Report"])
home_tab, data_tab, automl_tab, explain_tab, opt_tab, report_tab = tabs

# Global state container (Streamlit session_state)
if "df" not in st.session_state:
    st.session_state.df = None
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = None
if "scores" not in st.session_state:
    st.session_state.scores = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
if "best_individual" not in st.session_state:
    st.session_state.best_individual = None

# -------------------------
# HOME TAB
# -------------------------
with home_tab:
    st.header("Welcome")
    st.markdown("""
    **AutoDFit** automates dataset fitness evaluation, model selection, explainability, and feature optimization.
    Upload a CSV on the right (or below on mobile), then go to **Dataset Intelligence** to start.
    """)
    st.info("Pro tip: Use small demo CSVs for live demos (Iris, Titanic). SHAP & GA are heavy — run only when needed.")
    st.markdown("**Quick capabilities:** Dataset profiling • AutoML (5 models) • SHAP explainability (on-demand) • Genetic feature optimization (optional) • PDF report")

# -------------------------
# DATASET INTELLIGENCE TAB
# -------------------------
with data_tab:
    st.header("Dataset Intelligence")
    uploaded = st.file_uploader("Upload CSV dataset (required for analysis)", type=["csv"], key="uploader")
    if uploaded is not None:
        try:
            df = load_csv(uploaded)
            st.session_state.df = df
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
            st.stop()
    else:
        if st.session_state.df is None:
            st.info("Please upload a CSV to proceed. Use a small test CSV for first run.")
            st.stop()
        df = st.session_state.df

    # Overview widgets
    rows, cols = df.shape
    missing = int(df.isnull().sum().sum())
    total = df.size if df.size else 1
    completeness = round((1 - missing / total) * 100, 2)
    uniqueness = round((df.nunique().sum() / total) * 100, 2)
    quality_score = round(completeness * 0.7 + uniqueness * 0.3, 2)

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Missing cells", missing)
    c4.metric("Dataset health", f"{quality_score}%")

    st.subheader("Data preview")
    st.dataframe(df.head())

    st.subheader("Feature summary")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    st.write(f"Numeric features: **{len(num_cols)}** • Categorical features: **{len(cat_cols)}**")

    corr = safe_corr(df)
    st.subheader("Correlation heatmap (numeric features)")
    if corr is not None and not corr.empty:
        fig = plt.figure(figsize=(6,4))
        sns.heatmap(corr, cmap="coolwarm")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Correlation heatmap not available (insufficient numeric data).")

    # Target selection
    st.subheader("Select target column")
    target = st.selectbox("Choose target (label) column", df.columns, key="target_select")
    st.session_state["target_col"] = target

# -------------------------
# AUTOML ENGINE TAB
# -------------------------
with automl_tab:
    st.header("AutoML Engine")
    if st.session_state.df is None:
        st.info("Upload dataset first in Dataset Intelligence.")
        st.stop()
    df = st.session_state.df.copy()
    target = st.session_state.get("target_col", None)
    if target is None:
        st.error("Target column not selected. Go to Dataset Intelligence tab.")
        st.stop()

    X_df = df.drop(columns=[target])
    y_ser = df[target]

    problem = "classification" if (y_ser.dtype == "object" or y_ser.nunique() < 20) else "regression"
    st.success(f"Detected problem type: {problem}")

    # Build preprocessor (safe)
    num_cols = X_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_df.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols))

    if not transformers:
        st.error("No features available for modeling after dropping target.")
        st.stop()

    pre = ColumnTransformer(transformers, remainder="drop")
    with st.spinner("Splitting and preprocessing..."):
        X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y_ser, test_size=0.2, random_state=42)
        try:
            X_train = pre.fit_transform(X_train_df)
            X_test = pre.transform(X_test_df)
        except Exception as e:
            st.error("Preprocessing failed: " + str(e))
            st.stop()
    st.session_state.preprocessor = pre

    # Model training (cached)
    run_train = st.button("Train models (5 algorithms)", key="train_models")
    if run_train:
        with st.spinner("Training models..."):
            scores, trained = train_models_cached(X_train, X_test, y_train, y_test, problem)
            st.session_state.trained_models = trained
            st.session_state.scores = scores
            # leaderboard
            df_scores = pd.DataFrame(list(scores.items()), columns=["Model", "Score"])
            df_scores = df_scores.sort_values("Score", ascending=False).reset_index(drop=True)
            st.subheader("Model Leaderboard")
            st.dataframe(df_scores.style.format({"Score":"{:.4f}"}))
            if df_scores.shape[0] > 0:
                best_name = df_scores.iloc[0]["Model"]
                st.session_state.best_model_name = best_name
                st.success(f"Best model: {best_name} ({df_scores.iloc[0]['Score']:.4f})")
    else:
        # show cached results if present
        if st.session_state.scores is not None:
            df_scores = pd.DataFrame(list(st.session_state.scores.items()), columns=["Model", "Score"]).sort_values("Score", ascending=False)
            st.subheader("Cached Leaderboard")
            st.dataframe(df_scores.style.format({"Score":"{:.4f}"}))
            if st.session_state.best_model_name:
                st.info(f"Best cached model: {st.session_state.best_model_name}")

    # If trained, show evaluation charts (on-demand)
    if st.session_state.trained_models is not None and st.session_state.scores is not None:
        df_scores = pd.DataFrame(list(st.session_state.scores.items()), columns=["Model", "Score"]).sort_values("Score", ascending=False)
        st.subheader("Model comparison chart")
        fig = plt.figure(figsize=(7,3))
        scores_numeric = [s if (not (isinstance(s, float) and np.isnan(s))) else 0.0 for s in df_scores["Score"].tolist()]
        plt.bar(df_scores["Model"], np.array(scores_numeric) * 100)
        plt.ylabel("Score (%)")
        plt.xticks(rotation=25)
        st.pyplot(fig)
        plt.close(fig)

        best_name = df_scores.iloc[0]["Model"]
        best_obj = st.session_state.trained_models.get(best_name)
        st.metric("Selected best model", best_name, f"{df_scores.iloc[0]['Score']*100:.2f}%")
        # Evaluation charts for classification
        if problem == "classification":
            try:
                preds = best_obj.predict(X_test)
                cm = confusion_matrix(y_test, preds)
                fig_cm = plt.figure(figsize=(4,3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                st.pyplot(fig_cm)
                plt.close(fig_cm)
                if hasattr(best_obj, "predict_proba"):
                    try:
                        probs = best_obj.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, probs)
                        roc_auc = auc(fpr, tpr)
                        fig_roc = plt.figure(figsize=(4,3))
                        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
                        plt.plot([0,1],[0,1],"--")
                        plt.legend()
                        st.pyplot(fig_roc)
                        plt.close(fig_roc)
                    except Exception:
                        st.info("ROC unavailable for this model/dataset.")
            except Exception:
                st.info("Evaluation charts could not be generated for selected model.")

        # Model export
        if st.button("Download best model (.pkl)"):
            try:
                buf = io.BytesIO()
                pickle.dump(best_obj, buf)
                buf.seek(0)
                st.download_button("Click to download model", data=buf.read(), file_name="autodfit_model.pkl", mime="application/octet-stream")
            except Exception as e:
                st.warning("Model download failed: " + str(e))

# -------------------------
# EXPLAINABLE AI TAB
# -------------------------
with explain_tab:
    st.header("Explainable AI")
    if st.session_state.trained_models is None:
        st.info("Train models first in AutoML Engine.")
        st.stop()
    best_name = st.session_state.best_model_name
    best_model = st.session_state.trained_models.get(best_name)
    st.write(f"Current best model: **{best_name}**")

    # Feature importance
    st.subheader("Feature importance")
    try:
        if hasattr(best_model, "feature_importances_"):
            feats = best_model.feature_importances_
            fig = plt.figure(figsize=(6,3))
            plt.bar(range(len(feats)), feats)
            plt.xlabel("Feature index (post-processing)")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Model does not provide feature_importances_.")
    except Exception:
        st.info("Could not compute feature importance for this model.")

    # SHAP - on demand
    st.subheader("SHAP (on-demand, sampled)")
    if st.button("Run SHAP (sampled)", key="shap_button"):
        with st.spinner("Computing SHAP values (sampled)..."):
            try:
                # safe small background and sample
                background = None
                sample = None
                # try to reconstruct X_train, X_test via preprocessor + stored splits
                # Note: preprocessor in session_state contains transformers but not X_train/X_test; we rely on training-time local variables
                # We'll attempt to reconstruct from the last training step's preprocessed arrays if present in cache (train_models_cached used them)
                # Fallback: use preprocessor to transform a slice of the original df
                df = st.session_state.df
                target = st.session_state.get("target_col")
                X_df = df.drop(columns=[target])
                X_train_df = X_df.sample(min(200, len(X_df)), random_state=42)
                background = st.session_state.get("background_cached")
                # If not cached, compute small background
                if background is None:
                    try:
                        pre = st.session_state.preprocessor
                        background = pre.transform(X_train_df)
                        st.session_state.background_cached = background
                    except Exception:
                        background = None
                # sample for explanation
                X_sample_df = X_df.sample(min(SHAP_SAMPLE, max(10, int(len(X_df)/10)))) if len(X_df) > 0 else X_df
                try:
                    sample = st.session_state.preprocessor.transform(X_sample_df)
                except Exception:
                    sample = None

                if background is None or sample is None:
                    st.warning("SHAP: insufficient preprocessed data (try re-training with more rows or run train models).")
                else:
                    explainer = shap.Explainer(best_model, background)
                    shap_vals = explainer(sample, check_additivity=False)
                    fig_shap = plt.figure(figsize=(7,4))
                    shap.summary_plot(shap_vals, sample, show=False)
                    st.pyplot(fig_shap)
                    plt.close(fig_shap)
            except Exception as e:
                st.error("SHAP failed: " + str(e))

# -------------------------
# OPTIMIZATION LAB TAB (GA)
# -------------------------
with opt_tab:
    st.header("Optimization Lab — Genetic Feature Selection")
    if st.session_state.trained_models is None:
        st.info("Train models first in AutoML Engine to use GA results baseline.")
        st.stop()
    if st.session_state.preprocessor is None:
        st.info("Preprocessor not available. Train models first.")
        st.stop()

    # get preprocessed feature count
    # use X_train shape by transforming small slice
    df = st.session_state.df
    target = st.session_state.get("target_col")
    X_df = df.drop(columns=[target])
    # transform a small slice to get feature count
    try:
        tmp = st.session_state.preprocessor.transform(X_df.head(5))
        n_features = tmp.shape[1]
    except Exception:
        st.error("Could not compute feature count for GA.")
        n_features = 0

    st.write(f"Features (post-preprocessing): {n_features}")
    ga_run = st.button("Run GA optimization (small run)", key="ga_button")
    pop_size = st.number_input("Population size", min_value=6, max_value=30, value=GA_DEFAULT_POP, step=2)
    gens = st.number_input("Generations", min_value=1, max_value=8, value=GA_DEFAULT_GEN, step=1)

    if ga_run:
        with st.spinner("Running GA (this may take a moment)..."):
            def eval_fn(ind):
                sel = [i for i, b in enumerate(ind) if b == 1]
                if len(sel) == 0:
                    return (0.0,)
                try:
                    # reconstruct a small training set to evaluate feature subset quickly
                    X_df_local = X_df.copy()
                    y_local = df[target].copy()
                    X_train_df_local, X_test_df_local, y_train_local, y_test_local = train_test_split(X_df_local, y_local, test_size=0.2, random_state=42)
                    pre_local = st.session_state.preprocessor
                    X_train_local = pre_local.transform(X_train_df_local)
                    X_test_local = pre_local.transform(X_test_df_local)
                    model_local = RandomForestClassifier(n_estimators=60) if (y_local.dtype == "object" or y_local.nunique() < 20) else RandomForestRegressor(n_estimators=60)
                    model_local.fit(X_train_local[:, sel], y_train_local)
                    pred = model_local.predict(X_test_local[:, sel])
                    sc = accuracy_score(y_test_local, pred) if (y_local.dtype == "object" or y_local.nunique() < 20) else r2_score(y_test_local, pred)
                    return (float(sc),)
                except Exception:
                    return (0.0,)
            best_ind, best_score = run_simple_ga(eval_fn, n_features, pop_size=int(pop_size), gens=int(gens))
            if best_ind is None:
                st.info("Not enough features for GA.")
            else:
                st.session_state.best_individual = best_ind
                st.success(f"GA finished. Best score (sampled): {best_score:.4f}")
                st.write("Selected features count:", int(sum(best_ind)))

# -------------------------
# AI REPORT TAB (PDF generation)
# -------------------------
with report_tab:
    st.header("AI Report (Download)")
    if st.session_state.df is None or st.session_state.trained_models is None:
        st.info("Complete Dataset Intelligence and AutoML Engine steps first.")
        st.stop()

    def pdf_header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(40, 800, "AutoDFit AI Analytics Platform")
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(550, 20, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    def generate_pdf_buffer():
        # Create temp directory for images
        tmpdir = tempfile.mkdtemp()
        styles = getSampleStyleSheet()
        story = []
        # Cover
        story.append(Spacer(1, 140))
        story.append(Paragraph("AutoDFit AI Intelligence Report", styles["Title"]))
        story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
        story.append(PageBreak())
        # TOC
        story.append(Paragraph("Table of Contents", styles["Heading1"]))
        for item in ["1. Executive Summary", "2. Dataset Overview", "3. Model Performance", "4. Visual Analytics", "5. Explainability", "6. Genetic Optimization", "7. Conclusion"]:
            story.append(Paragraph(item, styles["Normal"]))
        story.append(PageBreak())
        # Executive summary
        story.append(Paragraph("1. Executive Summary", styles["Heading1"]))
        try:
            best_name = st.session_state.best_model_name
            best_score = st.session_state.scores.get(best_name, None)
            story.append(Paragraph(f"Best model: {best_name}", styles["Normal"]))
            story.append(Paragraph(f"Best score: {best_score}", styles["Normal"]))
        except Exception:
            story.append(Paragraph("Best model: N/A", styles["Normal"]))
        story.append(PageBreak())
        # Dataset overview
        df = st.session_state.df
        rows, cols = df.shape
        missing = int(df.isnull().sum().sum())
        story.append(Paragraph("2. Dataset Overview", styles["Heading1"]))
        story.append(Paragraph(f"Rows: {rows}", styles["Normal"]))
        story.append(Paragraph(f"Columns: {cols}", styles["Normal"]))
        story.append(Paragraph(f"Missing cells: {missing}", styles["Normal"]))
        story.append(PageBreak())
        # Model performance table
        story.append(Paragraph("3. Model Performance", styles["Heading1"]))
        table_data = [["Model", "Score (%)"]]
        for m, s in st.session_state.scores.items():
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
        corr = safe_corr(df)
        if corr is not None and not corr.empty:
            heatpath = os.path.join(tmpdir, "heatmap.png")
            plt.figure(figsize=(6,4))
            sns.heatmap(corr, cmap="coolwarm")
            plt.tight_layout()
            plt.savefig(heatpath)
            plt.close()
            story.append(RLImage(heatpath, width=400, height=300))
        else:
            story.append(Paragraph("Correlation heatmap unavailable.", styles["Normal"]))
        story.append(PageBreak())
        # Explainability (feature importance & SHAP if available)
        story.append(Paragraph("5. Explainability", styles["Heading1"]))
        try:
            best_name = st.session_state.best_model_name
            best_obj = st.session_state.trained_models.get(best_name)
            if hasattr(best_obj, "feature_importances_"):
                featpath = os.path.join(tmpdir, "feat.png")
                plt.figure(figsize=(6,3))
                plt.bar(range(len(best_obj.feature_importances_)), best_obj.feature_importances_)
                plt.tight_layout()
                plt.savefig(featpath)
                plt.close()
                story.append(RLImage(featpath, width=400, height=250))
        except Exception:
            story.append(Paragraph("Feature importance unavailable.", styles["Normal"]))
        # SHAP image if cached
        try:
            shappath = os.path.join(tmpdir, "shap.png")
            # Try a lightweight SHAP generation (sample)
            df_local = st.session_state.df
            target_col = st.session_state.get("target_col")
            X_df = df_local.drop(columns=[target_col])
            # sample small
            sample_df = X_df.sample(min(50, len(X_df))) if len(X_df) > 0 else X_df
            try:
                pre = st.session_state.preprocessor
                sample = pre.transform(sample_df)
                explainer = shap.Explainer(st.session_state.trained_models[st.session_state.best_model_name], sample[:min(20, sample.shape[0])])
                shap_vals = explainer(sample[:min(20, sample.shape[0])], check_additivity=False)
                plt.figure(figsize=(6,4))
                shap.summary_plot(shap_vals, sample[:min(20, sample.shape[0])], show=False)
                plt.tight_layout()
                plt.savefig(shappath)
                plt.close()
                story.append(RLImage(shappath, width=400, height=250))
            except Exception:
                story.append(Paragraph("SHAP plot unavailable (sample too small or model incompatible).", styles["Normal"]))
        except Exception:
            story.append(Paragraph("SHAP generation failed.", styles["Normal"]))

        story.append(PageBreak())
        # GA
        story.append(Paragraph("6. Genetic Optimization", styles["Heading1"]))
        bi = st.session_state.get("best_individual")
        if bi is not None:
            story.append(Paragraph(f"Selected features (count): {int(sum(bi))}", styles["Normal"]))
        else:
            story.append(Paragraph("GA not executed or no result.", styles["Normal"]))
        story.append(PageBreak())
        # Conclusion
        story.append(Paragraph("7. Conclusion", styles["Heading1"]))
        story.append(Paragraph("AutoDFit provides an automated pipeline for dataset evaluation, model comparison, optimization and explainability.", styles["Normal"]))
        # Build PDF into BytesIO
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
        doc.build(story, onFirstPage=pdf_header_footer, onLaterPages=pdf_header_footer)
        buf.seek(0)
        # cleanup tmpdir
        try:
            for f in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, f))
            os.rmdir(tmpdir)
        except Exception:
            pass
        return buf

    if st.button("Generate & Download full PDF report"):
        with st.spinner("Building PDF report..."):
            try:
                pdf_buf = generate_pdf_buffer()
                st.download_button("Download Report (PDF)", data=pdf_buf, file_name="AutoDFit_Report.pdf", mime="application/pdf")
            except Exception as e:
                st.error("PDF generation failed: " + str(e))

# End of app.py
