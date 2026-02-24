# AutoDFit - Final production-ready Streamlit app
# Full features: UI, AutoML, GA, SHAP, PDF report (multi-page), model export

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
import base64
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

from deap import base, creator, tools, algorithms

# ---------------------------
# App config & logo (center)
# ---------------------------
st.set_page_config(page_title="AutoDFit", layout="wide", initial_sidebar_state="expanded")

# Simple animated SVG logo (centered)
st.markdown(
    """
    <div style="text-align:center; margin-top:12px; margin-bottom:8px;">
      <svg width="260" height="90" viewBox="0 0 260 90" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="g1" x1="0" x2="1">
            <stop stop-color="#00F5FF" offset="0%"/>
            <stop stop-color="#7B61FF" offset="100%"/>
          </linearGradient>
        </defs>
        <circle cx="52" cy="45" r="20" fill="url(#g1)">
          <animate attributeName="r" values="16;24;16" dur="2s" repeatCount="indefinite"/>
        </circle>
        <text x="95" y="54" font-family="Arial" font-size="30" font-weight="700" fill="#1f2937">AutoDFit</text>
      </svg>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("AutoDFit — AI Dataset Intelligence Platform")
st.caption("Automated dataset fitness, model comparison, explainability and professional reporting")

# ---------------------------
# Helper: LabelEncoder wrapper
# ---------------------------
class LabelEncoderWrapper:
    def fit(self, X, y=None):
        self.encoders = {}
        # X is numpy
        for i in range(X.shape[1]):
            le = LabelEncoder()
            # safe convert to str to avoid issues
            col = X[:, i].astype(str)
            X[:, i] = le.fit_transform(col)
            self.encoders[i] = le
        return self

    def transform(self, X):
        for i in range(X.shape[1]):
            X[:, i] = self.encoders[i].transform(X[:, i].astype(str))
        return X

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
    total = df.size
    completeness = (1 - missing/total) * 100
    uniqueness = (df.nunique().sum()/total) * 100
    quality_score = round(completeness*0.7 + uniqueness*0.3, 2)
    return rows, cols, missing, quality_score

@st.cache_resource
def train_models_cached(X_train, X_test, y_train, y_test, problem):
    # Train a standard set of models (kept reasonably light)
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
            scores[name] = float(score)
            trained[name] = m
        except Exception as e:
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
    ga_gen = st.slider("GA generations (if enabled)", min_value=3, max_value=10, value=5, step=1)
    dark_theme = st.checkbox("Dark theme (UI hint only)", value=False)
    st.write("Tip: use small datasets for faster runs (<= 10000 rows)")

# ---------------------------
# File upload
# ---------------------------
uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to start. Example: small classification dataset (Iris, Titanic, etc.).")
    st.stop()

# load dataframe
df = load_csv(uploaded)

# show quick stats
rows, cols, missing, quality_score = quick_data_stats(df)
st.header("Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", rows)
c2.metric("Cols", cols)
c3.metric("Missing cells", missing)
st.write(f"Dataset health score: **{quality_score}%**")
st.dataframe(df.head())

# correlation (matplotlib)
st.subheader("Correlation heatmap")
fig_corr = plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
st.pyplot(fig_corr)
plt.close(fig_corr)

# target selection
target_col = st.selectbox("Select target column", df.columns)
X_df = df.drop(columns=[target_col])
y_ser = df[target_col]

# detect problem type
problem_type = "classification" if (y_ser.dtype == "object" or y_ser.nunique() < 20) else "regression"
st.write(f"Detected problem type: **{problem_type}**")

# preprocessing pipeline creation
num_cols = X_df.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_df.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("enc", LabelEncoderWrapper())]), cat_cols)
])

# split and preprocess
with st.spinner("Preparing data (split + preprocess)..."):
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y_ser, test_size=0.2, random_state=42)
    # Convert to numpy for wrapper encoder
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

# Train models (cached): fast models only
with st.spinner("Training models (this may take a moment)..."):
    scores, trained_models = train_models_cached(X_train, X_test, y_train, y_test, problem_type)

# Display leaderboard
leader_df = pd.DataFrame(list(scores.items()), columns=["Model", "Score"]).sort_values("Score", ascending=False)
st.subheader("Model leaderboard")
st.dataframe(leader_df.style.format({"Score": "{:.4f}"}))

best_model_name = leader_df.iloc[0]["Model"]
best_model_obj = trained_models[best_model_name]
best_score = float(leader_df.iloc[0]["Score"])

st.metric("Best model", best_model_name, delta=f"{best_score*100:.2f}%")

# performance chart (bar)
st.subheader("Model comparison chart")
fig_bar = plt.figure(figsize=(6,3))
plt.bar(leader_df["Model"], leader_df["Score"] * 100)
plt.ylabel("Score (%)")
plt.xticks(rotation=25)
st.pyplot(fig_bar)
plt.close(fig_bar)

# Confusion matrix + ROC for classification
if problem_type == "classification":
    st.subheader("Model evaluation: confusion matrix & ROC (best model)")
    preds = best_model_obj.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig_cm = plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # ROC (if probability available)
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
            pass

# Feature importance (if available)
st.subheader("Feature importance (if supported)")
if hasattr(best_model_obj, "feature_importances_"):
    feats = best_model_obj.feature_importances_
    fig_feat = plt.figure(figsize=(6,3))
    plt.bar(range(len(feats)), feats)
    plt.xlabel("Feature index")
    st.pyplot(fig_feat)
    plt.close(fig_feat)
else:
    st.info("Selected model doesn't expose feature_importances_ (try RandomForest).")

# SHAP explainability (sampled, stable)
if show_shap_checkbox:
    st.subheader("SHAP explainability (sampled)")
    try:
        # small background + sample to keep fast & stable
        background = X_train[:200] if X_train.shape[0] > 200 else X_train
        sample = X_test[:100] if X_test.shape[0] > 100 else X_test

        # Use TreeExplainer for tree models when possible
        try:
            explainer = shap.Explainer(best_model_obj, background)
        except Exception:
            explainer = shap.Explainer(best_model_obj, background)

        shap_values = explainer(sample, check_additivity=False)
        fig_shap = plt.figure(figsize=(6,4))
        shap.summary_plot(shap_values, sample, show=False)
        st.pyplot(fig_shap)
        plt.close(fig_shap)
    except Exception as e:
        st.warning("SHAP explanation could not be generated. " + str(e))

# Genetic Algorithm (optional) - small and safe
best_individual = None
if run_ga_checkbox:
    st.subheader("Genetic Feature Optimization (sampled / optional)")
    n_features = X_train.shape[1]
    if n_features <= 1:
        st.info("Not enough features for GA.")
    else:
        # build GA toolbox (small)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval_ind(ind):
            sel = [i for i, b in enumerate(ind) if b == 1]
            if len(sel) == 0:
                return 0.0,
            try:
                m = RandomForestClassifier(n_estimators=60) if problem_type == "classification" else RandomForestRegressor(n_estimators=60)
                m.fit(X_train[:, sel], y_train)
                pred = m.predict(X_test[:, sel])
                s = accuracy_score(y_test, pred) if problem_type == "classification" else r2_score(y_test, pred)
                return float(s),
            except Exception:
                return 0.0,

        toolbox.register("evaluate", eval_ind)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=ga_pop)
        with st.spinner("Running GA (this may take a moment)..."):
            algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.2, ngen=ga_gen, verbose=False)
        best_individual = tools.selBest(pop, 1)[0]
        st.write("GA selected features count:", sum(best_individual))

# Prediction interface (basic)
st.subheader("Quick prediction (sample)")
if st.button("Predict on first test row"):
    try:
        sample_pred = best_model_obj.predict(X_test[:1])
        st.write("Prediction:", sample_pred[0])
    except Exception as e:
        st.warning("Prediction failed: " + str(e))

# Download trained model
st.download_button("Download trained model (.pkl)",
                   data=pickle.dumps(best_model_obj),
                   file_name="autodfit_model.pkl",
                   mime="application/octet-stream")

# -----------------------------
# PDF generation (full multi-page)
# -----------------------------
# Helper: header and footer for PDF (ReportLab)
def pdf_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(40, 800, "AutoDFit AI Analytics Platform")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(550, 20, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()

# The heavy PDF build must be inside the runtime block where all variables exist
def generate_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # Cover
    story.append(Spacer(1, 180))
    story.append(Paragraph("AutoDFit AI Intelligence Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    story.append(PageBreak())

    # Table of contents (simple)
    story.append(Paragraph("Table of Contents", styles["Heading1"]))
    toc_items = [
        "1. Executive Summary",
        "2. Dataset Overview",
        "3. Model Performance",
        "4. Visual Analytics",
        "5. Explainability (SHAP)",
        "6. Genetic Optimization",
        "7. Conclusion"
    ]
    for t in toc_items:
        story.append(Paragraph(t, styles["Normal"]))
    story.append(PageBreak())

    # Executive summary
    story.append(Paragraph("1. Executive Summary", styles["Heading1"]))
    story.append(Paragraph(f"Best model: {best_model_name}", styles["Normal"]))
    story.append(Paragraph(f"Best score: {best_score*100:.2f}%", styles["Normal"]))
    story.append(Paragraph(f"Dataset health score: {quality_score}%", styles["Normal"]))
    story.append(PageBreak())

    # Dataset overview
    story.append(Paragraph("2. Dataset Overview", styles["Heading1"]))
    story.append(Paragraph(f"Rows: {rows}", styles["Normal"]))
    story.append(Paragraph(f"Columns: {cols}", styles["Normal"]))
    story.append(Paragraph(f"Missing cells: {missing}", styles["Normal"]))
    story.append(PageBreak())

    # Model performance table
    story.append(Paragraph("3. Model Performance", styles["Heading1"]))
    table_data = [["Model", "Score (%)"]]
    for m, s in scores.items():
        table_data.append([m, f"{s*100:.2f}"])
    tbl = Table(table_data)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.black),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ALIGN", (1,1), (-1,-1), "CENTER")
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # Visual analytics (correlation heatmap)
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

    # Model evaluation (confusion, ROC)
    story.append(Paragraph("5. Model Evaluation", styles["Heading1"]))
    if problem_type == "classification":
        try:
            plt.figure()
            sns.heatmap(confusion_matrix(y_test, best_model_obj.predict(X_test)), annot=True, fmt="d")
            plt.tight_layout()
            plt.savefig("tmp_cm.png")
            plt.close()
            story.append(RLImage("tmp_cm.png", width=350, height=250))
        except Exception:
            story.append(Paragraph("Confusion matrix unavailable.", styles["Normal"]))

        try:
            if hasattr(best_model_obj, "predict_proba"):
                probs = best_model_obj.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                plt.figure()
                plt.plot(fpr, tpr)
                plt.plot([0,1],[0,1], "--")
                plt.tight_layout()
                plt.savefig("tmp_roc.png")
                plt.close()
                story.append(RLImage("tmp_roc.png", width=350, height=250))
        except Exception:
            story.append(Paragraph("ROC curve unavailable.", styles["Normal"]))
    else:
        story.append(Paragraph("ROC/Confusion only relevant for classification.", styles["Normal"]))

    story.append(PageBreak())

    # Explainability (feature importance + SHAP)
    story.append(Paragraph("6. Explainability (SHAP & Feature importance)", styles["Heading1"]))
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
        # generate SHAP image (sampled)
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

    # Genetic optimization
    story.append(Paragraph("7. Genetic Optimization (if run)", styles["Heading1"]))
    if best_individual is not None:
        story.append(Paragraph(f"Selected features count: {sum(best_individual)}", styles["Normal"]))
    else:
        story.append(Paragraph("GA not executed or returned no result.", styles["Normal"]))

    story.append(PageBreak())

    # Conclusion
    story.append(Paragraph("8. Conclusion", styles["Heading1"]))
    story.append(Paragraph("AutoDFit provides an automated pipeline for dataset evaluation, model comparison, "
                           "optimization and explainability. Use the report to make decisions on feature engineering "
                           "and model deployment.", styles["Normal"]))

    # Build PDF with header/footer
    doc.build(story, onFirstPage=pdf_header_footer, onLaterPages=pdf_header_footer)

    # clean up temp images
    for tmp in ["tmp_heatmap.png", "tmp_cm.png", "tmp_roc.png", "tmp_feat.png", "tmp_shap.png"]:
        if os.path.exists(tmp):
            os.remove(tmp)

    buffer.seek(0)
    return buffer

# Download button (calls generator when user clicks)
st.download_button(
    label="📄 Download Full Analysis Report (PDF)",
    data=generate_pdf_report(),
    file_name="AutoDFit_Report.pdf",
    mime="application/pdf"
)

# End of app
