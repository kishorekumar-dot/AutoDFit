# app.py — AutoDFit (enterprise-grade production-ready)
# Tested for Python 3.11+ (see requirements.txt). Avoid Python 3.14 due to binary build issues with some wheels.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import random
import pickle
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from math import pi

# sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, roc_curve, auc

# PDF (ReportLab)
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak,
    Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# ---------------------------
# Page configuration & CSS
# ---------------------------
st.set_page_config(page_title="AutoDFit — Dataset Intelligence", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .main-title { font-size:28px; font-weight:700; color:#0f172a; margin-bottom:0.1rem; }
      .subtitle { color:#475569; margin-bottom:1rem; }
      .metric-card { background:#f8fafc; padding:10px; border-radius:10px; border:1px solid #e2e8f0; text-align:center }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Robust helper functions
# ---------------------------
@st.cache_data
def load_csv(file) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    rows, cols = df.shape
    missing_cells = int(df.isnull().sum().sum())
    total = df.size if df.size > 0 else 1
    completeness = max(0.0, min(1.0, (1 - missing_cells / total)))
    uniqueness = (df.nunique().sum() / total) if total > 0 else 0.0
    return {"rows": rows, "cols": cols, "missing_cells": missing_cells, "completeness": completeness, "uniqueness": uniqueness}

def separability_classification(X: np.ndarray, y: np.ndarray) -> float:
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0
    centroids = []
    spreads = []
    for c in classes:
        mask = (y == c)
        if mask.sum() == 0:
            centroids.append(np.zeros(X.shape[1]))
            spreads.append(0.0)
            continue
        group = X[mask]
        centroid = np.mean(group, axis=0)
        centroids.append(centroid)
        dists = np.linalg.norm(group - centroid, axis=1)
        spreads.append(np.mean(dists) if dists.size else 0.0)
    intra = np.mean(spreads) if spreads else 0.0
    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    inter = np.mean(inter_dists) if inter_dists else 0.0
    return float(inter / (intra + 1e-12))

def separability_regression(X: np.ndarray, y: np.ndarray) -> float:
    X = np.asarray(X)
    y = np.asarray(y).astype(float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    corrs = []
    for i in range(X.shape[1]):
        col = X[:, i].astype(float)
        if np.nanstd(col) == 0 or np.nanstd(y) == 0:
            corrs.append(0.0)
            continue
        c = np.corrcoef(col, y)[0, 1]
        corrs.append(abs(c) if not np.isnan(c) else 0.0)
    return float(np.mean(corrs)) if corrs else 0.0

def interaction_density(X_df: pd.DataFrame, threshold: float = 0.7) -> float:
    num = X_df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return 0.0
    corr = num.corr().abs()
    n = corr.shape[0]
    total_pairs = n * (n - 1) / 2
    strong = 0
    for i in range(n):
        for j in range(i + 1, n):
            if corr.iat[i, j] >= threshold:
                strong += 1
    return float(strong / total_pairs) if total_pairs > 0 else 0.0

def imbalance_penalty(y: pd.Series) -> float:
    counts = y.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0.0
    max_share = counts.max()
    ideal = 1.0 / len(counts)
    return float((max_share - ideal) / (1 - ideal))

def noise_index(model, X_train, y_train, scoring: str, cv: int = 4) -> float:
    try:
        model.fit(X_train, y_train)
        train_score = (accuracy_score(y_train, model.predict(X_train)) if scoring == "accuracy"
                       else r2_score(y_train, model.predict(X_train)))
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_mean = float(np.nanmean(cv_scores))
        return float(abs(train_score - cv_mean))
    except Exception:
        return 0.0

def compute_efficiency(model, X_train, y_train, X_test, y_test, problem_type: str) -> Tuple[float, float]:
    t0 = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception:
        return float('nan'), float('nan')
    t1 = time.time()
    train_time = t1 - t0
    try:
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds) if problem_type == "classification" else r2_score(y_test, preds)
        return float(score), float(train_time)
    except Exception:
        return float('nan'), float(train_time)

def draw_accuracy_gauge(pct: float) -> io.BytesIO:
    pct = float(np.clip(pct, 0.0, 1.0))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_aspect('equal')
    ax.axis('off')
    theta_bg = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), color='#eee', linewidth=20, solid_capstyle='round')
    theta_fg = np.linspace(np.pi, np.pi - np.pi * pct, 200)
    ax.plot(np.cos(theta_fg), np.sin(theta_fg), color='#4f46e5', linewidth=20, solid_capstyle='round')
    angle = np.pi - np.pi * pct
    ax.plot([0, 0.9 * np.cos(angle)], [0, 0.9 * np.sin(angle)], color='#111827', linewidth=3)
    ax.text(0, -0.25, f"{pct*100:.1f}%", ha='center', va='center', fontsize=16, fontweight='bold')
    plt.xlim(-1.2, 1.2); plt.ylim(-0.4, 1.1)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_confusion_matrix_buf(y_true, y_pred) -> io.BytesIO:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_roc_curve_buf(y_true, y_prob) -> Optional[io.BytesIO]:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.25)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None

def plot_pred_vs_actual_buf(y_true, y_pred) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(y_true, y_pred, alpha=0.6)
    mn = min(np.min(y_true), np.min(y_pred)); mx = max(np.max(y_true), np.max(y_pred))
    ax.plot([mn, mx], [mn, mx], '--', color='k', alpha=0.5)
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.set_title('Predicted vs Actual')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_feature_importance_buf(importances) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(5, 3))
    idx = np.argsort(importances)[::-1]
    ax.bar(range(len(importances)), importances[idx], color='#4f46e5')
    ax.set_title('Feature Importances'); ax.set_xlabel('Feature (index)')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def fitness_score(stats, sep_norm, interaction, noise, imbalance, weights=None):
    if weights is None:
        weights = {"completeness":0.22, "separability":0.28, "uniqueness":0.12, "interaction":-0.10, "noise":-0.18, "imbalance":-0.10}
    fitness = (
        weights["completeness"] * stats["completeness"] +
        weights["separability"] * sep_norm +
        weights["uniqueness"] * stats["uniqueness"] +
        weights["interaction"] * (1 - interaction) +
        weights["noise"] * (1 - noise) +
        weights["imbalance"] * (1 - imbalance)
    )
    return float(np.clip((fitness + 0.5) * 100, 0, 100))

# ---------------------------
# Sidebar (upload + options)
# ---------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=AutoDFit", use_column_width=False)
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.markdown("---")
    st.subheader("Advanced options")
    enable_feature_search = st.checkbox("Enable Random Feature Search (light)", value=False)
    if enable_feature_search:
        search_iters = st.slider("Random subsets (iterations)", 10, 200, 40, step=10)
        subset_frac = st.slider("Subset fraction (per try)", 0.2, 0.9, 0.6, step=0.1)
    else:
        search_iters, subset_frac = 40, 0.6
    include_pdf_images = st.checkbox("Include plots in PDF", value=True)
    st.markdown("---")
    st.caption("Recommended: use small datasets for fast demo (Iris, Titanic).")

# ---------------------------
# Basic guard
# ---------------------------
if uploaded_file is None:
    st.markdown('<div class="main-title">AutoDFit</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a CSV to analyze dataset fitness, separability, and compare models.</div>', unsafe_allow_html=True)
    st.info("Upload a CSV from the sidebar to start.")
    st.stop()

df = load_csv(uploaded_file)
if df is None:
    st.stop()

# minimum rows check
if df.shape[0] < 8:
    st.error("Dataset too small for reliable modeling (need >= 8 rows). Use a larger dataset.")
    st.stop()

# ---------------------------
# Overview & QC
# ---------------------------
st.markdown('<div class="main-title">Dataset Overview</div>', unsafe_allow_html=True)
stats = compute_basic_stats(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", stats["rows"])
c2.metric("Columns", stats["cols"])
c3.metric("Missing cells", stats["missing_cells"])
c4.metric("Completeness", f"{stats['completeness']*100:.1f}%")
st.dataframe(df.head(8), use_container_width=True)

# choose target
target_col = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)
X_df = df.drop(columns=[target_col])
y_ser = df[target_col]

# improved problem detection
if pd.api.types.is_object_dtype(y_ser) or y_ser.dtype == "category":
    problem_type = "classification"
elif pd.api.types.is_numeric_dtype(y_ser) and y_ser.nunique() <= 10:
    # if numeric but few unique values: likely classification
    problem_type = "classification"
else:
    problem_type = "regression"
st.write(f"**Problem type detected:** {problem_type}")

# Preprocessor build
num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
transformers = []
if num_cols:
    transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("enc", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols))
if not transformers:
    st.error("No usable features found after dropping the target.")
    st.stop()
preprocessor = ColumnTransformer(transformers, remainder="drop")

# safe stratify
stratify_col = None
if problem_type == "classification":
    vc = y_ser.value_counts()
    if vc.min() >= 2:
        stratify_col = y_ser
    else:
        stratify_col = None  # fallback

# split and preprocess
X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y_ser, test_size=0.2, random_state=42, stratify=stratify_col)
with st.spinner("Preprocessing..."):
    try:
        X_train = preprocessor.fit_transform(X_train_df)
        X_test = preprocessor.transform(X_test_df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

# compute data quality metrics
interaction = interaction_density(X_train_df)
imb_pen = imbalance_penalty(y_train) if problem_type == "classification" else 0.0
base_model = DecisionTreeClassifier(random_state=42) if problem_type == "classification" else DecisionTreeRegressor()
scoring = "accuracy" if problem_type == "classification" else "r2"
noise = noise_index(base_model, X_train, y_train, scoring, cv=4)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Interaction Density", f"{interaction:.3f}")
col_b.metric("Imbalance Penalty", f"{imb_pen:.3f}" if problem_type == "classification" else "N/A")
col_c.metric("Noise Index", f"{noise:.3f}")

# ---------------------------
# Separability
# ---------------------------
st.markdown("## Separability")
if problem_type == "classification":
    sep = separability_classification(X_train, y_train)
    sep_norm = min(sep / 3.0, 1.0)
    st.metric("Separability (inter/intra)", f"{sep:.3f}")
else:
    sep = separability_regression(X_train, y_train)
    sep_norm = min(sep, 1.0)
    st.metric("Separability (mean |corr|)", f"{sep:.3f}")

# ---------------------------
# Model Benchmark
# ---------------------------
st.markdown("## Model Benchmark")
if problem_type == "classification":
    model_pool = {
        "Logistic Regression": LogisticRegression(max_iter=400, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
else:
    model_pool = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor()
    }

results = []
progress = st.progress(0)
status = st.empty()
for i, (name, model) in enumerate(model_pool.items()):
    status.text(f"Training {name}...")
    score, tt = compute_efficiency(model, X_train, y_train, X_test, y_test, problem_type)
    results.append({"model": name, "score": score, "train_time_s": tt})
    progress.progress((i + 1) / len(model_pool))
progress.empty(); status.empty()

results_df = pd.DataFrame(results)
results_df["efficiency"] = results_df.apply(
    lambda r: (r["score"] / r["train_time_s"]) if (r["train_time_s"] > 0 and not np.isnan(r["score"])) else np.nan,
    axis=1
)
results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)
st.dataframe(results_df.style.format({"score": "{:.4f}", "train_time_s": "{:.2f}", "efficiency": "{:.4f}"}), use_container_width=True)

if results_df.shape[0] == 0 or results_df["score"].isnull().all():
    st.error("No valid model scores produced.")
    st.stop()

best_row = results_df.iloc[0]
best_model_name = best_row["model"]
best_model_obj = model_pool[best_model_name]
try:
    best_model_obj.fit(X_train, y_train)
    st.session_state["best_model"] = best_model_obj
except Exception as e:
    st.warning(f"Retrain best model failed: {e}")

display_score = best_row["score"] if not np.isnan(best_row["score"]) else 0.0
# clamp for gauge
gauge_display = float(np.clip(display_score, 0.0, 1.0))
gauge_buf = draw_accuracy_gauge(gauge_display)
st.image(gauge_buf, width=320)

# performance bar chart (consistent labels)
fig, ax = plt.subplots(figsize=(8, 3))
vals = results_df["score"].fillna(0).values
names = results_df["model"].tolist()
heights = vals * (100 if problem_type == "classification" else 1)
ax.bar(range(len(vals)), heights, color='#4f46e5')
ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_ylabel("Score (%)" if problem_type == "classification" else "R²")
for i, v in enumerate(heights):
    ax.text(i, v + (1.5 if problem_type == "classification" else 0.01), f"{v:.2f}", ha='center', fontsize=9)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# optional random feature search (light)
if enable_feature_search:
    st.markdown("### Random feature search (light)")
    n_features = X_train.shape[1]
    if n_features <= 1:
        st.info("Not enough features for search.")
    else:
        best_subset = None
        best_subset_score = -1.0
        for it in range(search_iters):
            k = max(1, int(n_features * subset_frac * random.uniform(0.5, 1.0)))
            sel = sorted(random.sample(range(n_features), k))
            try:
                m = RandomForestClassifier(n_estimators=60, random_state=42) if problem_type == "classification" else RandomForestRegressor(n_estimators=60, random_state=42)
                m.fit(X_train[:, sel], y_train)
                preds = m.predict(X_test[:, sel])
                s = accuracy_score(y_test, preds) if problem_type == "classification" else r2_score(y_test, preds)
                if s > best_subset_score:
                    best_subset_score = s
                    best_subset = sel
            except Exception:
                continue
        st.write("Best subset score:", best_subset_score)
        if best_subset is not None:
            st.write("Selected feature count:", len(best_subset))

# feature importances
if hasattr(best_model_obj, "feature_importances_"):
    st.markdown("### Feature importances (best model)")
    fi = best_model_obj.feature_importances_
    fig_fi = plt.figure(figsize=(6, 3))
    plt.bar(range(len(fi)), fi, color='#4f46e5')
    plt.xlabel("Feature index (post-preprocessing)")
    plt.tight_layout()
    st.pyplot(fig_fi)
    plt.close(fig_fi)

# evaluate & show plots
if problem_type == "classification":
    try:
        preds = best_model_obj.predict(X_test)
        cm_buf = plot_confusion_matrix_buf(y_test, preds)
        st.image(cm_buf)
    except Exception:
        st.warning("Confusion matrix unavailable.")
    if hasattr(best_model_obj, "predict_proba") and len(np.unique(y_test)) == 2:
        try:
            probs = best_model_obj.predict_proba(X_test)[:, 1]
            roc_buf = plot_roc_curve_buf(y_test, probs)
            if roc_buf:
                st.image(roc_buf)
        except Exception:
            pass
else:
    try:
        preds = best_model_obj.predict(X_test)
        reg_buf = plot_pred_vs_actual_buf(y_test, preds)
        st.image(reg_buf)
    except Exception:
        pass

# store for report & downloads
st.session_state["df"] = df
st.session_state["X_train"] = X_train; st.session_state["X_test"] = X_test
st.session_state["y_train"] = y_train; st.session_state["y_test"] = y_test
st.session_state["preprocessor"] = preprocessor
st.session_state["problem_type"] = problem_type
st.session_state["stats"] = stats
st.session_state["interaction"] = interaction
st.session_state["imb_pen"] = imb_pen
st.session_state["noise"] = noise
st.session_state["sep"] = sep
st.session_state["sep_norm"] = sep_norm
st.session_state["results_df"] = results_df
st.session_state["best_model_name"] = best_model_name
st.session_state["best_model_score"] = float(display_score)

# compute and display fitness score
fit_score = fitness_score(stats, sep_norm, interaction, noise, imb_pen)
st.markdown("### Dataset Fitness Score")
st.markdown(f"<div class='metric-card'><strong style='font-size:20px'>{fit_score:.1f}%</strong><div style='font-size:0.9rem;color:#475569'>0-40% Poor, 40-70% Moderate, 70-100% Good</div></div>", unsafe_allow_html=True)

# ---------------------------
# Download trained model
# ---------------------------
if "best_model" in st.session_state:
    try:
        st.download_button("Download trained model (.pkl)", data=pickle.dumps(st.session_state["best_model"]), file_name="autodfit_model.pkl", mime="application/octet-stream")
    except Exception:
        st.info("Model cannot be serialized for download.")

# ---------------------------
# PDF report generation (Cover + TOC + Header/Footer)
# ---------------------------
def pdf_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(40, A4[1] - 30, "AutoDFit — Dataset Intelligence")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(A4[0] - 40, 20, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()

def generate_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=72, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # Cover
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, spaceAfter=12)
    story.append(Paragraph("AutoDFit — Dataset Intelligence Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Automated dataset fitness evaluation, model benchmarking and export.", styles['Normal']))
    story.append(PageBreak())

    # Table of contents (simple)
    story.append(Paragraph("Table of Contents", styles['Heading2']))
    for line in ["1. Executive Summary", "2. Dataset Overview", "3. Model Benchmark", "4. Visuals & Metrics", "5. Conclusion"]:
        story.append(Paragraph(line, styles['Normal']))
    story.append(PageBreak())

    # Executive summary
    story.append(Paragraph("1. Executive Summary", styles['Heading2']))
    story.append(Paragraph(f"Dataset fitness score: {fit_score:.1f}%", styles['Normal']))
    story.append(Paragraph(f"Best model: {st.session_state.get('best_model_name', 'N/A')} — score: {st.session_state.get('best_model_score', 0.0):.4f}", styles['Normal']))
    story.append(PageBreak())

    # Dataset Overview
    story.append(Paragraph("2. Dataset Overview", styles['Heading2']))
    ds_table = [["Metric", "Value"],
                ["Rows", str(stats["rows"])],
                ["Columns", str(stats["cols"])],
                ["Missing cells", str(stats["missing_cells"])],
                ["Completeness", f"{stats['completeness']*100:.1f}%"],
                ["Target", str(target_col)],
                ["Problem type", problem_type.capitalize()]]
    tbl = Table(ds_table, colWidths=[3 * inch, 3 * inch])
    tbl.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)]))
    story.append(tbl)
    story.append(PageBreak())

    # Model Benchmark
    story.append(Paragraph("3. Model Benchmark", styles['Heading2']))
    perf = [["Model", "Score", "Train time (s)", "Efficiency"]]
    for _, r in results_df.iterrows():
        perf.append([
            r['model'],
            f"{r['score']:.4f}" if not np.isnan(r['score']) else "N/A",
            f"{r['train_time_s']:.2f}",
            f"{r['efficiency']:.4f}" if not np.isnan(r['efficiency']) else "N/A"
        ])
    perf_tbl = Table(perf, colWidths=[2.5 * inch, 1.0 * inch, 1.0 * inch, 1.2 * inch])
    perf_tbl.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)]))
    story.append(perf_tbl)
    story.append(PageBreak())

    # Visuals & Metrics
    story.append(Paragraph("4. Visuals & Metrics", styles['Heading2']))
    story.append(Paragraph(f"Separability: {st.session_state.get('sep', 0):.3f}", styles['Normal']))
    story.append(Paragraph(f"Interaction density: {interaction:.3f}", styles['Normal']))
    story.append(Paragraph(f"Noise index: {noise:.3f}", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    # add generated images (in-memory)
    images_to_add = []
    try:
        if problem_type == "classification":
            preds = st.session_state["best_model"].predict(st.session_state["X_test"])
            images_to_add.append(plot_confusion_matrix_buf(st.session_state["y_test"], preds))
            if hasattr(st.session_state["best_model"], "predict_proba") and len(np.unique(st.session_state["y_test"])) == 2:
                probs = st.session_state["best_model"].predict_proba(st.session_state["X_test"])[:, 1]
                roc_buf = plot_roc_curve_buf(st.session_state["y_test"], probs)
                if roc_buf:
                    images_to_add.append(roc_buf)
        else:
            preds = st.session_state["best_model"].predict(st.session_state["X_test"])
            images_to_add.append(plot_pred_vs_actual_buf(st.session_state["y_test"], preds))
    except Exception:
        pass

    try:
        if hasattr(st.session_state["best_model"], "feature_importances_"):
            images_to_add.append(plot_feature_importance_buf(st.session_state["best_model"].feature_importances_))
    except Exception:
        pass

    for b in images_to_add:
        try:
            b.seek(0)
            story.append(RLImage(b, width=4 * inch, height=3 * inch))
            story.append(Spacer(1, 0.1 * inch))
        except Exception:
            continue

    story.append(PageBreak())
    story.append(Paragraph("5. Conclusion", styles['Heading2']))
    story.append(Paragraph("AutoDFit provides automated dataset intelligence: quality metrics, separability, model benchmarking and exportable report.", styles['Normal']))

    # Build with header/footer
    doc.build(story, onFirstPage=pdf_header_footer, onLaterPages=pdf_header_footer)
    buffer.seek(0)
    return buffer

# Download PDF button
if st.button("Generate & Download PDF report"):
    with st.spinner("Building PDF..."):
        pdf_buf = generate_pdf_report()
        st.download_button("📥 Download Report (PDF)", data=pdf_buf, file_name="AutoDFit_Report.pdf", mime="application/pdf")
        st.success("PDF ready.")

st.write("---")
st.caption("AutoDFit — production-ready pipeline (lightweight). For heavy explainability (SHAP) run locally.")
