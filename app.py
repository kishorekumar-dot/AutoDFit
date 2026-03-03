# app.py — AutoDFit (updated, production-ready)
# - No shap, no deap (avoids heavy compiled deps)
# - Safe stratify handling
# - ROC multiclass guard
# - Fixed xticks/labels warnings
# - In-memory PDF images
# - Light random feature search instead of DEAP GA
# - Accuracy gauge visualization

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

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# Streamlit page config
st.set_page_config(page_title="AutoDFit", layout="wide", initial_sidebar_state="expanded")

# Minimal CSS
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .header { font-size:28px; font-weight:700; color:#0f172a; }
      .sub { color:#475569; margin-bottom:12px; }
      .card { background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True
)

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_csv(file) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        return None

def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    rows, cols = df.shape
    missing_cells = int(df.isnull().sum().sum())
    total_cells = df.size if df.size > 0 else 1
    completeness = 1 - missing_cells / total_cells
    uniqueness = df.nunique().sum() / total_cells
    return {"rows": rows, "cols": cols, "missing_cells": missing_cells, "completeness": completeness, "uniqueness": uniqueness}

def separability_classification(X: np.ndarray, y: np.ndarray) -> float:
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
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
        spreads.append(np.mean(dists) if len(dists)>0 else 0.0)
    intra = np.mean(spreads) if len(spreads)>0 else 0.0
    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i+1,len(centroids)):
            inter_dists.append(np.linalg.norm(centroids[i]-centroids[j]))
    inter = np.mean(inter_dists) if inter_dists else 0.0
    return float(inter / (intra + 1e-12))

def separability_regression(X: np.ndarray, y: np.ndarray) -> float:
    X = np.asarray(X); y = np.asarray(y).astype(float)
    corrs = []
    if X.shape[1] == 0:
        return 0.0
    for i in range(X.shape[1]):
        col = X[:,i].astype(float)
        if np.nanstd(col)==0 or np.nanstd(y)==0:
            corrs.append(0.0)
            continue
        c = np.corrcoef(col, y)[0,1]
        corrs.append(abs(c) if not np.isnan(c) else 0.0)
    return float(np.mean(corrs)) if corrs else 0.0

def interaction_density(X_df: pd.DataFrame, threshold: float=0.7) -> float:
    num = X_df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return 0.0
    corr = num.corr().abs()
    n = corr.shape[0]
    total_pairs = n*(n-1)/2
    strong = 0
    for i in range(n):
        for j in range(i+1,n):
            if corr.iat[i,j] >= threshold:
                strong += 1
    return float(strong/total_pairs) if total_pairs>0 else 0.0

def imbalance_penalty(y: pd.Series) -> float:
    counts = y.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0.0
    max_share = counts.max()
    ideal = 1.0 / len(counts)
    return float((max_share - ideal) / (1 - ideal))

def noise_index(model, X_train, y_train, scoring: str, cv: int=4) -> float:
    try:
        model.fit(X_train,y_train)
        train_score = (accuracy_score(y_train, model.predict(X_train)) if scoring=="accuracy" else r2_score(y_train, model.predict(X_train)))
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_mean = float(np.nanmean(cv_scores))
        return float(abs(train_score - cv_mean))
    except Exception:
        return 0.0

def compute_efficiency(model, X_train, y_train, X_test, y_test, problem_type: str) -> Tuple[float,float]:
    t0 = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception:
        return float('nan'), float('nan')
    t1 = time.time()
    train_time = t1 - t0
    try:
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds) if problem_type=="classification" else r2_score(y_test, preds)
        return float(score), float(train_time)
    except Exception:
        return float('nan'), float(train_time)

def plot_confusion_matrix(y_true, y_pred) -> io.BytesIO:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_roc_curve(y_true, y_prob) -> Optional[io.BytesIO]:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0,1],[0,1], '--', alpha=0.3)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
        ax.legend(loc='lower right')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None

def plot_pred_vs_actual(y_true, y_pred) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(4,3))
    ax.scatter(y_true, y_pred, alpha=0.6)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    ax.plot([mn,mx],[mn,mx], '--', color='k', alpha=0.5)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Predicted vs Actual")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_feature_importance(importances) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(5,3))
    idx = np.argsort(importances)[::-1]
    ax.bar(range(len(importances)), importances[idx])
    ax.set_xlabel("Feature (index)")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def draw_accuracy_gauge(pct: float) -> io.BytesIO:
    # semicircle gauge: pct in 0..1
    fig, ax = plt.subplots(figsize=(4,2.2))
    ax.axis('off')
    # draw background arc
    theta = np.linspace(pi, 0, 100)
    ax.plot(np.cos(theta), np.sin(theta), color='#ddd', linewidth=20, solid_capstyle='round')
    # draw value arc
    theta_val = np.linspace(pi, pi - pi*pct, 100)
    ax.plot(np.cos(theta_val), np.sin(theta_val), color='#4f46e5', linewidth=20, solid_capstyle='round')
    # needle
    angle = pi - pi*pct
    ax.plot([0, 0.9*np.cos(angle)], [0, 0.9*np.sin(angle)], color='#111827', linewidth=3)
    ax.text(0, -0.2, f"{pct*100:.1f}%", ha='center', va='center', fontsize=14, fontweight='bold')
    plt.xlim(-1.1,1.1); plt.ylim(-0.3,1.05)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
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
# Sidebar: upload & options
# ---------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=AutoDFit", use_column_width=False)
    st.markdown("## Configuration")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.markdown("---")
    st.markdown("### Advanced options")
    enable_feature_search = st.checkbox("Enable Random Feature Search (light GA)", value=False)
    if enable_feature_search:
        search_iters = st.slider("Random subsets (iterations)", 10, 200, 40, step=10)
        subset_frac = st.slider("Subset fraction (per try)", 0.2, 0.9, 0.6, step=0.1)
    else:
        search_iters = 40; subset_frac = 0.6
    enable_pdf_images = st.checkbox("Include plots in PDF", value=True)
    st.markdown("---")
    st.write("Tip: use small demo CSVs (Iris, Titanic) for fast runs.")

if uploaded_file is None:
    st.markdown('<div class="header">AutoDFit — Dataset Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Upload a CSV to analyse dataset fitness, separability, model comparison and to generate a PDF report.</div>', unsafe_allow_html=True)
    st.info("Upload a CSV from the sidebar to begin.")
    st.stop()

# load dataframe
df = load_csv(uploaded_file)
if df is None:
    st.stop()

# BASIC OVERVIEW
st.markdown('<div class="header">Dataset Overview</div>', unsafe_allow_html=True)
stats = compute_basic_stats(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", stats["rows"])
c2.metric("Columns", stats["cols"])
c3.metric("Missing cells", stats["missing_cells"])
c4.metric("Completeness", f"{stats['completeness']*100:.1f}%")
st.dataframe(df.head(8), use_container_width=True)

# Data Quality tab-like section
st.markdown("## Data Quality & Preprocessing")
target_col = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)
X_df = df.drop(columns=[target_col])
y_ser = df[target_col]

# detect problem type
problem_type = "classification" if (y_ser.dtype == "object" or y_ser.nunique() < 20) else "regression"
st.write(f"**Problem type:** {problem_type}")

# Preprocessor
num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
transformers = []
if num_cols:
    transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols))
if not transformers:
    st.error("No usable features found after dropping the target.")
    st.stop()
preprocessor = ColumnTransformer(transformers, remainder="drop")

# safe stratify: only if classification and each label has at least 2 samples
stratify_col = None
if problem_type == "classification":
    vc = y_ser.value_counts()
    if vc.min() > 1:
        stratify_col = y_ser
    else:
        stratify_col = None

# split + transform
X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y_ser, test_size=0.2, random_state=42, stratify=stratify_col)
with st.spinner("Preprocessing data..."):
    try:
        X_train = preprocessor.fit_transform(X_train_df)
        X_test = preprocessor.transform(X_test_df)
    except Exception as e:
        st.error("Preprocessing failed: " + str(e))
        st.stop()

# compute QC metrics
interaction = interaction_density(X_train_df)
imb_pen = imbalance_penalty(y_train) if problem_type=="classification" else 0.0
base_model = DecisionTreeClassifier() if problem_type=="classification" else DecisionTreeRegressor()
scoring = "accuracy" if problem_type=="classification" else "r2"
noise = noise_index(base_model, X_train, y_train, scoring, cv=4)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Interaction Density", f"{interaction:.3f}")
col_b.metric("Imbalance Penalty", f"{imb_pen:.3f}" if problem_type=="classification" else "N/A")
col_c.metric("Noise Index (|train-CV|)", f"{noise:.3f}")

# store session
st.session_state.update({
    "df": df,
    "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test,
    "preprocessor": preprocessor,
    "problem_type": problem_type,
    "stats": stats,
    "interaction": interaction,
    "imb_pen": imb_pen,
    "noise": noise
})

# Separability
st.markdown("## Separability")
if problem_type == "classification":
    sep = separability_classification(X_train, y_train)
    sep_norm = min(sep/3.0, 1.0)
    st.metric("Separability Index (inter/intra)", f"{sep:.3f}")
else:
    sep = separability_regression(X_train, y_train)
    sep_norm = min(sep, 1.0)
    st.metric("Separability (mean |corr|)", f"{sep:.3f}")
st.session_state["sep"] = sep
st.session_state["sep_norm"] = sep_norm

# Model Benchmark
st.markdown("## Model Benchmark")
if problem_type == "classification":
    model_pool = {
        "Logistic Regression": LogisticRegression(max_iter=400),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM (RBF)": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }
else:
    model_pool = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
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
    progress.progress((i+1)/len(model_pool))
progress.empty(); status.empty()
results_df = pd.DataFrame(results)
results_df["efficiency"] = results_df.apply(lambda r: (r["score"]/r["train_time_s"]) if (r["train_time_s"]>0 and not np.isnan(r["score"])) else np.nan, axis=1)
results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)
st.dataframe(results_df.style.format({"score":"{:.4f}", "train_time_s":"{:.2f}", "efficiency":"{:.4f}"}), use_container_width=True)

# best model
if results_df.shape[0] == 0 or results_df["score"].isnull().all():
    st.error("No model produced valid scores.")
    st.stop()

best_row = results_df.iloc[0]
best_model_name = best_row["model"]
best_model_obj = model_pool[best_model_name]
try:
    best_model_obj.fit(X_train, y_train)
    st.session_state["best_model"] = best_model_obj
except Exception as e:
    st.warning("Retrain of best model failed: " + str(e))

st.markdown(f"### Best model: {best_model_name} — score: {best_row['score']:.4f}")
gauge_buf = draw_accuracy_gauge(best_row['score'] if not np.isnan(best_row['score']) else 0.0)
st.image(gauge_buf, width=320)

# Performance bar chart
fig, ax = plt.subplots(figsize=(8,3))
vals = results_df["score"].fillna(0).values
names = results_df["model"].tolist()
ax.bar(range(len(vals)), vals * (100 if problem_type=="classification" else 1), color='#4f46e5')
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_ylabel("Score (%)" if problem_type=="classification" else "R²")
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Optional random feature search (light GA replacement)
if enable_feature_search:
    st.markdown("## Random feature search (light)")
    n_features = X_train.shape[1]
    if n_features <= 1:
        st.info("Not enough features for search.")
    else:
        best_subset = None
        best_subset_score = -1.0
        for it in range(search_iters):
            k = max(1, int(n_features * subset_frac * random.uniform(0.5,1.0)))
            sel = sorted(random.sample(range(n_features), k))
            try:
                m = RandomForestClassifier(n_estimators=60) if problem_type=="classification" else RandomForestRegressor(n_estimators=60)
                m.fit(X_train[:, sel], y_train)
                preds = m.predict(X_test[:, sel])
                s = accuracy_score(y_test, preds) if problem_type=="classification" else r2_score(y_test, preds)
                if s > best_subset_score:
                    best_subset_score = s
                    best_subset = sel
            except Exception:
                continue
        st.write("Best subset score:", best_subset_score)
        if best_subset is not None:
            st.write("Selected feature count:", len(best_subset))

# Feature importances (if available)
if hasattr(best_model_obj, "feature_importances_"):
    st.markdown("### Feature importances (best model)")
    fi = best_model_obj.feature_importances_
    fig_fi = plt.figure(figsize=(6,3))
    plt.bar(range(len(fi)), fi)
    plt.xlabel("Feature index (post-preprocessing)")
    plt.tight_layout()
    st.pyplot(fig_fi)
    plt.close(fig_fi)

# Evaluate & plots for best model
if problem_type == "classification":
    try:
        preds = best_model_obj.predict(X_test)
        st.image(plot_confusion_matrix(y_test, preds))
    except Exception:
        st.warning("Confusion matrix unavailable.")
    if hasattr(best_model_obj, "predict_proba") and len(np.unique(y_test)) == 2:
        try:
            probs = best_model_obj.predict_proba(X_test)[:,1]
            roc_buf = plot_roc_curve(y_test, probs)
            if roc_buf:
                st.image(roc_buf)
        except Exception:
            pass
else:
    try:
        preds = best_model_obj.predict(X_test)
        st.image(plot_pred_vs_actual(y_test, preds))
    except Exception:
        pass

# Store results for report
st.session_state["results_df"] = results_df
st.session_state["best_model_name"] = best_model_name
st.session_state["best_model_score"] = float(best_row["score"])

# REPORT generation
st.markdown("## Report & Export")
if st.button("Generate PDF report"):
    if "results_df" not in st.session_state:
        st.warning("Run benchmark first.")
    else:
        with st.spinner("Building PDF..."):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=50, bottomMargin=40)
            styles = getSampleStyleSheet()
            story = []

            # Title & summary
            story.append(Paragraph("AutoDFit — Dataset Intelligence Report", styles["Title"]))
            story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
            story.append(Spacer(1, 12))

            # Exec summary
            story.append(Paragraph("Executive Summary", styles["Heading2"]))
            story.append(Paragraph(f"Dataset fitness overview and model benchmark generated by AutoDFit.", styles["Normal"]))
            story.append(Spacer(1, 8))

            # Dataset overview table
            story.append(Paragraph("Dataset Overview", styles["Heading2"]))
            ds_table = [["Metric","Value"],
                        ["Rows", str(stats["rows"])],
                        ["Columns", str(stats["cols"])],
                        ["Missing cells", str(stats["missing_cells"])],
                        ["Completeness", f"{stats['completeness']*100:.1f}%"],
                        ["Target", str(target_col)],
                        ["Problem type", problem_type.capitalize()]]
            tbl = Table(ds_table, colWidths=[2.5*inch, 2.5*inch])
            tbl.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey), ('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
            story.append(tbl)
            story.append(Spacer(1, 8))

            # Metrics table
            story.append(Paragraph("Quality & Separability", styles["Heading2"]))
            sep_table = [["Metric","Value"],
                         ["Separability", f"{st.session_state.get('sep', 0):.3f}"],
                         ["Interaction density", f"{st.session_state.get('interaction', 0):.3f}"],
                         ["Noise index", f"{st.session_state.get('noise', 0):.3f}"],
                         ["Imbalance penalty", f"{st.session_state.get('imb_pen', 0):.3f}"]]
            stbl = Table(sep_table, colWidths=[2.5*inch, 2.5*inch])
            stbl.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
            story.append(stbl)
            story.append(Spacer(1, 8))

            # Model performance
            story.append(Paragraph("Model Benchmark", styles["Heading2"]))
            perf = [["Model","Score","Train time (s)","Efficiency"]]
            for _, r in results_df.iterrows():
                perf.append([r['model'], f"{r['score']:.4f}" if not np.isnan(r['score']) else "N/A",
                             f"{r['train_time_s']:.2f}", f"{r['efficiency']:.4f}" if not np.isnan(r['efficiency']) else "N/A"])
            perf_tbl = Table(perf, colWidths=[2.3*inch,1.1*inch,1.1*inch,1.2*inch])
            perf_tbl.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey), ('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
            story.append(perf_tbl)
            story.append(Spacer(1, 8))

            # Add saved plots (in-memory)
            if enable_pdf_images:
                # confusion / regression / feature importance
                imgs = []
                try:
                    if problem_type=="classification":
                        preds = st.session_state["best_model"].predict(st.session_state["X_test"])
                        imgs.append(plot_confusion_matrix(st.session_state["y_test"], preds))
                        if hasattr(st.session_state["best_model"], "predict_proba") and len(np.unique(st.session_state["y_test"]))==2:
                            probs = st.session_state["best_model"].predict_proba(st.session_state["X_test"])[:,1]
                            roc_buf = plot_roc_curve(st.session_state["y_test"], probs)
                            if roc_buf: imgs.append(roc_buf)
                    else:
                        preds = st.session_state["best_model"].predict(st.session_state["X_test"])
                        imgs.append(plot_pred_vs_actual(st.session_state["y_test"], preds))
                except Exception:
                    pass
                try:
                    if hasattr(st.session_state["best_model"], "feature_importances_"):
                        imgs.append(plot_feature_importance(st.session_state["best_model"].feature_importances_))
                except Exception:
                    pass

                if imgs:
                    story.append(PageBreak())
                    story.append(Paragraph("Visualisations", styles["Heading2"]))
                    for b in imgs:
                        # ReportLab Image accepts file-like
                        try:
                            story.append(RLImage(b, width=4*inch, height=3*inch))
                            story.append(Spacer(1, 6))
                        except Exception:
                            continue

            # Finish
            story.append(PageBreak())
            story.append(Paragraph("Report generated by AutoDFit", styles["Normal"]))
            doc.build(story)
            buffer.seek(0)
            st.download_button("Download PDF report", data=buffer, file_name="AutoDFit_Report.pdf", mime="application/pdf")
            st.success("PDF ready.")

# Model download
if "best_model" in st.session_state and st.session_state["best_model"] is not None:
    try:
        st.download_button("Download trained model (.pkl)", data=pickle.dumps(st.session_state["best_model"]), file_name="autodfit_model.pkl")
    except Exception:
        st.info("Model download not available for this object type.")

st.write("---")
st.caption("AutoDFit — lightweight production-ready pipeline. For heavy explainability (SHAP) or advanced GA use, run locally with a suitable environment.")
