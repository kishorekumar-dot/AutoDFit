# app.py — AutoDFit (production-ready single-file)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import random
import pickle
import os
from datetime import datetime
from typing import Tuple, Dict, Any, List

from sklearn.model_selection import train_test_split, cross_val_score
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

# Report generation
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------
# UI: fonts + styles
# ---------------------------
st.set_page_config(page_title="AutoDFit", layout="wide", initial_sidebar_state="expanded")

# Inject CSS for fonts and theme
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
      html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
      .title { font-weight:700; color: #0f172a; }
      .metric-big { font-size: 1.1rem; }
      .card { background: linear-gradient(180deg,#ffffff,#fbfbff); padding:14px; border-radius:12px;
              box-shadow: 0 6px 18px rgba(16,24,40,0.06); }
      .small-muted { color: #6b7280; font-size:0.9rem; }
      /* dark theme toggle hint - doesn't force system dark, just colors some elements */
      .dark .card { background: #0b1220; color: #e6eef8; }
      .spinner-svg {width:44px; height:44px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# small animated header
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:18px">
      <div style="width:70px">
        <svg viewBox="0 0 120 40" class="spinner-svg">
          <defs>
            <linearGradient id="g" x1="0" x2="1">
              <stop offset="0%" stop-color="#00F5FF"/>
              <stop offset="100%" stop-color="#7B61FF"/>
            </linearGradient>
          </defs>
          <circle cx="20" cy="20" r="12" fill="url(#g)">
            <animate attributeName="r" values="10;14;10" dur="1.8s" repeatCount="indefinite"/>
          </circle>
        </svg>
      </div>
      <div>
        <h1 class="title">AutoDFit — Dataset Intelligence</h1>
        <div class="small-muted">Dataset fitness, separability, recommendation, model comparison & professional report</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Options")
    enable_ga = st.checkbox("Enable Genetic Feature Search (sampled)", value=False)
    ga_pop = st.slider("GA population", min_value=6, max_value=30, value=10, step=2)
    ga_gen = st.slider("GA generations", min_value=2, max_value=8, value=4, step=1)
    enable_pdf_images = st.checkbox("Include visual analytics in PDF", value=True)
    dark_ui = st.checkbox("Dark UI (visual hint)", value=False)
    st.write("Note: keep datasets moderate (<= 50k rows) for responsive experience.")

# small toggle effect for dark (only affects cards via applied class)
if dark_ui:
    st.markdown("<script>document.body.classList.add('dark')</script>", unsafe_allow_html=True)
else:
    st.markdown("<script>document.body.classList.remove('dark')</script>", unsafe_allow_html=True)

# ---------------------------
# Utility functions
# ---------------------------

def safe_read_csv(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

def normalize01(x: float, minv: float, maxv: float) -> float:
    if np.isnan(x) or maxv == minv:
        return 0.0
    return float((x - minv) / (maxv - minv))

# ---------------------------
# Core analytics functions
# ---------------------------

def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    rows, cols = df.shape
    missing_cells = int(df.isnull().sum().sum())
    total = df.size if df.size > 0 else 1
    completeness = (1 - missing_cells / total)
    uniqueness = (df.nunique().sum() / total)
    return {"rows": rows, "cols": cols, "missing_cells": missing_cells,
            "completeness": completeness, "uniqueness": uniqueness}

def separability_classification(X: np.ndarray, y: np.ndarray) -> float:
    # centroid distance ratio (inter / intra)
    dfX = np.array(X)
    labels = np.array(y)
    classes = np.unique(labels)
    centroids = []
    spreads = []
    for c in classes:
        mask = labels == c
        data = dfX[mask]
        if data.shape[0] == 0:
            centroids.append(np.zeros(dfX.shape[1]))
            spreads.append(0.0)
            continue
        centroid = np.mean(data, axis=0)
        centroids.append(centroid)
        d = np.linalg.norm(data - centroid, axis=1)
        spreads.append(np.mean(d) if len(d) > 0 else 0.0)
    intra = np.mean(spreads) if len(spreads) > 0 else 0.0
    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    inter = np.mean(inter_dists) if len(inter_dists) > 0 else 0.0
    if intra == 0:
        return float(inter)
    return float(inter / (intra + 1e-12))

def separability_regression(X: np.ndarray, y: np.ndarray) -> float:
    # mean absolute correlation of features with target (post-preprocessing)
    try:
        arrX = np.array(X)
        yv = np.array(y).astype(float)
        corrs = []
        for i in range(arrX.shape[1]):
            col = arrX[:, i].astype(float)
            if np.std(col) == 0 or np.std(yv) == 0:
                corrs.append(0.0)
                continue
            corr = np.corrcoef(col, yv)[0, 1]
            corrs.append(abs(corr))
        return float(np.nanmean(corrs))
    except Exception:
        return 0.0

def interaction_density(X_df: pd.DataFrame, threshold: float = 0.7) -> float:
    # fraction of strong absolute correlations among numeric features
    num = X_df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return 0.0
    corr = num.corr().abs()
    # count pairs upper triangle excluding diagonal
    n = corr.shape[0]
    total_pairs = n * (n - 1) / 2
    strong = 0
    for i in range(n):
        for j in range(i + 1, n):
            if corr.iat[i, j] >= threshold:
                strong += 1
    return float(strong / total_pairs) if total_pairs > 0 else 0.0

def imbalance_penalty(y: pd.Series) -> float:
    # 0 if perfectly balanced, higher if imbalanced (0..1)
    counts = y.value_counts(normalize=True)
    if counts.empty:
        return 0.0
    max_share = counts.max()
    # penalty is relative to 1 / num_classes
    ideal = 1.0 / len(counts)
    penalty = max(0.0, (max_share - ideal) / (1 - ideal))  # normalized 0..1
    return float(penalty)

def noise_index_model_based(model, X_train, y_train, cv=4, scoring=None) -> float:
    # train score vs cross-validation mean -> difference indicates noise/overfit
    try:
        model.fit(X_train, y_train)
        train_score = (accuracy_score(y_train, model.predict(X_train)) if scoring == "accuracy" else
                       r2_score(y_train, model.predict(X_train)) if scoring == "r2" else None)
    except Exception:
        train_score = None
    try:
        if scoring == "accuracy":
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
            cv_mean = float(np.nanmean(cv_scores))
        elif scoring == "r2":
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)
            cv_mean = float(np.nanmean(cv_scores))
        else:
            cv_mean = None
    except Exception:
        cv_mean = None
    if train_score is None or cv_mean is None:
        return 0.0
    return float(abs(train_score - cv_mean))

def compute_efficiency(model, X_train, y_train, X_test, y_test, problem_type: str) -> Tuple[float, float]:
    # returns (score, train_time_seconds)
    t0 = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception:
        return float("nan"), float("nan")
    t1 = time.time()
    train_time = t1 - t0
    try:
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds) if problem_type == "classification" else r2_score(y_test, preds)
    except Exception:
        score = float("nan")
    return float(score), float(train_time)

# ---------------------------
# File upload and preprocessing
# ---------------------------
uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to start. Example datasets: Iris, Titanic, Diabetes, etc.")
    st.stop()

df = safe_read_csv(uploaded)

# show basic dataset overview
stats = compute_basic_stats(df)
st.header("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", stats["rows"])
col2.metric("Columns", stats["cols"])
col3.metric("Missing cells", stats["missing_cells"])
st.write(f"Dataset completeness: **{stats['completeness']*100:.2f}%**, uniqueness indicator: **{stats['uniqueness']*100:.2f}%**")
st.dataframe(df.head(8))

# target selection
target_col = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)
X_df = df.drop(columns=[target_col]).copy()
y_ser = df[target_col].copy()

# detect problem
is_classification = (y_ser.dtype == "object") or (y_ser.nunique() <= 20 and y_ser.dtype != float)
problem_type = "classification" if is_classification else "regression"
st.success(f"Detected problem type: {problem_type}")

# separate numeric/categorical columns
num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

# build preprocessor
transformers = []
if num_cols:
    transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols))
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

# split
with st.spinner("Splitting and preprocessing data..."):
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y_ser, test_size=0.2, random_state=42, stratify=y_ser if is_classification else None)
    try:
        X_train = preprocessor.fit_transform(X_train_df)
        X_test = preprocessor.transform(X_test_df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

# compute structural metrics
st.header("Structural Analysis")
sep = separability_classification(X_train, y_train) if is_classification else separability_regression(X_train, y_train)
interaction = interaction_density(X_train_df)
imb_pen = imbalance_penalty(y_train) if is_classification else 0.0

st.metric("Data separability", f"{sep:.3f}")
st.metric("Interaction density", f"{interaction:.3f}")
if is_classification:
    st.metric("Imbalance penalty", f"{imb_pen:.3f}")

# noise index using a light base model
st.subheader("Noise & Stability")
if is_classification:
    base_model = DecisionTreeClassifier()
    scoring = "accuracy"
else:
    base_model = DecisionTreeRegressor()
    scoring = "r2"
noise = noise_index_model_based(base_model, X_train, y_train, cv=4, scoring=scoring)
st.write("Noise index (|train - CV|):", f"{noise:.3f}")

# ---------------------------
# Dataset Fitness Score (formal)
# ---------------------------
st.header("Dataset Fitness Score")
# normalize components to 0..1
# completeness and uniqueness from stats
comp = stats["completeness"]
uniq = stats["uniqueness"]
# separability normalization: map typical ranges to 0..1
if is_classification:
    # centroid ratio: values can be >1; we clip scale by mapping 0..3 -> 0..1 (3+ is strong)
    sep_norm = min(sep / 3.0, 1.0)
else:
    # regression separability (mean abs corr) already in 0..1 range
    sep_norm = min(sep, 1.0)

interaction_pen = interaction  # higher interaction density is a penalty (redundancy)
noise_pen = min(noise / 0.5, 1.0)  # normalize assuming 0.5+ is very noisy
imbalance_pen_norm = imb_pen

# define weights
weights = {
    "completeness": 0.22,
    "separability": 0.28,
    "uniqueness": 0.12,
    "interaction": -0.10,  # penalty -> subtract
    "noise": -0.18,        # penalty
    "imbalance": -0.10     # penalty
}
# compute weighted sum (clamp)
fitness = (weights["completeness"] * comp +
           weights["separability"] * sep_norm +
           weights["uniqueness"] * uniq +
           weights["interaction"] * (1 - interaction_pen) +  # convert redundancy to positive
           weights["noise"] * (1 - noise_pen) +
           weights["imbalance"] * (1 - imbalance_pen_norm))

# map fitness to 0..100
fitness_pct = float(np.clip((fitness + 0.5) * 100, 0.0, 100.0))  # shift so approx mid->50
st.metric("Dataset fitness score", f"{fitness_pct:.2f}%")
st.write("Interpretation: 0–40% (poor), 40–70% (moderate), 70–100% (good)")

# show a radar-like summary
with st.expander("Detailed metric breakdown"):
    st.write({
        "completeness": round(comp, 4),
        "separability_norm": round(sep_norm, 4),
        "uniqueness": round(uniq, 4),
        "interaction_density": round(interaction_pen, 4),
        "noise_index": round(noise_pen, 4),
        "imbalance_penalty": round(imbalance_pen_norm, 4)
    })

# ---------------------------
# Model recommendation engine
# ---------------------------
st.header("Model Recommendation & Auto Training")
# Simple data-driven rules
recommendation = "RandomForest (default)"
if is_classification:
    if sep_norm > 0.66 and noise_pen < 0.15:
        recommendation = "Logistic Regression / Linear"
    elif interaction > 0.6:
        recommendation = "Regularized tree-based (RandomForest)"
    elif stats["rows"] < 1000:
        recommendation = "SVM / KNN (small dataset)"
    else:
        recommendation = "RandomForest / Ensemble"
else:
    if sep_norm > 0.6:
        recommendation = "Linear Regression (or regularized)"
    elif stats["cols"] > stats["rows"]/2:
        recommendation = "Regularized model (Ridge/Lasso) or tree-based"
    else:
        recommendation = "RandomForest Regressor"

st.info(f"Recommended model family: **{recommendation}** — (based on separability, size, interactions, noise)")

# model set
if is_classification:
    model_pool = {
        "LogisticRegression": LogisticRegression(max_iter=300),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier()
    }
else:
    model_pool = {
        "LinearRegression": LinearRegression(),
        "RandomForestReg": RandomForestRegressor(n_estimators=100),
        "SVR": SVR(),
        "KNNReg": KNeighborsRegressor(),
        "DecisionTreeReg": DecisionTreeRegressor()
    }

# train & measure efficiency
results = []
st.write("Training selected models and computing efficiency (score + time)...")
progress_text = "Training models..."
my_bar = st.progress(0)
n_models = len(model_pool)
i = 0
for name, mod in model_pool.items():
    i += 1
    my_bar.progress(int(i / n_models * 100))
    score, ttime = compute_efficiency(mod, X_train, y_train, X_test, y_test, problem_type=problem_type)
    results.append({"model": name, "score": score, "train_time_s": ttime})
my_bar.empty()

results_df = pd.DataFrame(results)
results_df["efficiency"] = results_df.apply(lambda r: (r["score"] / (r["train_time_s"] + 1e-6)) if (not np.isnan(r["score"]) and not np.isnan(r["train_time_s"])) else np.nan, axis=1)
results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)
st.subheader("Model leaderboard")
st.dataframe(results_df.style.format({"score": "{:.4f}", "train_time_s": "{:.2f}", "efficiency": "{:.4f}"}))

# best model selection
best_row = results_df.iloc[0]
best_model_name = best_row["model"]
best_model_obj = model_pool[best_model_name]
# retrain best model fully on training data (already trained in compute_efficiency, but ensure fit)
try:
    best_model_obj.fit(X_train, y_train)
except Exception:
    pass

st.metric("Best model", f"{best_model_name}", f"Score: {best_row['score']:.4f}")

# visual comparison chart
fig = plt.figure(figsize=(7, 3))
plt.bar(results_df["model"], results_df["score"] * (100 if problem_type == "classification" else 1))
plt.ylabel("Score" + (" (percent)" if problem_type == "classification" else " (R2)"))
plt.xticks(rotation=20)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------
# Model evaluation visuals
# ---------------------------
st.header("Model Evaluation")
if problem_type == "classification":
    try:
        preds = best_model_obj.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig_cm = plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    except Exception:
        st.info("Confusion matrix unavailable for this model.")
    # ROC if prob available
    if hasattr(best_model_obj, "predict_proba"):
        try:
            probs = best_model_obj.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            aucv = auc(fpr, tpr)
            fig_roc = plt.figure(figsize=(4, 3))
            plt.plot(fpr, tpr, label=f"AUC={aucv:.3f}")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.legend()
            st.pyplot(fig_roc)
            plt.close(fig_roc)
        except Exception:
            pass
else:
    # regression diagnostics: scatter of predicted vs actual
    try:
        preds = best_model_obj.predict(X_test)
        fig_reg = plt.figure(figsize=(5, 4))
        plt.scatter(y_test, preds, alpha=0.6)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "--", color="gray")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")
        st.pyplot(fig_reg)
        plt.close(fig_reg)
    except Exception:
        pass

# feature importance (if available)
st.subheader("Feature importance (if available)")
if hasattr(best_model_obj, "feature_importances_"):
    try:
        fi = best_model_obj.feature_importances_
        fig_fi = plt.figure(figsize=(6,3))
        plt.bar(range(len(fi)), fi)
        plt.xlabel("Feature (post-preprocess index)")
        st.pyplot(fig_fi)
        plt.close(fig_fi)
    except Exception:
        st.info("Feature importance not displayable.")
else:
    st.info("Selected model does not expose feature_importances_")

# ---------------------------
# Genetic algorithm (simple, pure python)
# ---------------------------
best_individual = None
if enable_ga:
    st.header("Genetic Feature Search (light)")
    n_features = X_train.shape[1]
    if n_features <= 1:
        st.info("Not enough features for GA.")
    else:
        # simple GA impl
        def evaluate_ind(individual):
            sel = [i for i, v in enumerate(individual) if v == 1]
            if len(sel) == 0:
                return 0.0
            try:
                if problem_type == "classification":
                    m = RandomForestClassifier(n_estimators=40)
                else:
                    m = RandomForestRegressor(n_estimators=40)
                m.fit(X_train[:, sel], y_train)
                pred = m.predict(X_test[:, sel])
                return float(accuracy_score(y_test, pred) if problem_type == "classification" else r2_score(y_test, pred))
            except Exception:
                return 0.0
        # init
        pop = []
        pop_size = ga_pop
        for _ in range(pop_size):
            ind = [random.choice([0,1]) for _ in range(n_features)]
            pop.append(ind)
        # evol
        for gen in range(ga_gen):
            scores_pop = [(evaluate_ind(ind), ind) for ind in pop]
            scores_pop.sort(key=lambda x: x[0], reverse=True)
            # keep top 40%
            keep = int(len(pop) * 0.4) or 1
            new_pop = [ind for _, ind in scores_pop[:keep]]
            # fill with crossover & mutation
            while len(new_pop) < pop_size:
                p1 = random.choice(scores_pop)[1]
                p2 = random.choice(scores_pop)[1]
                # single point crossover
                pt = random.randint(1, n_features-1)
                child = p1[:pt] + p2[pt:]
                # mutation
                if random.random() < 0.15:
                    mi = random.randint(0, n_features-1)
                    child[mi] = 1 - child[mi]
                new_pop.append(child)
            pop = new_pop
        # select best
        scored = [(evaluate_ind(ind), ind) for ind in pop]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_individual = scored[0][1]
        st.write("GA selected feature count:", sum(best_individual))
        st.write(f"Example selected indices (first 50): { [i for i,v in enumerate(best_individual) if v==1][:50] }")

# ---------------------------
# Download model, results & PDF report
# ---------------------------
st.header("Export & Report")

# Download trained model
try:
    bmodel_bytes = pickle.dumps(best_model_obj)
    st.download_button("Download trained model (.pkl)", data=bmodel_bytes, file_name="autodfit_model.pkl", mime="application/octet-stream")
except Exception:
    st.info("Model download unavailable for this object.")

# Results csv
results_csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=results_csv, file_name="autodfit_results.csv", mime="text/csv")

# PDF report generator
def pdf_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(40, 800, "AutoDFit — Dataset Intelligence")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(550, 20, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()

def generate_pdf():
    tmp_files = []
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, rightMargin=36, leftMargin=36, topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # cover
    story.append(Spacer(1, 140))
    story.append(Paragraph("AutoDFit — Analysis Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    story.append(PageBreak())

    # TOC
    story.append(Paragraph("Table of Contents", styles["Heading2"]))
    for item in ["1 Executive summary", "2 Dataset overview", "3 Structural analysis",
                 "4 Model performance", "5 Genetic search", "6 Conclusion"]:
        story.append(Paragraph(item, styles["Normal"]))
    story.append(PageBreak())

    # Exec summary
    story.append(Paragraph("1 Executive summary", styles["Heading1"]))
    story.append(Paragraph(f"Best model: {best_model_name}", styles["Normal"]))
    try:
        story.append(Paragraph(f"Best score: {best_row['score']:.4f}", styles["Normal"]))
    except Exception:
        pass
    story.append(Paragraph(f"Dataset fitness: {fitness_pct:.2f}%", styles["Normal"]))
    story.append(PageBreak())

    # dataset overview
    story.append(Paragraph("2 Dataset overview", styles["Heading1"]))
    story.append(Paragraph(f"Rows: {stats['rows']}, Columns: {stats['cols']}", styles["Normal"]))
    story.append(Paragraph(f"Missing cells: {stats['missing_cells']}", styles["Normal"]))
    story.append(PageBreak())

    # structural analysis
    story.append(Paragraph("3 Structural analysis", styles["Heading1"]))
    story.append(Paragraph(f"Separability (norm): {sep if is_classification else sep:.4f}", styles["Normal"]))
    story.append(Paragraph(f"Interaction density: {interaction:.4f}", styles["Normal"]))
    story.append(Paragraph(f"Noise index (|train-cv|): {noise:.4f}", styles["Normal"]))
    story.append(PageBreak())

    # model performance table + images
    story.append(Paragraph("4 Model performance", styles["Heading1"]))
    table_data = [["Model", "Score", "Train Time (s)"]]
    for r in results:
        s = r["score"]
        tt = r["train_time_s"]
        table_data.append([r["model"], f"{s:.4f}" if (s is not None and not np.isnan(s)) else "N/A", f"{tt:.2f}" if (tt is not None and not np.isnan(tt)) else "N/A"])
    tbl = Table(table_data)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.grey),("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),("GRID",(0,0),(-1,-1),0.5,colors.black)]))
    story.append(tbl)
    story.append(PageBreak())

    # visual analytics (optionally include images)
    if enable_pdf_images:
        try:
            # heatmap
            plt.figure(figsize=(6,4)); sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm"); plt.tight_layout()
            fn = "tmp_heatmap.png"; plt.savefig(fn); plt.close(); tmp_files.append(fn); story.append(RLImage(fn, width=400, height=280)); story.append(PageBreak())
        except Exception:
            pass

    # GA
    story.append(Paragraph("5 Genetic search results", styles["Heading1"]))
    if best_individual:
        story.append(Paragraph(f"Selected feature count: {sum(best_individual)}", styles["Normal"]))
    else:
        story.append(Paragraph("GA not executed or no result", styles["Normal"]))
    story.append(PageBreak())

    story.append(Paragraph("6 Conclusion", styles["Heading1"]))
    story.append(Paragraph("AutoDFit produced dataset-level diagnostics, model comparison and an exportable package to support decisions on model deployment.", styles["Normal"]))

    doc.build(story, onFirstPage=pdf_header_footer, onLaterPages=pdf_header_footer)

    # cleanup tmp files
    for f in tmp_files:
        try:
            os.remove(f)
        except Exception:
            pass

    buffer.seek(0)
    return buffer

try:
    pdf_bytes = generate_pdf()
    st.download_button("📄 Download Full Report (PDF)", data=pdf_bytes, file_name="AutoDFit_Report.pdf", mime="application/pdf")
except Exception as e:
    st.warning("PDF generation failed: " + str(e))

st.success("Analysis complete — export available above.")
