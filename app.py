# app.py — AutoDFit Production-Ready Version
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
from typing import Tuple, Dict, Any, List, Optional
import base64

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, roc_curve, auc

# Report generation
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak,
    Table, TableStyle, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="AutoDFit – Dataset Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS for polished look
# ---------------------------
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 0.75rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .info-box {
        background: #eef2ff;
        border-left: 4px solid #4f46e5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Session state initialisation
# ---------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'fitness_score' not in st.session_state:
    st.session_state.fitness_score = None

# ---------------------------
# Helper functions
# ---------------------------

@st.cache_data
def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load CSV from uploaded file."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    rows, cols = df.shape
    missing_cells = int(df.isnull().sum().sum())
    total_cells = df.size
    completeness = 1 - missing_cells / total_cells if total_cells > 0 else 0
    uniqueness = df.nunique().sum() / total_cells if total_cells > 0 else 0
    return {
        "rows": rows,
        "cols": cols,
        "missing_cells": missing_cells,
        "completeness": completeness,
        "uniqueness": uniqueness
    }

def separability_classification(X: np.ndarray, y: np.ndarray) -> float:
    """Centroid distance ratio (inter / intra)."""
    X = np.array(X)
    y = np.array(y)
    classes = np.unique(y)
    centroids = []
    spreads = []
    for c in classes:
        mask = y == c
        data = X[mask]
        if data.shape[0] == 0:
            centroids.append(np.zeros(X.shape[1]))
            spreads.append(0.0)
            continue
        centroid = np.mean(data, axis=0)
        centroids.append(centroid)
        dists = np.linalg.norm(data - centroid, axis=1)
        spreads.append(np.mean(dists) if len(dists) > 0 else 0.0)
    intra = np.mean(spreads) if spreads else 0.0
    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    inter = np.mean(inter_dists) if inter_dists else 0.0
    return float(inter / (intra + 1e-12))

def separability_regression(X: np.ndarray, y: np.ndarray) -> float:
    """Mean absolute correlation with target."""
    X = np.array(X)
    y = np.array(y).astype(float)
    corrs = []
    for i in range(X.shape[1]):
        col = X[:, i].astype(float)
        if np.std(col) == 0 or np.std(y) == 0:
            corrs.append(0.0)
        else:
            corr = np.corrcoef(col, y)[0, 1]
            corrs.append(abs(corr) if not np.isnan(corr) else 0.0)
    return float(np.mean(corrs)) if corrs else 0.0

def interaction_density(X_df: pd.DataFrame, threshold: float = 0.7) -> float:
    """Fraction of feature pairs with correlation >= threshold."""
    num = X_df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return 0.0
    corr = num.corr().abs()
    n = corr.shape[0]
    total_pairs = n * (n - 1) / 2
    strong = 0
    for i in range(n):
        for j in range(i+1, n):
            if corr.iat[i, j] >= threshold:
                strong += 1
    return float(strong / total_pairs) if total_pairs > 0 else 0.0

def imbalance_penalty(y: pd.Series) -> float:
    """Normalised penalty for class imbalance (0 = balanced)."""
    counts = y.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0.0
    max_share = counts.max()
    ideal = 1.0 / len(counts)
    return float((max_share - ideal) / (1 - ideal))  # 0..1

def noise_index(model, X_train, y_train, scoring: str, cv: int = 4) -> float:
    """|train score - CV mean| as noise proxy."""
    try:
        model.fit(X_train, y_train)
        train_score = (accuracy_score(y_train, model.predict(X_train)) if scoring == "accuracy"
                       else r2_score(y_train, model.predict(X_train)))
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_mean = np.nanmean(cv_scores)
        return float(abs(train_score - cv_mean))
    except Exception:
        return 0.0

def compute_efficiency(model, X_train, y_train, X_test, y_test, problem_type: str) -> Tuple[float, float]:
    """Train model and return (score, training_time_seconds)."""
    t0 = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception:
        return float('nan'), float('nan')
    t1 = time.time()
    train_time = t1 - t0
    try:
        preds = model.predict(X_test)
        if problem_type == "classification":
            score = accuracy_score(y_test, preds)
        else:
            score = r2_score(y_test, preds)
        return float(score), train_time
    except Exception:
        return float('nan'), train_time

def plot_confusion_matrix(y_true, y_pred) -> io.BytesIO:
    """Generate confusion matrix plot and return as BytesIO."""
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

def plot_roc_curve(y_true, y_prob) -> Optional[io.BytesIO]:
    """Generate ROC curve plot and return as BytesIO."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
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
    """Scatter plot for regression."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Actual')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_feature_importance(importances, feature_names=None) -> io.BytesIO:
    """Bar plot of feature importances."""
    fig, ax = plt.subplots(figsize=(5, 3))
    indices = np.argsort(importances)[::-1]
    ax.bar(range(len(importances)), importances[indices])
    ax.set_title('Feature Importances')
    ax.set_xlabel('Feature (index)')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def fitness_score(stats, sep_norm, interaction, noise, imbalance, weights=None):
    """Compute weighted fitness score."""
    if weights is None:
        weights = {
            "completeness": 0.22,
            "separability": 0.28,
            "uniqueness": 0.12,
            "interaction": -0.10,   # penalty -> subtract
            "noise": -0.18,
            "imbalance": -0.10
        }
    # Convert penalties to positive contributions
    fitness = (
        weights["completeness"] * stats["completeness"] +
        weights["separability"] * sep_norm +
        weights["uniqueness"] * stats["uniqueness"] +
        weights["interaction"] * (1 - interaction) +
        weights["noise"] * (1 - noise) +
        weights["imbalance"] * (1 - imbalance)
    )
    # Normalise to 0-100
    return float(np.clip((fitness + 0.5) * 100, 0, 100))

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=AutoDFit", use_column_width=True)  # replace with your logo
    st.markdown("## **Configuration**")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df = load_data(uploaded_file)

    st.markdown("---")
    st.markdown("### **Advanced Options**")
    enable_ga = st.checkbox("Enable Genetic Feature Search (slow)", value=False)
    if enable_ga:
        ga_pop = st.slider("Population size", 10, 50, 20)
        ga_gen = st.slider("Generations", 2, 10, 5)
    else:
        ga_pop, ga_gen = 20, 5  # default values not used
    st.markdown("---")
    st.markdown("**About**")
    st.info("AutoDFit evaluates dataset quality, separability, and recommends models. Generates professional PDF reports.")

if st.session_state.df is None:
    st.markdown('<div class="main-header">AutoDFit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dataset Intelligence & Fitness Evaluation</div>', unsafe_allow_html=True)
    st.info("👈 Please upload a CSV file to begin.")
    st.stop()

df = st.session_state.df

# ---------------------------
# Main interface with tabs
# ---------------------------
tabs = st.tabs(["📋 Overview", "📊 Data Quality", "🔬 Separability", "🤖 Model Benchmark", "📄 Report"])

with tabs[0]:
    st.markdown('<div class="main-header">Dataset Overview</div>', unsafe_allow_html=True)
    stats = compute_basic_stats(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-label">Rows</div><div class="metric-value">{}</div></div>'.format(stats['rows']), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-label">Columns</div><div class="metric-value">{}</div></div>'.format(stats['cols']), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-label">Missing cells</div><div class="metric-value">{}</div></div>'.format(stats['missing_cells']), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-label">Completeness</div><div class="metric-value">{:.1%}</div></div>'.format(stats['completeness']), unsafe_allow_html=True)

    st.markdown("### **Data Preview**")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### **Column Summary**")
    col_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Unique Values': df.nunique(),
        'Missing %': (df.isnull().sum() / len(df)) * 100
    })
    st.dataframe(col_info, use_container_width=True)

with tabs[1]:
    st.markdown('<div class="main-header">Data Quality Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">These metrics help assess dataset health before modelling.</div>', unsafe_allow_html=True)

    target_col = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)
    X_df = df.drop(columns=[target_col])
    y = df[target_col]

    # Detect problem type
    if y.dtype == 'object' or y.nunique() <= 20:
        problem_type = "classification"
    else:
        problem_type = "regression"
    st.write(f"**Problem type:** {problem_type}")

    # Preprocessing split
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Build preprocessor
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_cols))
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    # Train/test split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42,
        stratify=y if problem_type == "classification" else None
    )

    # Transform
    with st.spinner("Preprocessing..."):
        X_train = preprocessor.fit_transform(X_train_df)
        X_test = preprocessor.transform(X_test_df)

    # Compute quality metrics
    interaction = interaction_density(X_train_df)
    imb_pen = imbalance_penalty(y_train) if problem_type == "classification" else 0.0

    # Noise index with a simple tree
    base_model = DecisionTreeClassifier() if problem_type == "classification" else DecisionTreeRegressor()
    scoring = "accuracy" if problem_type == "classification" else "r2"
    noise = noise_index(base_model, X_train, y_train, scoring, cv=4)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Interaction Density</div><div class="metric-value">{interaction:.3f}</div><div style="font-size:0.8rem;">High → redundant features</div></div>', unsafe_allow_html=True)
    with col2:
        if problem_type == "classification":
            st.markdown(f'<div class="metric-card"><div class="metric-label">Imbalance Penalty</div><div class="metric-value">{imb_pen:.3f}</div><div style="font-size:0.8rem;">0 = balanced</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Imbalance Penalty</div><div class="metric-value">N/A</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Noise Index</div><div class="metric-value">{noise:.3f}</div><div style="font-size:0.8rem;">|train - CV|</div></div>', unsafe_allow_html=True)

    # Store in session for later tabs
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.problem_type = problem_type
    st.session_state.preprocessor = preprocessor
    st.session_state.stats = stats
    st.session_state.interaction = interaction
    st.session_state.imb_pen = imb_pen
    st.session_state.noise = noise
    st.session_state.target_col = target_col

with tabs[2]:
    st.markdown('<div class="main-header">Separability Analysis</div>', unsafe_allow_html=True)
    if 'X_train' not in st.session_state:
        st.warning("Please run Data Quality tab first.")
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        problem_type = st.session_state.problem_type

        if problem_type == "classification":
            sep = separability_classification(X_train, y_train)
            sep_norm = min(sep / 3.0, 1.0)  # typical range 0-3
            st.markdown(f'<div class="metric-card"><div class="metric-label">Separability Index</div><div class="metric-value">{sep:.3f}</div><div style="font-size:0.9rem;">(inter/intra class distance)</div></div>', unsafe_allow_html=True)
        else:
            sep = separability_regression(X_train, y_train)
            sep_norm = min(sep, 1.0)
            st.markdown(f'<div class="metric-card"><div class="metric-label">Separability (mean |corr|)</div><div class="metric-value">{sep:.3f}</div></div>', unsafe_allow_html=True)

        st.session_state.sep = sep
        st.session_state.sep_norm = sep_norm

with tabs[3]:
    st.markdown('<div class="main-header">Model Benchmark</div>', unsafe_allow_html=True)
    if 'X_train' not in st.session_state:
        st.warning("Please run Data Quality tab first.")
    else:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        problem_type = st.session_state.problem_type

        # Model pool
        if problem_type == "classification":
            model_pool = {
                "Logistic Regression": LogisticRegression(max_iter=300),
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
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (name, model) in enumerate(model_pool.items()):
            status_text.text(f"Training {name}...")
            score, ttime = compute_efficiency(model, X_train, y_train, X_test, y_test, problem_type)
            results.append({"model": name, "score": score, "train_time_s": ttime})
            progress_bar.progress((i+1)/len(model_pool))

        progress_bar.empty()
        status_text.empty()

        results_df = pd.DataFrame(results)
        # Efficiency: score per second (handle zeros/inf)
        results_df["efficiency"] = results_df.apply(
            lambda r: (r["score"] / r["train_time_s"]) if (r["train_time_s"] > 0 and not np.isnan(r["score"])) else np.nan,
            axis=1
        )
        results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)
        st.session_state.results_df = results_df

        st.markdown("### **Leaderboard**")
        st.dataframe(
            results_df.style.format({
                "score": "{:.4f}",
                "train_time_s": "{:.2f}",
                "efficiency": "{:.4f}"
            }),
            use_container_width=True
        )

        # Best model
        best_row = results_df.iloc[0]
        best_model_name = best_row["model"]
        best_model_obj = model_pool[best_model_name]
        # Retrain on full training data (already fitted in compute_efficiency, but ensure)
        try:
            best_model_obj.fit(X_train, y_train)
            st.session_state.best_model = best_model_obj
        except Exception as e:
            st.error(f"Could not retrain best model: {e}")

        st.markdown(f"### **Best Model: {best_model_name}**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score", f"{best_row['score']:.4f}")
        with col2:
            st.metric("Training Time", f"{best_row['train_time_s']:.2f} s")

        # Visual comparison
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(results_df["model"], results_df["score"] * (100 if problem_type=="classification" else 1))
        ax.set_ylabel("Score (%)" if problem_type=="classification" else "R² Score")
        ax.set_xticklabels(results_df["model"], rotation=30, ha='right')
        ax.set_title("Model Performance Comparison")
        for bar, score in zip(bars, results_df["score"]):
            height = bar.get_height()
            ax.annotate(f'{score:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Feature importance if available
        if hasattr(best_model_obj, "feature_importances_"):
            st.markdown("### **Feature Importances**")
            fi = best_model_obj.feature_importances_
            fig_fi = plt.figure(figsize=(6,3))
            plt.bar(range(len(fi)), fi)
            plt.xlabel("Feature (transformed index)")
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close(fig_fi)

with tabs[4]:
    st.markdown('<div class="main-header">Report & Export</div>', unsafe_allow_html=True)
    if 'results_df' not in st.session_state or st.session_state.results_df is None:
        st.warning("Please run Model Benchmark first.")
    else:
        # Fitness score (combine metrics)
        stats = st.session_state.stats
        sep_norm = st.session_state.get('sep_norm', 0.5)
        interaction = st.session_state.get('interaction', 0)
        noise = st.session_state.get('noise', 0)
        imb_pen = st.session_state.get('imb_pen', 0)

        fitness = fitness_score(stats, sep_norm, interaction, noise, imb_pen)
        st.session_state.fitness_score = fitness

        st.markdown(f'<div class="metric-card"><div class="metric-label">Dataset Fitness Score</div><div class="metric-value">{fitness:.1f}%</div><div>0-40% Poor, 40-70% Moderate, 70-100% Good</div></div>', unsafe_allow_html=True)

        st.markdown("### **Download Options**")

        # Download model
        if st.session_state.best_model is not None:
            model_bytes = pickle.dumps(st.session_state.best_model)
            st.download_button(
                "📦 Download Trained Model (pickle)",
                data=model_bytes,
                file_name="autodfit_model.pkl",
                mime="application/octet-stream"
            )

        # Download results CSV
        csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📊 Download Results CSV",
            data=csv,
            file_name="autodfit_results.csv",
            mime="text/csv"
        )

        # PDF Generation (in-memory)
        st.markdown("### **Generate Professional PDF Report**")
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Building PDF report..."):
                # Prepare plots in memory
                plot_bufs = {}
                if enable_pdf_images:
                    # Confusion matrix or regression plot
                    if st.session_state.problem_type == "classification":
                        try:
                            y_pred = st.session_state.best_model.predict(st.session_state.X_test)
                            plot_bufs['cm'] = plot_confusion_matrix(st.session_state.y_test, y_pred)
                        except Exception:
                            pass
                        # ROC if probabilities available
                        if hasattr(st.session_state.best_model, "predict_proba"):
                            try:
                                y_prob = st.session_state.best_model.predict_proba(st.session_state.X_test)[:,1]
                                roc_buf = plot_roc_curve(st.session_state.y_test, y_prob)
                                if roc_buf:
                                    plot_bufs['roc'] = roc_buf
                            except Exception:
                                pass
                    else:
                        # Regression
                        try:
                            y_pred = st.session_state.best_model.predict(st.session_state.X_test)
                            plot_bufs['reg'] = plot_pred_vs_actual(st.session_state.y_test, y_pred)
                        except Exception:
                            pass

                    # Feature importance
                    if hasattr(st.session_state.best_model, "feature_importances_"):
                        try:
                            fi = st.session_state.best_model.feature_importances_
                            plot_bufs['fi'] = plot_feature_importance(fi)
                        except Exception:
                            pass

                # Build PDF
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(
                    buffer,
                    pagesize=A4,
                    rightMargin=36,
                    leftMargin=36,
                    topMargin=50,
                    bottomMargin=40
                )
                styles = getSampleStyleSheet()
                story = []

                # Title
                title_style = ParagraphStyle(
                    'Title',
                    parent=styles['Heading1'],
                    fontSize=24,
                    spaceAfter=12,
                    textColor=colors.HexColor('#0f172a')
                )
                story.append(Paragraph("AutoDFit – Dataset Intelligence Report", title_style))
                story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
                story.append(Spacer(1, 0.2*inch))

                # Executive summary
                story.append(Paragraph("Executive Summary", styles['Heading2']))
                story.append(Paragraph(f"Dataset fitness score: {fitness:.1f}%", styles['Normal']))
                story.append(Paragraph(f"Best model: {st.session_state.results_df.iloc[0]['model']} with score {st.session_state.results_df.iloc[0]['score']:.4f}", styles['Normal']))
                story.append(Spacer(1, 0.1*inch))

                # Dataset overview
                story.append(Paragraph("Dataset Overview", styles['Heading2']))
                data_summary = [
                    ["Metric", "Value"],
                    ["Rows", str(stats['rows'])],
                    ["Columns", str(stats['cols'])],
                    ["Missing cells", str(stats['missing_cells'])],
                    ["Completeness", f"{stats['completeness']*100:.1f}%"],
                    ["Target column", st.session_state.target_col],
                    ["Problem type", st.session_state.problem_type.capitalize()]
                ]
                tbl = Table(data_summary, colWidths=[2*inch, 2*inch])
                tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,-1), 10),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 0.2*inch))

                # Separability & fitness
                story.append(Paragraph("Separability & Quality Metrics", styles['Heading2']))
                sep_data = [
                    ["Separability Index", f"{st.session_state.sep:.3f}"],
                    ["Interaction Density", f"{interaction:.3f}"],
                    ["Noise Index", f"{noise:.3f}"],
                ]
                if st.session_state.problem_type == "classification":
                    sep_data.append(["Imbalance Penalty", f"{imb_pen:.3f}"])
                sep_tbl = Table(sep_data, colWidths=[2*inch, 2*inch])
                sep_tbl.setStyle(TableStyle([
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('FONTSIZE', (0,0), (-1,-1), 10),
                ]))
                story.append(sep_tbl)
                story.append(Spacer(1, 0.2*inch))

                # Model performance table
                story.append(Paragraph("Model Benchmark", styles['Heading2']))
                perf_data = [["Model", "Score", "Train Time (s)", "Efficiency"]]
                for _, row in st.session_state.results_df.iterrows():
                    perf_data.append([
                        row['model'],
                        f"{row['score']:.4f}",
                        f"{row['train_time_s']:.2f}",
                        f"{row['efficiency']:.4f}" if not np.isnan(row['efficiency']) else "N/A"
                    ])
                perf_tbl = Table(perf_data, colWidths=[1.5*inch, 1.0*inch, 1.0*inch, 1.2*inch])
                perf_tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('FONTSIZE', (0,0), (-1,-1), 9),
                ]))
                story.append(perf_tbl)
                story.append(Spacer(1, 0.2*inch))

                # Add plots if any
                if plot_bufs:
                    story.append(PageBreak())
                    story.append(Paragraph("Model Visualizations", styles['Heading2']))
                    for key, buf in plot_bufs.items():
                        img = RLImage(buf, width=4*inch, height=3*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.1*inch))

                # Footer
                story.append(PageBreak())
                story.append(Paragraph("Report generated by AutoDFit – Dataset Intelligence Platform", styles['Italic']))

                # Build PDF
                doc.build(story)
                buffer.seek(0)

                st.download_button(
                    "📥 Download PDF Report",
                    data=buffer,
                    file_name="AutoDFit_Report.pdf",
                    mime="application/pdf"
                )
                st.success("PDF generated successfully!")
