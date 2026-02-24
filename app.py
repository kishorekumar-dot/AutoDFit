import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import random

# ML
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

from sklearn.metrics import accuracy_score, r2_score

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Genetic Algorithm
from deap import base, creator, tools, algorithms

# =====================================================
# LABEL ENCODER WRAPPER (defined before use)
# =====================================================
class LabelEncoderWrapper:
    def fit(self, X, y=None):
        self.encoders = {}
        for i in range(X.shape[1]):
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i])
            self.encoders[i] = le
        return self

    def transform(self, X):
        for i in range(X.shape[1]):
            X[:, i] = self.encoders[i].transform(X[:, i])
        return X


# =====================================================
# PAGE CONFIG + PREMIUM UI
# =====================================================
st.set_page_config(page_title="AutoDFit", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#0F1117,#151823,#1B1E2B);
}
.stMetric {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    padding:15px;
    border-radius:12px;
}
.hero {
    text-align:center;
    font-size:42px;
    font-weight:700;
    background: linear-gradient(90deg,#00F5FF,#7B61FF);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.subtitle {
    text-align:center;
    color:#9CA3AF;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# ANIMATED LOGO (INLINE SVG — DEPLOYMENT SAFE)
# =====================================================
st.markdown("""
<div style="text-align:center;">
<svg width="260" height="100" viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#00F5FF"/>
      <stop offset="100%" stop-color="#7B61FF"/>
    </linearGradient>
  </defs>

  <circle cx="60" cy="60" r="26" fill="url(#grad)">
    <animate attributeName="r" values="22;30;22" dur="2s" repeatCount="indefinite"/>
  </circle>

  <text x="110" y="70"
        font-size="34"
        font-family="Segoe UI, Arial"
        fill="url(#grad)"
        font-weight="bold">
    AutoDFit
  </text>
</svg>
</div>
""", unsafe_allow_html=True)

time.sleep(0.8)
st.divider()

# =====================================================
# HERO SECTION
# =====================================================
st.markdown('<div class="hero">AI-Powered Dataset Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Automated preprocessing • Model selection • Optimization • Insights</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# =====================================================
# DATA UPLOAD
# =====================================================
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)
    st.success("Dataset Loaded Successfully")

    # =====================================================
    # DATASET OVERVIEW
    # =====================================================
    st.header("📊 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.dataframe(df.head())

    # =====================================================
    # TARGET SELECTION
    # =====================================================
    target = st.selectbox("Select Target Column", df.columns)
    X = df.drop(columns=[target])
    y = df[target]

    # =====================================================
    # PROBLEM DETECTION
    # =====================================================
    if y.dtype == "object" or y.nunique() < 20:
        problem = "classification"
    else:
        problem = "regression"

    st.success(f"Detected Problem Type: {problem}")

    # =====================================================
    # PIPELINE ANIMATION
    # =====================================================
    st.header("⚙ AI Processing Pipeline")

    progress = st.progress(0)
    status = st.empty()

    steps = [
        "Cleaning dataset...",
        "Handling missing values...",
        "Encoding categorical features...",
        "Scaling numerical features...",
        "Splitting data...",
        "Training models...",
        "Running genetic optimization...",
        "Evaluating performance...",
        "Preparing dashboard..."
    ]

    for i, step in enumerate(steps):
        status.info(step)
        progress.progress((i + 1) / len(steps))
        time.sleep(0.35)

    status.success("Processing Complete ✔")
    st.divider()

    # =====================================================
    # PREPROCESSING
    # =====================================================
    numeric_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", LabelEncoderWrapper())
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)

    # =====================================================
    # MODELS
    # =====================================================
    if problem == "classification":
        models = {
            "Logistic": LogisticRegression(max_iter=500),
            "RandomForest": RandomForestClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier()
        }
    else:
        models = {
            "Linear": LinearRegression(),
            "RandomForest": RandomForestRegressor(),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor()
        }

    # =====================================================
    # MODEL TRAINING
    # =====================================================
    st.header("🤖 Model Comparison")

    scores = {}
    trained_models = {}

    with st.spinner("Training models..."):
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            if problem == "classification":
                score = accuracy_score(y_test, pred)
            else:
                score = r2_score(y_test, pred)

            scores[name] = score
            trained_models[name] = model

    result_df = pd.DataFrame(scores.items(), columns=["Model","Score"]).sort_values(by="Score",ascending=False)
    st.dataframe(result_df)
    st.bar_chart(result_df.set_index("Model"))

    best_model_name = result_df.iloc[0]["Model"]
    best_score = result_df.iloc[0]["Score"]

    st.header("🏆 Best Model")
    colA, colB = st.columns(2)
    colA.metric("Model", best_model_name)
    colB.metric("Score", f"{best_score:.3f}")

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    st.header("📌 Feature Importance")

    if "RandomForest" in trained_models:
        rf = trained_models["RandomForest"]
        if hasattr(rf, "feature_importances_"):
            st.bar_chart(pd.Series(rf.feature_importances_))

    # =====================================================
    # GENETIC ALGORITHM FEATURE SELECTION
    # =====================================================
    st.header("🧬 Genetic Algorithm Optimization")

    n_features = X_train.shape[1]

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalGA(ind):
        sel = [i for i,v in enumerate(ind) if v==1]
        if not sel:
            return 0,
        X_sel = X_train[:,sel]
        model = RandomForestClassifier() if problem=="classification" else RandomForestRegressor()
        model.fit(X_sel, y_train)
        pred = model.predict(X_test[:,sel])
        score = accuracy_score(y_test,pred) if problem=="classification" else r2_score(y_test,pred)
        return score,

    toolbox.register("evaluate", evalGA)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=20)
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.2, ngen=5, verbose=False)

    best_ind = tools.selBest(pop,1)[0]
    st.success(f"Selected Features: {sum(best_ind)}")

    # =====================================================
    # PDF REPORT
    # =====================================================
    st.header("📄 Download Report")

    def make_pdf():
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf)
        style = getSampleStyleSheet()
        elems = []
        elems.append(Paragraph("AutoDFit Analysis Report", style["Title"]))
        elems.append(Spacer(1,12))
        elems.append(Paragraph(f"Problem Type: {problem}", style["Normal"]))
        elems.append(Paragraph(f"Best Model: {best_model_name}", style["Normal"]))
        elems.append(Paragraph(f"Score: {best_score:.3f}", style["Normal"]))
        doc.build(elems)
        buf.seek(0)
        return buf

    st.download_button("Download PDF", make_pdf(), "AutoDFit_Report.pdf")
