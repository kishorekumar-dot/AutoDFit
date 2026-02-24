import streamlit as st
import pandas as pd
import numpy as np
import io

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

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from deap import base, creator, tools, algorithms
import random

# ------------------------------
# UI CONFIG
# ------------------------------
st.set_page_config(layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0F1117;
}
.stMetric {
    background-color:#1A1C23;
    padding:12px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AutoDFit — Intelligent AutoML Platform")

# ------------------------------
# Upload Dataset
# ------------------------------
file = st.file_uploader("Upload Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)
    st.success("Dataset Loaded")

    # ---------------- EDA ----------------
    st.header("📊 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head())

    # ---------------- Target ----------------
    target = st.selectbox("Select Target Column", df.columns)
    X = df.drop(columns=[target])
    y = df[target]

    # ---------------- Problem Detection ----------------
    if y.dtype == "object" or y.nunique() < 20:
        problem = "classification"
    else:
        problem = "regression"

    st.success(f"Detected Problem Type: {problem}")

    # ---------------- Preprocessing ----------------
    numeric_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", LabelEncoderWrapper())
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

    # ---------------- Models ----------------
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

    # ---------------- Model Training ----------------
    st.header("🤖 Model Comparison")

    scores = {}
    trained_models = {}

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
    best_model = trained_models[best_model_name]
    best_score = result_df.iloc[0]["Score"]

    st.success(f"Best Model: {best_model_name}")
    st.success(f"Best Score: {best_score:.3f}")

    # ---------------- Feature Importance ----------------
    st.header("📌 Feature Importance")

    if "RandomForest" in trained_models:
        rf = trained_models["RandomForest"]
        if hasattr(rf, "feature_importances_"):
            imp = pd.Series(rf.feature_importances_)
            st.bar_chart(imp)

    # ---------------- Genetic Algorithm ----------------
    st.header("🧬 Genetic Algorithm Feature Selection")

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
    st.write("Selected Features:", sum(best_ind))

    # ---------------- PDF REPORT ----------------
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


# ---------------- Label Encoder Wrapper ----------------
class LabelEncoderWrapper:
    def fit(self, X, y=None):
        self.enc = {}
        for i in range(X.shape[1]):
            le = LabelEncoder()
            X[:,i] = le.fit_transform(X[:,i])
            self.enc[i] = le
        return self

    def transform(self, X):
        for i in range(X.shape[1]):
            X[:,i] = self.enc[i].transform(X[:,i])
        return X
