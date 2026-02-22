import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import io
import time
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from deap import base, creator, tools, algorithms

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="AutoDFit", layout="wide")

# -------------------------------------------------
# Minimal Dark Theme
# -------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0F1117;
}
h1, h2, h3 {
    font-weight: 600;
}
.stMetric {
    background-color: #1A1C23;
    padding: 12px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Splash Screen
# -------------------------------------------------
if "loaded" not in st.session_state:
    splash = st.empty()
    splash.markdown("""
    <div style='text-align:center;margin-top:200px'>
        <h1>AutoDFit</h1>
        <h3>Initializing Intelligent Dataset Analyzer...</h3>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2)
    splash.empty()
    st.session_state.loaded = True

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("AutoDFit — Intelligent Dataset Fitness Analyzer")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Model",
    ["Random Forest", "Logistic Regression", "SVM"]
)

monte_runs = st.sidebar.slider("Monte Carlo Runs", 20, 100, 50)

# -------------------------------------------------
# Main App
# -------------------------------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)

    if st.button("Analyze Dataset"):

        # -----------------------------------------
        # Clean Animated Processing
        # -----------------------------------------
        status = st.empty()
        progress = st.progress(0)

        steps = [
            "Loading Dataset",
            "Data Profiling",
            "Training Baseline Model",
            "Running Genetic Algorithm",
            "Monte Carlo Simulation",
            "Computing Fitness Score",
            "Generating Report"
        ]

        for i, step in enumerate(steps):
            status.write(f"Processing: {step}...")
            progress.progress((i + 1) / len(steps))
            time.sleep(0.25)

        # -----------------------------------------
        # Data Profiling
        # -----------------------------------------
        missing_percentage = df.isnull().mean().mean() * 100
        rows, cols = df.shape

        df = df.dropna()
        X = df.drop(columns=[target_column])
        y = df[target_column]

        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # -----------------------------------------
        # Model Selection
        # -----------------------------------------
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=50)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=500)
        else:
            model = SVC()

        # -----------------------------------------
        # Baseline Model
        # -----------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        baseline_acc = accuracy_score(y_test, model.predict(X_test))

        # -----------------------------------------
        # GA Feature Selection
        # -----------------------------------------
        num_features = X.shape[1]

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, n=num_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval_individual(ind):
            selected = [i for i in range(len(ind)) if ind[i] == 1]
            if len(selected) == 0:
                return 0,
            X_sel = X.iloc[:, selected]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sel, y, test_size=0.2, random_state=42
            )
            model.fit(X_tr, y_tr)
            return accuracy_score(y_te, model.predict(X_te)),

        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=25)
        algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.2, ngen=8, verbose=False)

        best = tools.selBest(pop, 1)[0]
        selected_features = [i for i in range(len(best)) if best[i] == 1]
        X_best = X.iloc[:, selected_features] if selected_features else X

        X_train, X_test, y_train, y_test = train_test_split(
            X_best, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        optimized_acc = accuracy_score(y_test, model.predict(X_test))

        # -----------------------------------------
        # Monte Carlo Simulation
        # -----------------------------------------
        scores = []
        for _ in range(monte_runs):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_best, y, test_size=0.2,
                random_state=random.randint(1, 10000)
            )
            model.fit(X_tr, y_tr)
            scores.append(accuracy_score(y_te, model.predict(X_te)))

        mean_acc = np.mean(scores)
        std_acc = np.std(scores)

        # -----------------------------------------
        # Fitness Score
        # -----------------------------------------
        completeness = 1 - (missing_percentage / 100)
        stability = 1 - std_acc

        fitness = (
            0.3 * completeness +
            0.3 * mean_acc +
            0.2 * optimized_acc +
            0.2 * stability
        )
        fitness = max(0, min(fitness, 1))

        # -----------------------------------------
        # Display Results
        # -----------------------------------------
        st.divider()
        st.subheader("Dataset Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", rows)
        c2.metric("Features", cols - 1)
        c3.metric("Missing %", f"{missing_percentage:.2f}")

        st.divider()
        st.subheader("Model Performance")

        c4, c5 = st.columns(2)
        c4.metric("Baseline Accuracy", f"{baseline_acc*100:.2f}%")
        c5.metric("Optimized Accuracy", f"{optimized_acc*100:.2f}%")

        st.divider()
        st.subheader("Monte Carlo Stability")

        st.metric("Mean Accuracy", f"{mean_acc*100:.2f}%")
        st.metric("Std Deviation", f"{std_acc*100:.2f}%")

        fig, ax = plt.subplots()
        ax.hist(scores, bins=12)
        ax.set_title("Accuracy Distribution")
        st.pyplot(fig)

        st.divider()
        st.subheader("Dataset Fitness Score")
        st.progress(fitness)
        st.write(f"{fitness*100:.2f}/100")

        # -----------------------------------------
        # PDF Report
        # -----------------------------------------
        def generate_pdf():
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("AutoDFit Dataset Report", styles["Title"]))
            elements.append(Spacer(1, 0.2 * inch))

            text = f"""
            Rows: {rows}<br/>
            Features: {cols - 1}<br/>
            Missing %: {missing_percentage:.2f}<br/>
            Baseline Accuracy: {baseline_acc*100:.2f}%<br/>
            Optimized Accuracy: {optimized_acc*100:.2f}%<br/>
            Mean Accuracy: {mean_acc*100:.2f}%<br/>
            Std Dev: {std_acc*100:.2f}%<br/>
            Dataset Fitness Score: {fitness*100:.2f}/100
            """
            elements.append(Paragraph(text, styles["Normal"]))
            doc.build(elements)
            buffer.seek(0)
            return buffer

        st.download_button(
            "Download PDF Report",
            generate_pdf(),
            file_name="AutoDFit_Report.pdf",
            mime="application/pdf"
        )

        status.success("Analysis Completed Successfully")