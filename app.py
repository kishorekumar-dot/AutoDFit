# ===============================================================
# AutoDFit — AI Dataset Intelligence Platform
# FIXED INDENTATION VERSION
# ===============================================================

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

from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, roc_curve

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    PageBreak, Table, TableStyle
)
# Fixed: Missing import for TableStyle was referenced in some logic
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from deap import base, creator, tools, algorithms

# ===============================================================
# PDF HEADER FOOTER
# ===============================================================
def add_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(40, 800, "AutoDFit AI Analytics Platform")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(550, 20, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()

# ===============================================================
# LABEL ENCODER WRAPPER
# ===============================================================
class LabelEncoderWrapper:
    def fit(self, X, y=None):
        self.encoders = {}
        for i in range(X.shape[1]):
            le = LabelEncoder()
            # We fit on the column data
            le.fit(X[:, i])
            self.encoders[i] = le
        return self

    def transform(self, X):
        X_copy = X.copy()
        for i in range(X_copy.shape[1]):
            X_copy[:, i] = self.encoders[i].transform(X_copy[:, i])
        return X_copy

# ===============================================================
# STREAMLIT UI
# ===============================================================
st.set_page_config(page_title="AutoDFit", layout="wide")
st.title("🚀 AutoDFit — AI Dataset Intelligence Platform")

file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    # ================= DATASET OVERVIEW =================
    st.header("Dataset Overview")
    rows, cols = df.shape
    missing = df.isnull().sum().sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Missing", missing)

    st.dataframe(df.head())

    total = df.size
    completeness = (1 - missing / total) * 100
    uniqueness = (df.nunique().sum() / total) * 100
    quality_score = completeness * 0.7 + uniqueness * 0.3
    st.metric("Dataset Health Score", f"{quality_score:.2f}%")

    # ================= CORRELATION =================
    st.subheader("Correlation Heatmap")
    fig_corr = plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    st.pyplot(fig_corr)

    # ================= TARGET =================
    target = st.selectbox("Select Target Column", df.columns)
    X = df.drop(columns=[target])
    y = df[target]

    problem = "classification" if y.dtype == "object" or y.nunique() < 20 else "regression"
    st.success(f"Detected: {problem}")

    # ================= PREPROCESS =================
    num = X.select_dtypes(include=np.number).columns
    cat = X.select_dtypes(exclude=np.number).columns

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", LabelEncoderWrapper())]), cat)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)

    # ================= MODELS =================
    if problem == "classification":
        models = {
            "Logistic": LogisticRegression(max_iter=300),
            "RandomForest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
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

    scores = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred) if problem == "classification" else r2_score(y_test, pred)
        scores[name] = score
        trained[name] = model

    result = pd.DataFrame(scores.items(), columns=["Model", "Score"]).sort_values("Score", ascending=False)
    st.subheader("Model Leaderboard")
    st.dataframe(result)

    best_model = result.iloc[0]["Model"]
    best_score = result.iloc[0]["Score"]
    st.metric("Best Score", f"{best_score * 100:.2f}%")

    # ================= VISUALS =================
    if problem == "classification":
        preds = trained[best_model].predict(X_test)
        fig_cm = plt.figure()
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d")
        st.pyplot(fig_cm)

    # ================= GENETIC ALGORITHM =================
    # ... (Keep your GA logic here, it was largely correct) ...
    # (Abbreviated for clarity, but ensure it's inside the 'if file:' block)

    # ================= DOWNLOAD MODEL =================
    st.download_button("Download Trained Model", pickle.dumps(trained[best_model]), "model.pkl")

    # ===============================================================
    # PROFESSIONAL PDF REPORT (INSIDE 'if file:')
    # ===============================================================
    def generate_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = []

        # COVER
        story.append(Spacer(1, 200))
        story.append(Paragraph("AutoDFit AI Intelligence Report", styles["Title"]))
        story.append(PageBreak())

        # EXECUTIVE
        story.append(Paragraph("Executive Summary", styles["Heading1"]))
        story.append(Paragraph(f"Best model {best_model} achieved {best_score*100:.2f}% performance. Dataset health score {quality_score:.2f}%.", styles["Normal"]))
        story.append(PageBreak())

        # DATASET INFO
        story.append(Paragraph("Dataset Overview", styles["Heading1"]))
        story.append(Paragraph(f"Rows: {rows}", styles["Normal"]))
        story.append(Paragraph(f"Columns: {cols}", styles["Normal"]))
        story.append(Paragraph(f"Missing values: {missing}", styles["Normal"]))
        story.append(PageBreak())

        # MODEL TABLE
        story.append(Paragraph("Model Comparison", styles["Heading1"]))
        table_data = [["Model", "Score"]]
        for m, s in scores.items():
            table_data.append([m, f"{s * 100:.2f}"])
        story.append(Table(table_data))
        story.append(PageBreak())

        # CORRELATION
        plt.figure(figsize=(5, 4))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
        plt.tight_layout()
        plt.savefig("heatmap.png")
        plt.close()
        story.append(Paragraph("Correlation Heatmap", styles["Heading1"]))
        story.append(Image("heatmap.png", width=400, height=300))
        story.append(PageBreak())

        # Finish Building PDF
        doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
        buffer.seek(0)

        # CLEANUP
        for f in ["heatmap.png", "cm.png", "feat.png", "shap.png"]:
            if os.path.exists(f):
                os.remove(f)
        
        return buffer # Critical: return inside function

    # DOWNLOAD BUTTON (INSIDE 'if file:' but OUTSIDE 'def generate_pdf')
    st.download_button(
        label="📄 Download Full Report",
        data=generate_pdf(),
        file_name="AutoDFit_Report.pdf",
        mime="application/pdf"
    )
