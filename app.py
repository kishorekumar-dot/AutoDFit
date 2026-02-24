import streamlit as st
import pandas as pd
import numpy as np
import time, io, random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import accuracy_score, r2_score

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import shap

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="AutoDFit", layout="wide")
st.title("🚀 AutoDFit — AI Dataset Intelligence")

# =====================================================
# LOAD DATA (CACHED → FAST)
# =====================================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# =====================================================
# ENCODER
# =====================================================
class LabelEncoderWrapper:
    def fit(self, X, y=None):
        self.encoders={}
        for i in range(X.shape[1]):
            le=LabelEncoder()
            X[:,i]=le.fit_transform(X[:,i])
            self.encoders[i]=le
        return self

    def transform(self,X):
        for i in range(X.shape[1]):
            X[:,i]=self.encoders[i].transform(X[:,i])
        return X

# =====================================================
# FILE UPLOAD
# =====================================================
file = st.file_uploader("Upload CSV dataset", type=["csv"])

if file:

    df = load_data(file)

    # =====================================================
    # DATASET OVERVIEW (INSTANT DISPLAY)
    # =====================================================
    st.header("📊 Dataset Overview")

    rows, cols = df.shape
    missing = df.isnull().sum().sum()

    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Missing Values", missing)

    st.dataframe(df.head())

    # =====================================================
    # TARGET
    # =====================================================
    target = st.selectbox("Select target column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    problem = "classification" if y.dtype=="object" or y.nunique()<20 else "regression"
    st.success(f"Detected problem type: {problem}")

    # =====================================================
    # PREPROCESS
    # =====================================================
    num = X.select_dtypes(include=np.number).columns
    cat = X.select_dtypes(exclude=np.number).columns

    pre = ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer()),("sc",StandardScaler())]), num),
        ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("enc",LabelEncoderWrapper())]), cat)
    ])

    # =====================================================
    # TRAIN SPLIT
    # =====================================================
    with st.spinner("Preparing data..."):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        X_train = pre.fit_transform(X_train)
        X_test = pre.transform(X_test)

    # =====================================================
    # MODELS (FAST SET)
    # =====================================================
    if problem=="classification":
        models={
            "Logistic": LogisticRegression(max_iter=200),
            "RandomForest": RandomForestClassifier(n_estimators=80),
            "DecisionTree": DecisionTreeClassifier()
        }
    else:
        models={
            "Linear": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=80),
            "DecisionTree": DecisionTreeRegressor()
        }

    # =====================================================
    # TRAIN MODELS (CACHED)
    # =====================================================
    @st.cache_resource
    def train_models(X_train,X_test,y_train,y_test):
        scores={}
        trained={}
        for name,m in models.items():
            m.fit(X_train,y_train)
            pred=m.predict(X_test)
            score=accuracy_score(y_test,pred) if problem=="classification" else r2_score(y_test,pred)
            scores[name]=score
            trained[name]=m
        return scores, trained

    with st.spinner("Training models..."):
        scores, trained = train_models(X_train,X_test,y_train,y_test)

    result = pd.DataFrame(scores.items(),columns=["Model","Score"]).sort_values("Score",ascending=False)
    st.subheader("🏆 Model Leaderboard")
    st.dataframe(result)

    best_model = result.iloc[0]["Model"]
    best_score = result.iloc[0]["Score"]

    st.metric("Best Model Accuracy", f"{best_score*100:.2f}%")

    # =====================================================
    # ACCURACY GAUGE
    # =====================================================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=best_score*100,
        number={'suffix':"%"},
        title={'text':"Prediction Accuracy"},
        gauge={'axis':{'range':[0,100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # SHAP (SAMPLED → FAST)
    # =====================================================
    if st.checkbox("Show model explainability (SHAP)"):
        try:
            sample_size = min(100, X_test.shape[0])
            sample = X_test[:sample_size]

            explainer = shap.Explainer(trained[best_model], X_train[:200])
            shap_values = explainer(sample)

            st.subheader("Feature Impact")
            fig = plt.figure()
            shap.summary_plot(shap_values, sample, show=False)
            st.pyplot(fig)
        except:
            st.warning("SHAP not supported for this model.")

    # =====================================================
    # PDF REPORT (FAST SIMPLE VERSION)
    # =====================================================
    def generate_pdf():
        buffer=io.BytesIO()
        doc=SimpleDocTemplate(buffer)
        styles=getSampleStyleSheet()
        story=[]
        story.append(Paragraph("AutoDFit AI Report",styles["Title"]))
        story.append(Paragraph(f"Rows: {rows}",styles["Normal"]))
        story.append(Paragraph(f"Columns: {cols}",styles["Normal"]))
        story.append(Paragraph(f"Missing: {missing}",styles["Normal"]))
        story.append(Paragraph(f"Best Model: {best_model}",styles["Normal"]))
        story.append(Paragraph(f"Score: {best_score*100:.2f}%",styles["Normal"]))
        doc.build(story)
        buffer.seek(0)
        return buffer

    st.download_button("📄 Download Report", generate_pdf(), "AutoDFit_Report.pdf")
