# ================= AUTO DFIT FULL INTELLIGENCE PLATFORM =================

import streamlit as st
import pandas as pd
import numpy as np
import io, time, random, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap
import os

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    PageBreak, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import PageTemplate, Frame

from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
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

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

from deap import base, creator, tools, algorithms


# =================================================
# PAGE
# =================================================
st.set_page_config(page_title="AutoDFit", layout="wide")
st.title("🚀 AutoDFit — AI Dataset Intelligence Platform")

# =================================================
# CACHE DATA
# =================================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# =================================================
# LABEL ENCODER
# =================================================
class LabelEncoderWrapper:
    def fit(self,X,y=None):
        self.enc={}
        for i in range(X.shape[1]):
            le=LabelEncoder()
            X[:,i]=le.fit_transform(X[:,i])
            self.enc[i]=le
        return self
    def transform(self,X):
        for i in range(X.shape[1]):
            X[:,i]=self.enc[i].transform(X[:,i])
        return X

# =================================================
# UPLOAD
# =================================================
file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    df = load_data(file)

    # ================= DATASET OVERVIEW =================
    st.header("📊 Dataset Overview")
    r,c = df.shape
    missing = df.isnull().sum().sum()
    st.write(f"Rows: {r} | Columns: {c} | Missing values: {missing}")
    st.dataframe(df.head())

    # ================= DATA QUALITY =================
    total=df.size
    completeness=(1-missing/total)*100
    uniqueness=(df.nunique().sum()/total)*100
    quality_score=(completeness*0.7+uniqueness*0.3)
    st.metric("Dataset Health Score", f"{quality_score:.2f}%")

    # ================= CORRELATION =================
    st.subheader("Correlation Heatmap")
    fig=plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    st.pyplot(fig)

    # ================= TARGET =================
    target=st.selectbox("Select target column", df.columns)
    X=df.drop(columns=[target])
    y=df[target]

    problem="classification" if y.dtype=="object" or y.nunique()<20 else "regression"
    st.success(f"Detected problem: {problem}")

    # ================= PREPROCESS =================
    num=X.select_dtypes(include=np.number).columns
    cat=X.select_dtypes(exclude=np.number).columns

    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer()),("sc",StandardScaler())]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                         ("enc",LabelEncoderWrapper())]),cat)
    ])

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    X_train=pre.fit_transform(X_train)
    X_test=pre.transform(X_test)

    # ================= MODELS =================
    if problem=="classification":
        models={
            "Logistic":LogisticRegression(max_iter=300),
            "RandomForest":RandomForestClassifier(),
            "SVM":SVC(probability=True),
            "KNN":KNeighborsClassifier(),
            "DecisionTree":DecisionTreeClassifier()
        }
    else:
        models={
            "Linear":LinearRegression(),
            "RandomForest":RandomForestRegressor(),
            "SVR":SVR(),
            "KNN":KNeighborsRegressor(),
            "DecisionTree":DecisionTreeRegressor()
        }

    scores={}
    trained={}

    for n,m in models.items():
        m.fit(X_train,y_train)
        pred=m.predict(X_test)
        s=accuracy_score(y_test,pred) if problem=="classification" else r2_score(y_test,pred)
        scores[n]=s
        trained[n]=m

    result=pd.DataFrame(scores.items(),columns=["Model","Score"]).sort_values("Score",ascending=False)
    st.subheader("🏆 Model Leaderboard")
    st.dataframe(result)

    best_model=result.iloc[0]["Model"]
    best_score=result.iloc[0]["Score"]
    st.metric("Best Accuracy", f"{best_score*100:.2f}%")

    # ================= ACCURACY GAUGE =================
    fig=go.Figure(go.Indicator(mode="gauge+number",
                               value=best_score*100,
                               number={'suffix':"%"},
                               title={'text':"Accuracy"},
                               gauge={'axis':{'range':[0,100]}}))
    st.plotly_chart(fig,use_container_width=True)

    # ================= CONFUSION / ROC =================
    if problem=="classification":
        pred=trained[best_model].predict(X_test)
        cm=confusion_matrix(y_test,pred)
        fig=plt.figure()
        sns.heatmap(cm,annot=True,fmt="d")
        st.pyplot(fig)

        try:
            probs=trained[best_model].predict_proba(X_test)[:,1]
            fpr,tpr,_=roc_curve(y_test,probs)
            fig=plt.figure()
            plt.plot(fpr,tpr)
            plt.plot([0,1],[0,1],'--')
            plt.title("ROC Curve")
            st.pyplot(fig)
        except:
            pass

    # ================= FEATURE IMPORTANCE =================
    if hasattr(trained[best_model],"feature_importances_"):
        st.subheader("Feature Importance")
        st.bar_chart(trained[best_model].feature_importances_)

    # ================= SHAP =================
  # =====================================================
# SHAP EXPLAINABILITY (STABLE VERSION)
# =====================================================
st.subheader("Model Explainability (SHAP)")

if st.checkbox("Show SHAP analysis"):

    try:
        model_for_shap = trained[best_model]

        # use small background sample
        background = X_train[:100]

        # create explainer
        explainer = shap.Explainer(model_for_shap, background)

        # explain small test sample
        sample = X_test[:50]

        shap_values = explainer(
            sample,
            check_additivity=False   # ⭐ CRITICAL FIX
        )

        st.write("Feature impact on predictions")

        fig = plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP explanation could not be generated for this model.")
        st.text(str(e))

    # ================= GENETIC ALGORITHM =================
    st.subheader("Genetic Feature Optimization")
    n=X_train.shape[1]
    if not hasattr(creator,"FitnessMax"):
        creator.create("FitnessMax",base.Fitness,weights=(1.0,))
        creator.create("Individual",list,fitness=creator.FitnessMax)

    toolbox=base.Toolbox()
    toolbox.register("attr_bool",random.randint,0,1)
    toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_bool,n=n)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)

    def evalGA(ind):
        sel=[i for i,v in enumerate(ind) if v==1]
        if not sel:return 0,
        m=RandomForestClassifier() if problem=="classification" else RandomForestRegressor()
        m.fit(X_train[:,sel],y_train)
        pred=m.predict(X_test[:,sel])
        return (accuracy_score(y_test,pred) if problem=="classification" else r2_score(y_test,pred),)

    toolbox.register("evaluate",evalGA)
    toolbox.register("mate",tools.cxTwoPoint)
    toolbox.register("mutate",tools.mutFlipBit,indpb=0.05)
    toolbox.register("select",tools.selTournament,tournsize=3)

    pop=toolbox.population(n=10)
    algorithms.eaSimple(pop,toolbox,0.6,0.2,5,verbose=False)
    best_ind=tools.selBest(pop,1)[0]
    st.write("Selected Features:", sum(best_ind))

    # ================= PREDICTION =================
    st.subheader("Prediction Interface")
    if st.button("Predict sample"):
        st.write(trained[best_model].predict(X_test[:1]))

    # ================= DOWNLOAD MODEL =================
    st.download_button("Download Trained Model",
                       pickle.dumps(trained[best_model]),
                       "model.pkl")

    # ================= EXECUTIVE SUMMARY =================
    st.info(f"""
Best model: {best_model}
Accuracy: {best_score*100:.2f}%
Dataset health: {quality_score:.2f}%
""")


def add_header_footer(canvas, doc):
    canvas.saveState()

    # Header Branding
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(40, 800, "AutoDFit AI Analytics Platform")

    # Footer Page Number
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(550, 20, text)

    canvas.restoreState()



    # ================= PDF =================

def generate_pdf():

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        rightMargin=40,
        leftMargin=40,
        topMargin=60,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

    # =====================================================
    # COVER PAGE
    # =====================================================
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("AutoDFit AI Intelligence Report", styles["Title"]))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Automated Machine Learning Analysis", styles["Heading2"]))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Generated Report", styles["Normal"]))
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Confidential – For Academic Use Only", styles["Normal"]))
    story.append(PageBreak())

    # =====================================================
    # TABLE OF CONTENTS
    # =====================================================
    story.append(Paragraph("Table of Contents", styles["Heading1"]))
    story.append(Spacer(1, 0.3*inch))

    toc_items = [
        "1. Executive Summary",
        "2. Dataset Overview",
        "3. Model Performance",
        "4. Visual Analytics",
        "5. Explainable AI (SHAP)",
        "6. Optimization Results",
        "7. Conclusion"
    ]

    for item in toc_items:
        story.append(Paragraph(item, styles["Normal"]))
        story.append(Spacer(1, 0.15*inch))

    story.append(PageBreak())

    # =====================================================
    # EXECUTIVE SUMMARY
    # =====================================================
    story.append(Paragraph("1. Executive Summary", styles["Heading1"]))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        f"The dataset demonstrates predictive performance of "
        f"{best_score*100:.2f}% using {best_model}. "
        f"Dataset health score: {quality_score:.2f}%.",
        styles["Normal"]
    ))
    story.append(PageBreak())

    # =====================================================
    # DATASET OVERVIEW
    # =====================================================
    story.append(Paragraph("2. Dataset Overview", styles["Heading1"]))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph(f"Rows: {r}", styles["Normal"]))
    story.append(Paragraph(f"Columns: {c}", styles["Normal"]))
    story.append(Paragraph(f"Missing Values: {missing}", styles["Normal"]))

    story.append(PageBreak())

    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================
    story.append(Paragraph("3. Model Performance", styles["Heading1"]))
    story.append(Spacer(1, 0.3*inch))

    table_data = [["Model", "Score (%)"]]
    for m, s in scores.items():
        table_data.append([m, f"{s*100:.2f}"])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.black),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ALIGN",(1,1),(-1,-1),"CENTER")
    ]))

    story.append(table)
    story.append(PageBreak())

    # =====================================================
    # VISUAL ANALYTICS (HEATMAP)
    # =====================================================
    story.append(Paragraph("4. Visual Analytics", styles["Heading1"]))
    story.append(Spacer(1, 0.3*inch))

    plt.figure(figsize=(5,4))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.tight_layout()
    plt.savefig("heatmap.png")
    plt.close()

    story.append(Image("heatmap.png", width=400, height=300))
    story.append(PageBreak())

    # =====================================================
    # SHAP
    # =====================================================
    story.append(Paragraph("5. Explainable AI (SHAP)", styles["Heading1"]))
    story.append(Spacer(1, 0.3*inch))

    try:
        background = X_train[:100]
        sample = X_test[:50]

        explainer = shap.Explainer(trained[best_model], background)
        shap_values = explainer(sample, check_additivity=False)

        plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig("shap.png")
        plt.close()

        story.append(Image("shap.png", width=400, height=250))

    except:
        story.append(Paragraph("SHAP explanation unavailable.", styles["Normal"]))

    story.append(PageBreak())

    # =====================================================
    # GA
    # =====================================================
    story.append(Paragraph("6. Optimization Results", styles["Heading1"]))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Selected features: {sum(best_ind)}", styles["Normal"]))
    story.append(PageBreak())

    # =====================================================
    # CONCLUSION
    # =====================================================
    story.append(Paragraph("7. Conclusion", styles["Heading1"]))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        "The AutoDFit platform provides automated machine learning "
        "evaluation, model comparison, optimization, and explainable AI insights.",
        styles["Normal"]
    ))

    # Build with header/footer
    doc.build(story, onFirstPage=add_header_footer,
              onLaterPages=add_header_footer)

    buffer.seek(0)
    return buffer
st.download_button(
    "Download Full Analysis Report",
    generate_pdf(),
    "AutoDFit_Full_Report.pdf"
)



