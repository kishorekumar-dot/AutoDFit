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
    if st.checkbox("Show SHAP Explainability"):
        explainer=shap.Explainer(trained[best_model], X_train[:200])
        shap_values=explainer(X_test[:100])
        fig=plt.figure()
        shap.summary_plot(shap_values,X_test[:100],show=False)
        st.pyplot(fig)

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

    # ================= PDF =================

def generate_pdf():

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    # =====================================================
    # TITLE
    # =====================================================
    story.append(Paragraph("AutoDFit AI Analysis Report", styles["Title"]))
    story.append(Spacer(1,20))

    # =====================================================
    # DATASET SUMMARY
    # =====================================================
    story.append(Paragraph("Dataset Summary", styles["Heading2"]))
    story.append(Paragraph(f"Rows: {r}", styles["Normal"]))
    story.append(Paragraph(f"Columns: {c}", styles["Normal"]))
    story.append(Paragraph(f"Missing values: {missing}", styles["Normal"]))
    story.append(Paragraph(f"Dataset health score: {quality_score:.2f}%", styles["Normal"]))
    story.append(Spacer(1,20))

    # =====================================================
    # MODEL LEADERBOARD TABLE
    # =====================================================
    story.append(Paragraph("Model Performance Comparison", styles["Heading2"]))

    table_data = [["Model","Score (%)"]]
    for m,s in scores.items():
        table_data.append([m, f"{s*100:.2f}"])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),1,colors.black)
    ]))

    story.append(table)
    story.append(Spacer(1,20))

    # =====================================================
    # BEST MODEL
    # =====================================================
    story.append(Paragraph("Best Model", styles["Heading2"]))
    story.append(Paragraph(f"Model: {best_model}", styles["Normal"]))
    story.append(Paragraph(f"Accuracy: {best_score*100:.2f}%", styles["Normal"]))
    story.append(Spacer(1,20))

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================
    plt.figure(figsize=(5,4))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("corr.png")
    plt.close()

    story.append(Paragraph("Feature Correlation Heatmap", styles["Heading2"]))
    story.append(RLImage("corr.png", width=400, height=300))
    story.append(Spacer(1,20))

    # =====================================================
    # CONFUSION MATRIX + ROC (classification only)
    # =====================================================
    if problem == "classification":

        preds = trained[best_model].predict(X_test)

        cm = confusion_matrix(y_test, preds)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("cm.png")
        plt.close()

        story.append(Paragraph("Confusion Matrix", styles["Heading2"]))
        story.append(RLImage("cm.png", width=300, height=220))
        story.append(Spacer(1,20))

        try:
            probs = trained[best_model].predict_proba(X_test)[:,1]
            fpr,tpr,_ = roc_curve(y_test, probs)
            plt.figure()
            plt.plot(fpr,tpr,label="ROC")
            plt.plot([0,1],[0,1],'--')
            plt.title("ROC Curve")
            plt.tight_layout()
            plt.savefig("roc.png")
            plt.close()

            story.append(Paragraph("ROC Curve", styles["Heading2"]))
            story.append(RLImage("roc.png", width=300, height=220))
            story.append(Spacer(1,20))
        except:
            pass

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    if hasattr(trained[best_model], "feature_importances_"):
        plt.figure()
        plt.bar(range(len(trained[best_model].feature_importances_)),
                trained[best_model].feature_importances_)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feat.png")
        plt.close()

        story.append(Paragraph("Feature Importance", styles["Heading2"]))
        story.append(RLImage("feat.png", width=400, height=250))
        story.append(Spacer(1,20))

    # =====================================================
    # SHAP
    # =====================================================
    try:
        explainer = shap.Explainer(trained[best_model], X_train[:200])
        shap_values = explainer(X_test[:100])

        plt.figure()
        shap.summary_plot(shap_values, X_test[:100], show=False)
        plt.tight_layout()
        plt.savefig("shap.png")
        plt.close()

        story.append(Paragraph("Model Explainability (SHAP)", styles["Heading2"]))
        story.append(RLImage("shap.png", width=400, height=250))
        story.append(Spacer(1,20))
    except:
        pass

    # =====================================================
    # GENETIC ALGORITHM
    # =====================================================
    story.append(Paragraph("Genetic Feature Optimization", styles["Heading2"]))
    story.append(Paragraph(f"Selected features: {sum(best_ind)}", styles["Normal"]))
    story.append(Spacer(1,20))

    # =====================================================
    # EXECUTIVE SUMMARY
    # =====================================================
    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    story.append(Paragraph(
        f"The dataset demonstrates predictive performance of {best_score*100:.2f}% "
        f"using {best_model}. Dataset health score is {quality_score:.2f}%.",
        styles["Normal"]
    ))

    # =====================================================
    doc.build(story)
    buffer.seek(0)

    # cleanup temp images
    for f in ["corr.png","cm.png","roc.png","feat.png","shap.png"]:
        if os.path.exists(f):
            os.remove(f)

    return buffer

    st.download_button("Download Full Report", generate_pdf(), "report.pdf")

