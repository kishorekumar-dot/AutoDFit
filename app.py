import streamlit as st
import pandas as pd
import numpy as np
import time, io, random
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.graph_objects as go

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

# =====================================================
# LABEL ENCODER
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
# UI
# =====================================================
st.set_page_config(page_title="AutoDFit", layout="wide")

st.markdown("""
<div style="text-align:center;margin-top:20px">
<svg width="240" height="90">
<circle cx="50" cy="45" r="18" fill="#00F5FF">
<animate attributeName="r" values="14;22;14" dur="2s" repeatCount="indefinite"/>
</circle>
<text x="85" y="52" font-size="28" font-weight="600" fill="#7B61FF">AutoDFit</text>
</svg>
</div>
""", unsafe_allow_html=True)

st.title("AI Dataset Intelligence Platform")

# =====================================================
# UPLOAD
# =====================================================
file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)
    st.dataframe(df.head())

    # =====================================================
    # DATA QUALITY
    # =====================================================
    total = df.size
    missing = df.isnull().sum().sum()
    completeness = (1-missing/total)*100
    uniqueness = (df.nunique().sum()/total)*100
    quality_score = round(completeness*0.7+uniqueness*0.3,2)

    st.subheader("Dataset Quality")
    c1,c2,c3=st.columns(3)
    c1.metric("Completeness",f"{completeness:.2f}%")
    c2.metric("Uniqueness",f"{uniqueness:.2f}%")
    c3.metric("Quality Score",f"{quality_score:.2f}%")

    # =====================================================
    # TARGET
    # =====================================================
    target = st.selectbox("Select Target", df.columns)
    X=df.drop(columns=[target])
    y=df[target]

    problem="classification" if y.dtype=="object" or y.nunique()<20 else "regression"
    st.info(f"Detected Problem: {problem}")

    # =====================================================
    # PREPROCESS
    # =====================================================
    num=X.select_dtypes(include=np.number).columns
    cat=X.select_dtypes(exclude=np.number).columns

    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer()),("sc",StandardScaler())]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("enc",LabelEncoderWrapper())]),cat)
    ])

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    X_train=pre.fit_transform(X_train)
    X_test=pre.transform(X_test)

    # =====================================================
    # MODELS
    # =====================================================
    if problem=="classification":
        models={
            "Logistic":LogisticRegression(max_iter=500),
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

    for name,m in models.items():
        m.fit(X_train,y_train)
        pred=m.predict(X_test)
        score=accuracy_score(y_test,pred) if problem=="classification" else r2_score(y_test,pred)
        scores[name]=score
        trained[name]=m

    result=pd.DataFrame(scores.items(),columns=["Model","Score"]).sort_values("Score",ascending=False)
    st.dataframe(result)

    best_model=result.iloc[0]["Model"]
    best_score=result.iloc[0]["Score"]
    score_percent=best_score*100
    reliability=score_percent*0.92

    st.subheader("Best Model")
    a,b,c=st.columns(3)
    a.metric("Model",best_model)
    b.metric("Accuracy",f"{score_percent:.2f}%")
    c.metric("Reliability",f"{reliability:.2f}%")

    # =====================================================
    # GA FEATURE SELECTION
    # =====================================================
    n_features=X_train.shape[1]

    if not hasattr(creator,"FitnessMax"):
        creator.create("FitnessMax",base.Fitness,weights=(1.0,))
        creator.create("Individual",list,fitness=creator.FitnessMax)

    toolbox=base.Toolbox()
    toolbox.register("attr_bool",random.randint,0,1)
    toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_bool,n=n_features)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)

    def evalGA(ind):
        sel=[i for i,v in enumerate(ind) if v==1]
        if not sel: return 0,
        model=RandomForestClassifier() if problem=="classification" else RandomForestRegressor()
        model.fit(X_train[:,sel],y_train)
        pred=model.predict(X_test[:,sel])
        score=accuracy_score(y_test,pred) if problem=="classification" else r2_score(y_test,pred)
        return score,

    toolbox.register("evaluate",evalGA)
    toolbox.register("mate",tools.cxTwoPoint)
    toolbox.register("mutate",tools.mutFlipBit,indpb=0.05)
    toolbox.register("select",tools.selTournament,tournsize=3)

    pop=toolbox.population(n=20)
    algorithms.eaSimple(pop,toolbox,cxpb=0.6,mutpb=0.2,ngen=5,verbose=False)
    best_ind=tools.selBest(pop,1)[0]

    st.info(f"GA Selected Features: {sum(best_ind)}")

    # =====================================================
    # SHAP
    # =====================================================
    st.subheader("SHAP Explainability")
    try:
        explainer=shap.Explainer(trained[best_model],X_train)
        shap_values=explainer(X_test)
        fig=plt.figure()
        shap.summary_plot(shap_values,X_test,show=False)
        st.pyplot(fig)
    except:
        st.warning("SHAP not supported")

    # =====================================================
    # EXECUTIVE SUMMARY
    # =====================================================
    summary=f"""
Best model is {best_model} with accuracy {score_percent:.2f}%.
Dataset quality score {quality_score:.2f}%.
Model reliability {reliability:.2f}%.
"""
    st.info(summary)

    # =====================================================
    # PDF REPORT
    # =====================================================
    def generate_pdf():

        buffer=io.BytesIO()
        doc=SimpleDocTemplate(buffer)
        styles=getSampleStyleSheet()
        story=[]

        story.append(Paragraph("AutoDFit AI Report",styles["Title"]))
        story.append(Spacer(1,15))

        story.append(Paragraph("Executive Summary",styles["Heading2"]))
        story.append(Paragraph(summary,styles["Normal"]))

        story.append(Paragraph("Dataset Quality",styles["Heading2"]))
        story.append(Paragraph(f"Quality Score {quality_score:.2f}%",styles["Normal"]))

        # correlation
        plt.figure(figsize=(4,3))
        sns.heatmap(df.corr(numeric_only=True))
        plt.tight_layout()
        plt.savefig("corr.png")
        plt.close()
        story.append(RLImage("corr.png",width=350,height=250))

        # confusion matrix
        if problem=="classification":
            cm=confusion_matrix(y_test,trained[best_model].predict(X_test))
            plt.figure()
            sns.heatmap(cm,annot=True,fmt="d")
            plt.savefig("cm.png")
            plt.close()
            story.append(RLImage("cm.png",width=300,height=220))

        doc.build(story)
        buffer.seek(0)
        return buffer

    st.download_button("Download Full Report",generate_pdf(),"AutoDFit_Report.pdf")
