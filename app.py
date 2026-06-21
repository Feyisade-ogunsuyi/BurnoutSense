import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BurnoutSense | Healthcare Burnout Risk System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 50%, #e94560 100%);
        padding: 30px 40px; border-radius: 16px; margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .header-banner h1 { color: white; font-size: 2.4em; font-weight: 800; margin: 0; }
    .header-banner p { color: #c8d6e5; font-size: 1.1em; margin: 8px 0 0 0; }
    .metric-card {
        background: white; border-radius: 16px; padding: 24px;
        text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-top: 4px solid #0f3460;
    }
    .metric-card h3 { color: #0f3460; font-size: 2em; font-weight: 800; margin: 0; }
    .metric-card p { color: #666; font-size: 0.9em; margin: 6px 0 0 0; }
    .risk-high {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        padding: 28px; border-radius: 16px; text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        padding: 28px; border-radius: 16px; text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        padding: 28px; border-radius: 16px; text-align: center;
    }
    .section-header {
        background: white; padding: 16px 24px; border-radius: 12px;
        border-left: 5px solid #0f3460; margin: 20px 0 16px 0;
    }
    .section-header h3 { color: #1a1a2e; margin: 0; font-size: 1.2em; font-weight: 700; }
    .rec-card-high {
        background: linear-gradient(135deg, #fff5f5, #ffe0e0);
        border-left: 5px solid #e74c3c; border-radius: 12px; padding: 20px 24px;
    }
    .rec-card-medium {
        background: linear-gradient(135deg, #fffbf0, #fff3cd);
        border-left: 5px solid #f39c12; border-radius: 12px; padding: 20px 24px;
    }
    .rec-card-low {
        background: linear-gradient(135deg, #f0fff4, #d4edda);
        border-left: 5px solid #27ae60; border-radius: 12px; padding: 20px 24px;
    }
    .welcome-card {
        background: white; border-radius: 16px; padding: 28px;
        text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .sidebar-section {
        background: rgba(255,255,255,0.1); border-radius: 10px;
        padding: 12px 16px; margin: 12px 0 8px 0;
        border-left: 3px solid #e94560;
    }
    .sidebar-section p {
        color: #e94560 !important; font-weight: 700 !important;
        font-size: 0.85em !important; text-transform: uppercase !important;
        letter-spacing: 1px !important; margin: 0 !important;
    }
    .info-box {
        background: linear-gradient(135deg, #e8f4fd, #d1ecf1);
        border-left: 5px solid #0f3460; border-radius: 12px;
        padding: 16px 20px; margin: 10px 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

DATA_URL = "https://raw.githubusercontent.com/Feyisade-ogunsuyi/BurnoutSense/refs/heads/main/mental%20health%20Dataset%20(1).csv"

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_URL)
    cols_to_drop = [c for c in ["OT2","WE","WPW","Study","Age"] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].replace({"": np.nan, " ": np.nan})
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "qDepressionc" in df.columns:
        df["qDepressionc"] = df["qDepressionc"].apply(
            lambda x: x if pd.isna(x) or 1 <= x <= 5 else np.nan)
    if "OT" in df.columns:
        df["OT"] = df["OT"].fillna(0)
    imputer = SimpleImputer(strategy="median")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    burnout_cols = [c for c in df.columns if "Burnout" in c]
    depression_cols = [c for c in df.columns if "Depression" in c]
    anxiety_cols = [c for c in df.columns if "Anexity" in c]
    stress_cols = ["q5SQb","q5SQc","q5SQd","q5SQe","q5SQf",
                   "q5SQg","q5SQh","q6SQ","q7SQ","q8SQ","q9SQ"]
    resilience_cols = [c for c in df.columns if "Resilence" in c]
    df["BurnoutScore"]         = df[burnout_cols].sum(axis=1)
    df["MentalHealthBurden"]   = (df[depression_cols].sum(axis=1) +
                                   df[anxiety_cols].sum(axis=1) +
                                   df[stress_cols].sum(axis=1))
    df["ResilienceProtection"] = df[resilience_cols].sum(axis=1)
    df["BurnoutRisk"]          = pd.qcut(df["BurnoutScore"], q=3,
                                          labels=[0,1,2]).astype(int)
    feature_cols = ["Age2","Sex","Married","Smoking","Disease","Educational",
                    "WE2","WPW2","Study2","Ward","Shift","ESh","ES","OT",
                    "MentalHealthBurden","ResilienceProtection"]
    X = df[feature_cols]
    y = df["BurnoutRisk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_s, y_train_s = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42)
    model.fit(X_train_s, y_train_s)
    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_cols

st.markdown("""
<div class="header-banner">
    <h1>🏥 BurnoutSense</h1>
    <p>Explainable Machine Learning System for Burnout Risk Prediction in Healthcare Workers</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("🔄 Loading BurnoutSense model... please wait"):
    rf_model, explainer, feature_cols = train_model()

# ── SIDEBAR ─────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding:10px 0 20px 0;">
    <h2 style="color:white; font-size:1.3em; font-weight:800;">📋 Worker Profile</h2>
    <p style="color:#c8d6e5; font-size:0.85em;">Fill in all sections and click predict</p>
</div>
""", unsafe_allow_html=True)

# Demographic
st.sidebar.markdown('<div class="sidebar-section"><p>👤 Demographic</p></div>',
                    unsafe_allow_html=True)
age2 = st.sidebar.selectbox("Age Group", options=[1,2,3,4,5],
    format_func=lambda x: {1:"Under 25",2:"25-34",3:"35-44",4:"45-54",5:"55+"}[x])
sex = st.sidebar.selectbox("Sex", options=[1,2],
    format_func=lambda x: {1:"Male",2:"Female"}[x])
married = st.sidebar.selectbox("Marital Status", options=[1,2,3],
    format_func=lambda x: {1:"Single",2:"Married",3:"Other"}[x])
smoking = st.sidebar.selectbox("Smoking Status", options=[1,2],
    format_func=lambda x: {1:"Non-Smoker",2:"Smoker"}[x])
disease = st.sidebar.selectbox("Chronic Disease", options=[1,2],
    format_func=lambda x: {1:"Yes",2:"No"}[x])
educational = st.sidebar.selectbox("Education Level", options=[1,2,3,4],
    format_func=lambda x: {1:"Diploma",2:"Bachelor",3:"Master",4:"PhD"}[x])

# Occupational
st.sidebar.markdown('<div class="sidebar-section"><p>💼 Occupational</p></div>',
                    unsafe_allow_html=True)
we2 = st.sidebar.selectbox("Work Experience", options=[1,2,3,4],
    format_func=lambda x: {1:"< 2 years",2:"2-5 years",3:"5-10 years",4:"> 10 years"}[x])
wpw2 = st.sidebar.selectbox("Weekly Working Hours", options=[1,2,3],
    format_func=lambda x: {1:"< 40 hrs",2:"40-50 hrs",3:"> 50 hrs"}[x])
study2 = st.sidebar.selectbox("Studying Alongside Work", options=[1,2],
    format_func=lambda x: {1:"Yes",2:"No"}[x])
ward = st.sidebar.selectbox("Ward Type", options=[1,2,3,4,5,6],
    format_func=lambda x: {1:"Medical",2:"Surgical",3:"ICU",
                            4:"Emergency",5:"Paediatric",6:"Other"}[x])
shift = st.sidebar.selectbox("Shift Pattern", options=[1,2,3,4,5,6],
    format_func=lambda x: {1:"Morning",2:"Afternoon",3:"Night",
                            4:"Rotating",5:"On-call",6:"Other"}[x])
esh = st.sidebar.selectbox("Employment Sector", options=[1,2,3],
    format_func=lambda x: {1:"Public",2:"Private",3:"Both"}[x])
es = st.sidebar.selectbox("Employment Status", options=[1,2,3,4,5],
    format_func=lambda x: {1:"Full-time",2:"Part-time",3:"Contract",
                            4:"Temporary",5:"Other"}[x])
ot = st.sidebar.number_input("Overtime Hours / Month",
    min_value=0, max_value=100, value=0, step=1)

# Psychological Questionnaire
st.sidebar.markdown('<div class="sidebar-section"><p>🧠 Psychological Assessment</p></div>',
                    unsafe_allow_html=True)
st.sidebar.markdown("""
<p style="color:#c8d6e5 !important; font-size:0.8em !important;
   margin:4px 0 10px 0 !important;">
Rate each item: 1=Never, 2=Rarely, 3=Sometimes, 4=Often, 5=Always
</p>""", unsafe_allow_html=True)

st.sidebar.markdown("""
<p style="color:#e94560 !important; font-size:0.8em !important;
   font-weight:700 !important; margin:8px 0 4px 0 !important;">
DEPRESSION
</p>""", unsafe_allow_html=True)
d1 = st.sidebar.selectbox("I feel hopeless about the future",
    options=[1,2,3,4,5], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often",5:"Always"}[x], key="d1")
d2 = st.sidebar.selectbox("I feel worthless or inadequate",
    options=[1,2,3,4,5], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often",5:"Always"}[x], key="d2")
d3 = st.sidebar.selectbox("I have difficulty concentrating",
    options=[1,2,3,4,5], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often",5:"Always"}[x], key="d3")

st.sidebar.markdown("""
<p style="color:#e94560 !important; font-size:0.8em !important;
   font-weight:700 !important; margin:8px 0 4px 0 !important;">
ANXIETY
</p>""", unsafe_allow_html=True)
a1 = st.sidebar.selectbox("I feel nervous or anxious at work",
    options=[1,2,3,4,5], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often",5:"Always"}[x], key="a1")
a2 = st.sidebar.selectbox("I experience physical tension or restlessness",
    options=[1,2,3,4,5], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often",5:"Always"}[x], key="a2")
a3 = st.sidebar.selectbox("I feel overwhelmed by my responsibilities",
    options=[1,2,3,4,5], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often",5:"Always"}[x], key="a3")

st.sidebar.markdown("""
<p style="color:#e94560 !important; font-size:0.8em !important;
   font-weight:700 !important; margin:8px 0 4px 0 !important;">
STRESS
</p>""", unsafe_allow_html=True)
s1 = st.sidebar.selectbox("I feel unable to cope with my workload",
    options=[1,2,3,4], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often"}[x], key="s1")
s2 = st.sidebar.selectbox("I feel emotionally drained after work",
    options=[1,2,3,4], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often"}[x], key="s2")
s3 = st.sidebar.selectbox("I feel my work demands are excessive",
    options=[1,2,3,4], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often"}[x], key="s3")

st.sidebar.markdown("""
<p style="color:#e94560 !important; font-size:0.8em !important;
   font-weight:700 !important; margin:8px 0 4px 0 !important;">
RESILIENCE
</p>""", unsafe_allow_html=True)
r1 = st.sidebar.selectbox("I recover well from setbacks at work",
    options=[1,2,3,4], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often"}[x], key="r1")
r2 = st.sidebar.selectbox("I feel confident handling difficult situations",
    options=[1,2,3,4], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often"}[x], key="r2")
r3 = st.sidebar.selectbox("I maintain focus well under pressure",
    options=[1,2,3,4], format_func=lambda x:
    {1:"Never",2:"Rarely",3:"Sometimes",4:"Often"}[x], key="r3")

# Calculate composite scores
# Scale depression/anxiety to match original dataset ranges
# Original: DepressionScore range 6-38 (16 items), AnxietyScore 18-71 (18 items)
# Stress range 11-31 (11 items), Resilience 25-110 (25 items)
# We scale our 3-item scores proportionally

dep_raw    = d1 + d2 + d3   # range 3-15
anx_raw    = a1 + a2 + a3   # range 3-15
stress_raw = s1 + s2 + s3   # range 3-12
res_raw    = r1 + r2 + r3   # range 3-12

# Scale to original dataset ranges
dep_scaled    = 6  + (dep_raw - 3)    / 12 * 32  # scaled to 6-38
anx_scaled    = 18 + (anx_raw - 3)   / 12 * 53  # scaled to 18-71
stress_scaled = 11 + (stress_raw - 3) / 9  * 20  # scaled to 11-31
res_scaled    = 25 + (res_raw - 3)   / 9  * 85  # low answers = low resilience

# Scale MentalHealthBurden to training data range (39-107)
mental_health_burden_raw = dep_scaled + anx_scaled + stress_scaled
mental_health_burden  = float(np.clip(39 + (mental_health_burden_raw - 39) / 101 * 68, 39, 107))
resilience_protection = float(np.clip(res_scaled, 25, 110))
# Display values (clipped to valid ranges)
display_burden = mental_health_burden
display_resilience = resilience_protection

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.sidebar.button("🔍  Predict Burnout Risk",
                                 use_container_width=True, type="primary")

# ── MAIN CONTENT ─────────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame({
        "Age2":[age2],"Sex":[sex],"Married":[married],
        "Smoking":[smoking],"Disease":[disease],"Educational":[educational],
        "WE2":[we2],"WPW2":[wpw2],"Study2":[study2],"Ward":[ward],
        "Shift":[shift],"ESh":[esh],"ES":[es],"OT":[ot],
        "MentalHealthBurden":[mental_health_burden],
        "ResilienceProtection":[resilience_protection]
    })

    prediction  = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0]

    risk_labels  = {0:"LOW RISK",  1:"MEDIUM RISK", 2:"HIGH RISK"}
    risk_icons   = {0:"✅",         1:"⚠️",           2:"🔴"}
    risk_classes = {0:"risk-low",  1:"risk-medium",  2:"risk-high"}
    risk_desc    = {
        0:"This worker shows low indicators of burnout. Continue routine monitoring.",
        1:"This worker shows moderate burnout indicators. Closer monitoring is advised.",
        2:"This worker shows high burnout indicators. Immediate support is recommended."
    }

    # Risk banner
    st.markdown(f"""
    <div class="{risk_classes[prediction]}">
        <h2 style="color:white; font-size:2em; font-weight:800; margin:0;">
            {risk_icons[prediction]} {risk_labels[prediction]}
        </h2>
        <p style="color:rgba(255,255,255,0.9); font-size:1em; margin:10px 0 0 0;">
            {risk_desc[prediction]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show calculated scores
    st.markdown(f"""
    <div class="info-box">
        <p style="color:#0f3460; font-weight:700; margin:0 0 6px 0;">
            📊 Calculated Psychological Scores
        </p>
        <p style="color:#333; margin:0; font-size:0.9em;">
            Mental Health Burden: <b>{display_burden:.1f}</b> (range 39–107) &nbsp;|&nbsp;
            Resilience Protection: <b>{display_resilience:.1f}</b> (range 25–110)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h3>{risk_icons[prediction]}</h3>
            <p><b>{risk_labels[prediction]}</b><br>Predicted Risk Level</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <h3>{probability[prediction]*100:.1f}%</h3>
            <p>Prediction Confidence</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <h3>16</h3>
            <p>Features Analysed</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <h3>91%</h3>
            <p>Model Accuracy</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability breakdown
    st.markdown("""<div class="section-header">
        <h3>📊 Risk Probability Breakdown</h3>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:white; border-radius:16px; padding:24px;
                box-shadow:0 4px 20px rgba(0,0,0,0.08); margin:16px 0;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span>✅ Low Risk</span><span><b>{probability[0]*100:.1f}%</b></span>
        </div>
        <div style="background:#f0f4f8; border-radius:50px; height:14px; margin-bottom:16px;">
            <div style="background:linear-gradient(90deg,#27ae60,#2ecc71);
                height:14px; border-radius:50px; width:{probability[0]*100:.1f}%"></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span>⚠️ Medium Risk</span><span><b>{probability[1]*100:.1f}%</b></span>
        </div>
        <div style="background:#f0f4f8; border-radius:50px; height:14px; margin-bottom:16px;">
            <div style="background:linear-gradient(90deg,#f39c12,#e67e22);
                height:14px; border-radius:50px; width:{probability[1]*100:.1f}%"></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span>🔴 High Risk</span><span><b>{probability[2]*100:.1f}%</b></span>
        </div>
        <div style="background:#f0f4f8; border-radius:50px; height:14px; margin-bottom:16px;">
            <div style="background:linear-gradient(90deg,#e74c3c,#c0392b);
                height:14px; border-radius:50px; width:{probability[2]*100:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SHAP
    st.markdown("""<div class="section-header">
        <h3>🔍 Explainable AI — Why This Prediction?</h3>
    </div>""", unsafe_allow_html=True)

    shap_vals  = explainer.shap_values(input_data)
    shap_array = np.array(shap_vals)
    sv = shap_array[0, :, prediction]

    # Top factors table
    st.markdown("""<div class="section-header">
        <h3>📋 Top Factors Influencing This Prediction</h3>
    </div>""", unsafe_allow_html=True)

    factors_df = pd.DataFrame({
        "Feature": feature_cols,
        "Worker Value": input_data.iloc[0].values,
        "SHAP Impact": sv
    })
    factors_df["Direction"] = factors_df["SHAP Impact"].apply(
        lambda x: "⬆️ Increases Risk" if x > 0 else "⬇️ Decreases Risk")
    factors_df["Abs Impact"] = factors_df["SHAP Impact"].abs()
    factors_df = factors_df.sort_values("Abs Impact", ascending=False).head(10)
    factors_df = factors_df[["Feature","Worker Value","Direction","SHAP Impact"]]
    factors_df["SHAP Impact"] = factors_df["SHAP Impact"].round(4)
    st.dataframe(factors_df, use_container_width=True, hide_index=True)

    # Recommendations
    st.markdown("""<div class="section-header">
        <h3>💡 Clinical Recommendations</h3>
    </div>""", unsafe_allow_html=True)

    if prediction == 2:
        st.markdown("""<div class="rec-card-high">
            <h4 style="color:#c0392b;">🔴 High Risk — Immediate Action Required</h4>
            <ul style="color:#333; margin:8px 0 0 0;">
                <li>Arrange an urgent wellbeing support session with occupational health</li>
                <li>Consider immediate workload reduction or reallocation</li>
                <li>Review and adjust shift patterns and overtime hours</li>
                <li>Provide access to mental health and counselling resources</li>
                <li>Conduct a one-to-one meeting with line manager within 48 hours</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    elif prediction == 1:
        st.markdown("""<div class="rec-card-medium">
            <h4 style="color:#e67e22;">⚠️ Medium Risk — Monitoring & Support Advised</h4>
            <ul style="color:#333; margin:8px 0 0 0;">
                <li>Schedule regular wellbeing check-ins (bi-weekly)</li>
                <li>Monitor workload and shift patterns closely</li>
                <li>Encourage participation in resilience-building programmes</li>
                <li>Review work-life balance and study commitments</li>
                <li>Consider peer support or mentoring arrangements</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="rec-card-low">
            <h4 style="color:#27ae60;">✅ Low Risk — Maintain & Protect</h4>
            <ul style="color:#333; margin:8px 0 0 0;">
                <li>Continue routine wellbeing monitoring (monthly)</li>
                <li>Maintain current support and working structures</li>
                <li>Encourage continued resilience and wellbeing practices</li>
                <li>Recognise and reward positive work contributions</li>
                <li>Re-assess if workload or personal circumstances change</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f8fafc; border-radius:10px; padding:16px 20px;
                border:1px solid #e0e0e0; margin-top:10px;">
        <p style="color:#888; font-size:0.8em; margin:0; text-align:center;">
            ⚠️ <b>Disclaimer:</b> This system is a decision-support tool only.
            Predictions should not replace clinical judgement or professional assessment.
            Psychological scores are derived from a simplified 12-item screening tool;
            full psychometric assessment is recommended for clinical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:20px 0 30px 0;">
        <h2 style="color:#1a1a2e; font-size:1.8em; font-weight:700;">
            Welcome to BurnoutSense
        </h2>
        <p style="color:#666; font-size:1.05em; max-width:700px; margin:0 auto;">
            An explainable machine learning system for transparent burnout risk
            monitoring in healthcare organisations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("🤖","Random Forest","91% accuracy on 831 healthcare worker records"),
        ("🔍","SHAP Explainability","Every prediction explained transparently"),
        ("🎯","3 Risk Levels","Low, Medium, and High burnout risk"),
        ("💡","Recommendations","Actionable clinical guidance"),
    ]
    for col, (icon, title, desc) in zip([col1,col2,col3,col4], cards):
        with col:
            st.markdown(f"""<div class="welcome-card">
                <div style="font-size:2.5em; margin-bottom:12px;">{icon}</div>
                <h4 style="color:#1a1a2e; font-weight:700; margin:0 0 8px 0;">{title}</h4>
                <p style="color:#666; font-size:0.9em; margin:0;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style="background:white; border-radius:16px; padding:28px;
                    box-shadow:0 4px 20px rgba(0,0,0,0.08);">
            <h3 style="color:#1a1a2e; font-weight:700;">📌 How to Use</h3>
            <ol style="color:#555; line-height:2;">
                <li>Enter worker demographic details in the sidebar</li>
                <li>Fill in occupational information</li>
                <li>Answer the 12 psychological screening questions</li>
                <li>Click <b>Predict Burnout Risk</b></li>
                <li>View the risk level, scores, and SHAP explanation</li>
                <li>Follow the clinical recommendations</li>
            </ol>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style="background:white; border-radius:16px; padding:28px;
                    box-shadow:0 4px 20px rgba(0,0,0,0.08);">
            <h3 style="color:#1a1a2e; font-weight:700;">📊 Model Performance</h3>
            <table style="width:100%; border-collapse:collapse; color:#555;">
                <tr style="border-bottom:2px solid #f0f4f8;">
                    <th style="text-align:left; padding:8px; color:#1a1a2e;">Model</th>
                    <th style="text-align:center; padding:8px;">Accuracy</th>
                    <th style="text-align:center; padding:8px;">F1 Score</th>
                </tr>
                <tr style="background:#f8fafc;">
                    <td style="padding:8px;"><b>Random Forest ⭐</b></td>
                    <td style="text-align:center; padding:8px;">90.4%</td>
                    <td style="text-align:center; padding:8px;">0.904</td>
                </tr>
                <tr>
                    <td style="padding:8px;">Logistic Regression</td>
                    <td style="text-align:center; padding:8px;">91.0%</td>
                    <td style="text-align:center; padding:8px;">0.912</td>
                </tr>
                <tr>
                    <td style="padding:8px;">XGBoost</td>
                    <td style="text-align:center; padding:8px;">88.0%</td>
                    <td style="text-align:center; padding:8px;">0.880</td>
                </tr>
            </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#0f3460);
                border-radius:16px; padding:24px; text-align:center; margin-top:20px;">
        <p style="color:#c8d6e5; margin:0; font-size:0.9em;">
            🏥 <b style="color:white;">BurnoutSense</b> —
            MRes Computing Research Project | University of Greater Manchester | 2026
        </p>
    </div>
    """, unsafe_allow_html=True)
