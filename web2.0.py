# =====================================================
#  ACL Injury Risk Predictor — Redesigned UI
#  SCI-journal / clinical-precision aesthetic
# =====================================================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

# 设置字体
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 0. 页面设置 =====
st.set_page_config(
    page_title="ACL Injury Risk Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== 全局 CSS — 深海军蓝 × 科研精密风格 =====
# 注意：这里直接紧跟引号，防止 React 渲染时产生多余的 <p> 标签
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main .block-container {
    background-color: #0A0F1E !important;
    color: #E8ECF4 !important;
    font-family: 'EB Garamond', Georgia, serif !important;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
}

[data-testid="stSidebar"] {
    background-color: #060B17 !important;
}

h1 {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    color: #FFFFFF !important;
    border-bottom: 2px solid #1E6FD9 !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 0.2rem !important;
}

h2, h3 {
    font-family: 'EB Garamond', Georgia, serif !important;
    color: #A8C4F0 !important;
    letter-spacing: 0.05em !important;
}

div.stNumberInput {
    margin-bottom: 0.55rem !important;
}

div.stNumberInput > label {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 0.92rem !important;
    color: #8FA8CC !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 2px !important;
}

div.stNumberInput input {
    background-color: #111827 !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 4px !important;
    color: #E8ECF4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.05rem !important;
    padding: 6px 10px !important;
    transition: border-color 0.2s ease;
}

div.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1E6FD9 0%, #0E4EA6 100%) !important;
    color: #FFFFFF !important;
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.65rem 1rem !important;
    margin-top: 0.8rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 18px rgba(30,111,217,0.35) !important;
}

hr {
    border-color: #1E3A5F !important;
    margin: 1.2rem 0 !important;
}

.badge-tag {
    display: inline-block;
    background: rgba(30,111,217,0.18);
    color: #5A9CF5;
    border: 1px solid rgba(30,111,217,0.4);
    border-radius: 3px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    margin-right: 6px;
    vertical-align: middle;
}

.risk-high {
    display: inline-block;
    background: rgba(220,50,50,0.15);
    color: #FF6B6B;
    border: 1px solid rgba(220,50,50,0.5);
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    padding: 4px 14px;
}

.risk-low {
    display: inline-block;
    background: rgba(30,180,100,0.15);
    color: #5BE3A0;
    border: 1px solid rgba(30,180,100,0.45);
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    padding: 4px 14px;
}

.acl-value-card {
    background: #060E1F;
    border: 1px solid #1E6FD9;
    border-left: 4px solid #1E6FD9;
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1rem;
}

.acl-value-card .label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #5A9CF5;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2px;
}

.acl-value-card .value {
    font-family: 'DM Mono', monospace;
    font-size: 2.2rem;
    color: #FFFFFF;
    font-weight: 500;
    line-height: 1.1;
}

.acl-value-card .unit {
    font-size: 1rem;
    color: #8FA8CC;
    margin-left: 4px;
}

.advice-item {
    display: flex;
    gap: 10px;
    margin-bottom: 0.7rem;
    align-items: flex-start;
}

.advice-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #1E6FD9;
    margin-top: 9px;
    flex-shrink: 0;
}

.advice-text {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 1.0rem;
    color: #C8D8EE;
    line-height: 1.7;
}

.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #5A9CF5;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1E3A5F;
}

.gauge-track {
    height: 8px;
    background: #1A2540;
    border-radius: 4px;
    position: relative;
    margin-top: 6px;
}

.gauge-fill-low {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #1E6FD9, #5BE3A0);
}

.gauge-fill-high {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #F5A623, #FF4B4B);
}

.gauge-marker {
    position: absolute;
    top: -4px;
    width: 2px;
    height: 16px;
    background: #FFFFFF;
    border-radius: 1px;
}

.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4A6A9A;
    margin-top: 4px;
}
</style>""", unsafe_allow_html=True)


# ===================== 1. 加载模型 =====================
@st.cache_resource
def load_model():
    return joblib.load('final_XGJ_model.pkl')

@st.cache_data
def load_test_data():
    return pd.read_csv('X_test.csv')

model = load_model()
X_test = load_test_data()

feature_names = ["HFA","HAA","KFA","ITR","KVA","ADF","FPA","TFA","HQ_ratio"]

# ===================== 2. 顶部标题区 =====================
st.markdown("""<div style="margin-bottom:0.2rem;">
    <span class="badge-tag">BIOMECHANICS</span>
    <span class="badge-tag">XGBoost MODEL</span>
    <span class="badge-tag">v1.0</span>
</div>""", unsafe_allow_html=True)

st.title("ACL Injury Risk Predictor")
st.markdown("""<div style="font-family:'EB Garamond',Georgia,serif; font-size:1.05rem;
   color:#6A8AB8; margin-top:-0.3rem; margin-bottom:1.4rem; letter-spacing:0.02em;">
   Machine-learning–based assessment of anterior cruciate ligament load from lower-extremity kinematics and muscle-strength ratios.
</div>""", unsafe_allow_html=True)

# ===================== 3. 主布局 =====================
left_col, spacer, right_col = st.columns([5, 0.3, 6])

# -------- 左侧输入区 --------
with left_col:
    st.markdown('<div class="section-title">Kinematic Input Parameters</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        HFA = st.number_input("Hip flexion angle (HFA) °", min_value=0.0, max_value=120.0, value=32.2, step=1.0)
        KFA = st.number_input("Knee flexion angle (KFA) °", min_value=0.0, max_value=120.0, value=11.0, step=1.0)
        KVA = st.number_input("Knee valgus angle (KVA) °", min_value=-15.0, max_value=30.0, value=11.29, step=1.0)
        FPA = st.number_input("Foot progression angle (FPA) °", min_value=-30.0, max_value=40.0, value=12.0, step=1.0)
        HAA = st.number_input("Hip abduction angle (HAA) °", min_value=-30.0, max_value=30.0, value=11.0, step=1.0)

    with col2:
        ITR = st.number_input("Internal tibial rotation (ITR) °", min_value=-30.0, max_value=30.0, value=6.0, step=1.0)
        ADF = st.number_input("Ankle dorsiflexion (ADF) °", min_value=-20.0, max_value=40.0, value=20.0, step=1.0)
        TFA = st.number_input("Trunk flexion angle (TFA) °", min_value=0.0, max_value=90.0, value=24.0, step=1.0)
        HQ_ratio = st.number_input("Hamstring / Quadriceps ratio", min_value=0.0, max_value=3.0, value=0.58, step=0.01, format="%.2f")

    predict_clicked = st.button("▶  Run Prediction", use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:1.2rem;">Feature Overview</div>', unsafe_allow_html=True)

    feature_values_raw = [HFA, HAA, KFA, ITR, KVA, ADF, FPA, TFA, HQ_ratio]
    short_labels = ["HFA","HAA","KFA","ITR","KVA","ADF","FPA","TFA","H/Q"]
    feat_max = [120, 30, 120, 30, 30, 40, 40, 90, 3.0]
    norm_vals = [v/m for v,m in zip(feature_values_raw, feat_max)]

    fig_bar, ax_bar = plt.subplots(figsize=(5, 2.3))
    fig_bar.patch.set_facecolor('#0D1526')
    ax_bar.set_facecolor('#0D1526')
    colors = ['#1E6FD9' if v < 0.6 else '#F5A623' if v < 0.85 else '#FF4B4B' for v in norm_vals]
    ax_bar.barh(short_labels, norm_vals, color=colors, edgecolor='none', height=0.58)
    ax_bar.axvline(0, color='#1E3A5F', linewidth=0.8)
    ax_bar.set_xlim(0, 1.05)
    ax_bar.tick_params(colors='#8FA8CC', labelsize=8)
    ax_bar.spines[:].set_color('#1E3A5F')
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['top'].set_visible(False)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)


# -------- 右侧结果区 --------
with right_col:
    st.markdown('<div class="section-title">Prediction & Clinical Explanation</div>', unsafe_allow_html=True)

    feature_values = [HFA, HAA, KFA, ITR, KVA, ADF, FPA, TFA, HQ_ratio]
    features = np.array([feature_values])
    HIGH_TH = 2.45
    MAX_LOAD = 4.8 

    if predict_clicked:
        acl_bw = float(np.asarray(model.predict(features)).ravel()[0])
        is_high = acl_bw >= HIGH_TH
        risk_label = "HIGH RISK" if is_high else "LOW RISK"
        gauge_pct = min(acl_bw / MAX_LOAD * 100, 100)
        fill_cls = "gauge-fill-high" if is_high else "gauge-fill-low"

        st.markdown(f"""<div class="acl-value-card">
            <div class="label">Predicted ACL Load</div>
            <div class="value">{acl_bw:.2f}<span class="unit">× BW</span></div>
        </div>""", unsafe_allow_html=True)

        risk_badge = f'<span class="risk-high">{risk_label}</span>' if is_high else f'<span class="risk-low">{risk_label}</span>'
        threshold_pct = HIGH_TH / MAX_LOAD * 100

        st.markdown(f"""<div style="margin-bottom:0.3rem;">
            <span style="font-family:'DM Mono',monospace;font-size:0.72rem; color:#4A6A9A;letter-spacing:.1em;text-transform:uppercase;">Risk Classification</span>&nbsp;&nbsp;{risk_badge}
        </div>
        <div class="gauge-wrap">
            <div class="gauge-track">
                <div class="{fill_cls}" style="width:{gauge_pct:.1f}%"></div>
                <div class="gauge-marker" style="left:{threshold_pct:.1f}%"></div>
            </div>
            <div class="gauge-labels">
                <span>0</span>
                <span>Threshold {HIGH_TH}×BW</span>
                <span>{MAX_LOAD}×BW</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        advice_items = [
            "Increase knee flexion at initial contact; avoid excessive tibial internal rotation or toe-in.",
            "Strengthen hamstrings and gluteal muscles; neuromuscular training.",
            "Monitor training load; consult a professional if pain occurs."
        ] if is_high else [
            "Maintain current technique; avoid abrupt footwork changes.",
            "Continue strength training; emphasize landing drills.",
            "Prioritize recovery and fatigue management."
        ]
        section_color = "#FF6B6B" if is_high else "#5BE3A0"

        st.markdown(f"""<div style="font-family:'DM Mono',monospace;font-size:0.72rem; color:{section_color};letter-spacing:.1em;text-transform:uppercase; margin-bottom:0.6rem;">Clinical Recommendations</div>""", unsafe_allow_html=True)

        for item in advice_items:
            st.markdown(f"""<div class="advice-item">
                <div class="advice-dot" style="background:{section_color};"></div>
                <div class="advice-text">{item}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # SHAP 分析
        st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:0.72rem; color:#5A9CF5;letter-spacing:.1em;text-transform:uppercase; margin-bottom:0.5rem;">SHAP Contribution Analysis</div>""", unsafe_allow_html=True)
        
        explainer_shap = shap.TreeExplainer(model)
        input_df = pd.DataFrame(features, columns=feature_names)
        shap_values = explainer_shap.shap_values(input_df)

        fig_shap, ax_shap = plt.subplots(figsize=(8, 2.6))
        fig_shap.patch.set_facecolor('#0D1526')
        shap.force_plot(explainer_shap.expected_value, shap_values[0, :], input_df.iloc[0, :], matplotlib=True, show=False, text_rotation=0)
        for ax in fig_shap.get_axes():
            ax.set_facecolor('#0D1526')
            for spine in ax.spines.values(): spine.set_color('#1E3A5F')
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

    else:
        # ✅ 这里是修复重点：去掉了多行字符串开头的换行
        st.markdown("""<div style="
            background:#0D1526;
            border:1px dashed #1E3A5F;
            border-radius:8px;
            padding:3rem 2rem;
            text-align:center;
            margin-top:1.5rem;
        ">
            <div style="font-family:'DM Mono',monospace;font-size:2rem; color:#1E3A5F;margin-bottom:1rem;">⬡</div>
            <div style="font-family:'EB Garamond',Georgia,serif; font-size:1.1rem;color:#3A5A8A;line-height:1.7;">
                Enter kinematic parameters on the left<br>
                and press <em>Run Prediction</em> to generate<br>
                the ACL load estimate and SHAP explanation.
            </div>
        </div>""", unsafe_allow_html=True)

# ===================== 底部说明 =====================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#2A4A6A; text-align:center;letter-spacing:0.06em;">
   FOR RESEARCH USE ONLY &nbsp;|&nbsp; XGBoost Regression Model &nbsp;|&nbsp; Threshold: 2.45 × BW
</div>""", unsafe_allow_html=True)