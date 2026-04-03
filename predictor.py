# =====================================================
#  ACL Injury Risk Predictor — Precision Image Match
#  High-Impact Journal Style (Clean White)
# =====================================================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 0. 页面配置 =====
st.set_page_config(
    page_title="ACL Stress Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== 1. 加载模型与原始特征 =====
@st.cache_resource
def load_assets():
    # 加载模型
    model = joblib.load('final_XGJ_model.pkl')
    # 加载 SHAP 解释器
    explainer = shap.TreeExplainer(model)
    # 严格保持你原来的特征名称
    features = ["HFA", "HAA", "KFA", "ITR", "KVA", "ADF", "FPA", "TFA", "H/Q"]
    return model, explainer, features

model, explainer, feature_names = load_assets()

# ===== 2. 精研 UI 样式 (CSS) =====
st.markdown("""
<style>
    .main { background-color: #FFFFFF !important; }
    
    /* 匹配图片红色标题 */
    .img-title {
        color: #8B0000;
        font-family: 'Arial Black', sans-serif;
        font-size: 2.6rem;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    
    /* 预测值区域 */
    .pred-header {
        color: #008000;
        font-family: 'Arial', sans-serif;
        font-size: 1.6rem;
        font-weight: bold;
        margin-top: 30px;
        padding-top: 15px;
        border-top: 1px solid #EEEEEE;
    }
    .pred-value {
        color: #0000FF;
        font-family: 'Arial Black', sans-serif;
        font-size: 3.5rem;
        margin-top: -5px;
    }
    
    /* 图表标题 */
    .shap-title { color: #FF8C00; font-size: 1.4rem; font-weight: bold; margin-bottom: 5px; }
    .force-title { color: #800080; font-size: 1.4rem; font-weight: bold; margin-top: 20px; }

    /* 输入框灰色背景 */
    div[data-baseweb="input"] {
        background-color: #F0F2F6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== 3. 页面标题 =====
st.markdown("<h1 class='img-title'>Predicting Peak ACL Stress in Cutting Movements</h1>", unsafe_allow_html=True)

# ===== 4. 主布局 =====
col_left, col_right = st.columns([1, 1.2], gap="large")

# -------- 左侧：参数输入与预测结果 --------
with col_left:
    # 按照图片样式的双列输入组件
    c1, c2 = st.columns(2)
    with c1:
        hfa = st.number_input("Hip Flexion Angle(HFA)", value=21.20, format="%.2f")
        kfa = st.number_input("Knee Flexion Angle(KFA)", value=30.10, format="%.2f")
        haa = st.number_input("Hip Adduction Angle(HAA)", value=21.32, format="%.2f")
        kva = st.number_input("Knee Valgus Angle(KVA)", value=0.22, format="%.2f")
        adf = st.number_input("Ankle Dorsiflexion (ADF)", value=20.00, format="%.2f")
    with c2:
        itr = st.number_input("Internal Tibial Rot. (ITR)", value=6.00, format="%.2f")
        fpa = st.number_input("Foot Progression (FPA)", value=12.00, format="%.2f")
        tfa = st.number_input("Trunk Flexion (TFA)", value=24.00, format="%.2f")
        hq = st.number_input("Hamstring/Quadriceps(H/Q)", value=0.31, format="%.2f")

    # 构建输入 DataFrame 并确保特征顺序正确
    input_values = {
        "HFA": hfa, "HAA": haa, "KFA": kfa, "ITR": itr, 
        "KVA": kva, "ADF": adf, "FPA": fpa, "TFA": tfa, "H/Q": hq
    }
    input_df = pd.DataFrame([input_values])[feature_names]
    
    # 核心修复：使用 .values 数组进行预测，避开列名不匹配的报错
    prediction = float(model.predict(input_df.values)[0])

    # 显示结果
    st.markdown("<div class='pred-header'>Predicted Value</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pred-value'>{prediction:.3f}</div>", unsafe_allow_html=True)

# -------- 右侧：SHAP 分析图表 --------
with col_right:
    shap_values = explainer(input_df)
    
    # 1. Waterfall Plot
    st.markdown("<div class='shap-title'>Waterfall Plot</div>", unsafe_allow_html=True)
    fig_wf = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())
    plt.close()

    # 2. Force Plot
    st.markdown("<div class='force-title'>Force Plot</div>", unsafe_allow_html=True)
    shap.force_plot(
        explainer.expected_value, 
        shap_values.values[0], 
        input_df.iloc[0], 
        matplotlib=True, 
        show=False
    )
    plt.gcf().set_size_inches(10, 3)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

# ===== 5. 页脚 =====
st.markdown("<br><hr><center><small>Clinical Research Tool | v1.5 | 2026</small></center>", unsafe_allow_html=True)