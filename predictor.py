# =====================================================
#  ACL Injury Risk Predictor — Custom Image Re-design
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

# ===== 0. 页面配置 =====
st.set_page_config(
    page_title="ACL Stress Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== 1. 加载模型 =====
@st.cache_resource
def load_assets():
    # 假设模型已经根据图中的特征训练好
    model = joblib.load('final_XGJ_model.pkl')
    explainer = shap.TreeExplainer(model)
    # 图片中的特征顺序: HFA, KVM, KFA, KFM, HAA, ASF, KVA, H/Q, AVA (共9个)
    # 注意：TFA, FPA, ADF 在图中被替换成了动力学参数如 KVM, KFM, ASF, AVA
    features = ["HFA", "KVM", "KFA", "KFM", "HAA", "ASF", "KVA", "H/Q", "AVA"]
    return model, explainer, features

model, explainer, feature_names = load_assets()

# ===== 2. 自定义 CSS (匹配图片样式) =====
st.markdown("""
<style>
    .main { background-color: #FFFFFF !important; }
    
    /* 图片红色大标题 */
    .img-title {
        color: #8B0000;
        font-family: 'Arial Black', sans-serif;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* 预测值标题 */
    .pred-header {
        color: #008000;
        font-family: 'Arial', sans-serif;
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 50px;
    }
    
    /* 预测值大数字 */
    .pred-value {
        color: #0000FF;
        font-family: 'Arial Black', sans-serif;
        font-size: 3.5rem;
        margin-top: -10px;
    }
    
    /* SHAP 标题 */
    .shap-title {
        color: #FF8C00;
        font-family: 'Arial', sans-serif;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .force-title {
        color: #800080;
        font-family: 'Arial', sans-serif;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 20px;
    }

    /* 输入框样式 */
    .stNumberInput div[data-baseweb="input"] {
        background-color: #F0F2F6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ===== 3. 头部标题 =====
st.markdown("<h1 class='img-title'>Predicting Peak ACL Stress in Cutting Movements</h1>", unsafe_allow_html=True)

# ===== 4. 主布局 =====
col_left, col_right = st.columns([1, 1.2], gap="large")

# -------- 左侧：输入与结果 --------
with col_left:
    c1, c2 = st.columns(2)
    with c1:
        hfa = st.number_input("Hip Flexion Angle(HFA)", value=21.20, step=0.1, format="%.2f")
        kfa = st.number_input("Knee Flexion Angle(KFA)", value=30.10, step=0.1, format="%.2f")
        haa = st.number_input("Hip Adduction Ankle(HAA)", value=21.32, step=0.1, format="%.2f")
        kva = st.number_input("Knee Valgus Ankle(KVA)", value=0.22, step=0.01, format="%.2f")
        ava = st.number_input("Ankle Valgus Ankle(AVA)", value=-10.20, step=0.1, format="%.2f")
    with c2:
        kvm = st.number_input("Knee Valgus Moment(KVM)", value=0.70, step=0.01, format="%.2f")
        kfm = st.number_input("Knee Flexion moment(KFM)", value=22.12, step=0.1, format="%.2f")
        asf = st.number_input("Anterior Tibial Shear Force (ASF)", value=2.06, step=0.01, format="%.2f")
        hq = st.number_input("Hamstring/Quadriceps(H/Q)", value=0.31, step=0.01, format="%.2f")

    # 计算与显示预测值
    input_data = np.array([[hfa, kvm, kfa, kfm, haa, asf, kva, hq, ava]])
    input_df = pd.DataFrame(input_data, columns=feature_names)
    prediction = float(model.predict(input_df)[0])

    st.markdown("<div class='pred-header'>Predicted Value</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pred-value'>{prediction:.3f}</div>", unsafe_allow_html=True)

# -------- 右侧：SHAP 图表 (瀑布图 + 力图) --------
with col_right:
    # 生成 SHAP 值
    shap_values = explainer(input_df)
    
    # 1. 瀑布图 (Waterfall Plot)
    st.markdown("<div class='shap-title'>Waterfall Plot</div>", unsafe_allow_html=True)
    fig_wf, ax_wf = plt.subplots(figsize=(8, 5))
    # 强制使用图片中的深红/蓝配色方案
    shap.plots.waterfall(shap_values[0], show=False)
    plt.gcf().set_size_inches(8, 5)
    st.pyplot(plt.gcf())
    plt.close()

    # 2. 力图 (Force Plot)
    st.markdown("<div class='force-title'>Force Plot</div>", unsafe_allow_html=True)
    # Matplotlib 版本的 Force plot
    shap.force_plot(
        explainer.expected_value, 
        shap_values.values[0], 
        input_df.iloc[0], 
        matplotlib=True, 
        show=False,
        plot_cmap=['#007bff', '#ff0051'] # 蓝色和红色的精准匹配
    )
    plt.gcf().set_size_inches(10, 3)
    # 调整布局以匹配图片
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

# ===== 5. 底部装饰 =====
st.markdown("<br><hr>", unsafe_allow_html=True)