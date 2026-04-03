# =====================================================
#  ACL Injury Risk Predictor — Clinical Excellence UI
#  High-Impact Journal Style (White/Clean)
# =====================================================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体以符合科研出版物要求
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 0. 页面配置 =====
st.set_page_config(
    page_title="ACL Risk Assessment | Clinical Research",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== 1. 加载模型与数据 =====
@st.cache_resource
def load_assets():
    model = joblib.load('final_XGJ_model.pkl')
    # 模拟或加载解释器
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_assets()
feature_names = ["HFA","HAA","KFA","ITR","KVA","ADF","FPA","TFA","H/Q"]

# ===== 2. 精研 UI 样式 (CSS) =====
st.markdown("""
<style>
    /* 全局背景与字体 */
    .main {
        background-color: #F8F9FA !important;
        color: #2C3E50 !important;
    }
    
    /* 标题样式 - SCI 风格 */
    .sci-header {
        font-family: 'Times New Roman', serif;
        border-bottom: 2px solid #2E5077;
        padding-bottom: 10px;
        margin-bottom: 20px;
        color: #1A3A5F;
    }

    /* 容器卡片 */
    .report-card {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #E1E4E8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }

    /* 重点字体颜色 */
    .highlight-blue { color: #2E5077; font-weight: 700; }
    .highlight-red { color: #C0392B; font-weight: 700; }
    
    /* 输入框微调 */
    .stNumberInput label {
        font-size: 0.9rem !important;
        color: #566573 !important;
        font-weight: 600 !important;
    }

    /* 按钮美化 */
    div.stButton > button {
        background-color: #2E5077 !important;
        color: white !important;
        border-radius: 5px !important;
        width: 100%;
        height: 3em;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #1A3A5F !important;
        box-shadow: 0 4px 12px rgba(46, 80, 119, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ===== 3. 头部信息 =====
st.markdown("<h1 class='sci-header'>Anterior Cruciate Ligament (ACL) Risk Intelligence</h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='color: #7F8C8D; font-style: italic; font-size: 0.95rem;'>
        Predictive Analytics Framework based on High-Fidelity Biomechanical Simulation and Gradient Boosting Regressor.
    </p>
""", unsafe_allow_html=True)

# ===== 4. 主页面布局 =====
col_input, col_result = st.columns([1, 1.8], gap="large")

# -------- 左侧：参数录入 --------
with col_input:
    st.markdown("### 🧬 <span class='highlight-blue'>Clinical Parameters</span>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            hfa = st.number_input("Hip Flexion (HFA) °", 0.0, 120.0, 32.2)
            kfa = st.number_input("Knee Flexion (KFA) °", 0.0, 120.0, 11.0)
            kva = st.number_input("Knee Valgus (KVA) °", -15.0, 30.0, 11.2)
            fpa = st.number_input("Foot Progression (FPA) °", -30.0, 40.0, 12.0)
        with c2:
            haa = st.number_input("Hip Abduction (HAA) °", -30.0, 30.0, 11.0)
            itr = st.number_input("Internal Tibial Rot. (ITR) °", -30.0, 30.0, 6.0)
            adf = st.number_input("Ankle Dorsiflexion (ADF) °", -20.0, 40.0, 20.0)
            hq = st.number_input("H/Q Ratio", 0.0, 3.0, 0.58)
        
        tfa = st.number_input("Trunk Flexion (TFA) °", 0.0, 90.0, 24.0)
        
        run_sim = st.button("EXECUTE ANALYSIS")
        st.markdown("</div>", unsafe_allow_html=True)

# -------- 右侧：分析结果 --------
with col_result:
    if run_sim:
        # 数据准备
        input_data = np.array([[hfa, haa, kfa, itr, kva, adf, fpa, tfa, hq]])
        input_df = pd.DataFrame(input_data, columns=feature_names)
        
        # 预测
        prediction = float(model.predict(input_df)[0])
        threshold = 2.45
        is_high_risk = prediction >= threshold
        status_color = "#C0392B" if is_high_risk else "#27AE60"
        status_text = "HIGH RISK" if is_high_risk else "LOW RISK"

        # 1. 核心结果显示
        st.markdown(f"""
            <div class='report-card' style='border-left: 5px solid {status_color};'>
                <h4 style='margin-top:0;'>Diagnostic Summary</h4>
                <p style='margin-bottom:5px;'>Predicted ACL Load: <span style='font-size: 24px; color:{status_color}; font-weight:bold;'>{prediction:.2f} × BW</span></p>
                <p>Risk Classification: <span style='color:{status_color}; font-weight:bold;'>{status_text}</span> 
                <small>(Threshold: {threshold} × BW)</small></p>
            </div>
        """, unsafe_allow_html=True)

        # 2. 解释性分析 (SHAP)
        st.markdown("### 📊 <span class='highlight-blue'>Explainable AI (XAI) Analytics</span>", unsafe_allow_html=True)
        
        shap_values = explainer(input_df)

        tab1, tab2 = st.tabs(["Waterfall Explanation", "Force Visualization"])

        with tab1:
            # 瀑布图：SCI 风格
            fig_wf, ax_wf = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.gcf().set_size_inches(8, 4)
            st.pyplot(plt.gcf())
            plt.close()
            st.caption("Waterfall plot: Shows the contribution of each feature to the deviation from the mean prediction.")

        with tab2:
            # 力图：Streamlit 支持 HTML 渲染
            # 使用 matplotlib 版本的 force plot 确保在页面稳定显示
            shap.force_plot(
                explainer.expected_value, 
                shap_values.values[0], 
                input_df.iloc[0], 
                matplotlib=True, 
                show=False,
                text_rotation=0
            )
            plt.gcf().set_size_inches(10, 3)
            st.pyplot(plt.gcf())
            plt.close()
            st.caption("Force plot: Visualizes the 'push' of each feature on the final prediction score.")

    else:
        # 默认占位
        st.markdown("""
            <div style='text-align: center; padding: 100px 20px; color: #BDC3C7;'>
                <h2 style='font-weight: 300;'>Awaiting Input...</h2>
                <p>Configure kinematic parameters on the left to generate clinical report.</p>
            </div>
        """, unsafe_allow_html=True)

# ===== 5. 底部页脚 =====
st.markdown("---")
footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.markdown("<small>Institutional Use Only | Model: XGB-v1.4-Clinical</small>", unsafe_allow_html=True)
with footer_col2:
    st.markdown("<div style='text-align:right;'><small>© 2024 Biomechanics Research Lab</small></div>", unsafe_allow_html=True)帮我按照图的样式改代码