import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib

# ===== 0. 全局配置与 SCI 字体优化 =====
st.set_page_config(
    page_title="ACL Stress Analysis | Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 模拟 SCI 论文绘图风格 - 显式关闭网格
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.titlesize': 12,
    'axes.grid': False,  # 全局关闭网格
    'grid.alpha': 0      # 确保透明度为0
})

# ===== 1. 加载资源 =====
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('final_XGJ_model.pkl')
        explainer = shap.TreeExplainer(model)
        features = ["HFA", "HAA", "KFA", "ITR", "KVA", "ADF", "FPA", "TFA", "H/Q"]
        return model, explainer, features
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, None

model, explainer, feature_names = load_assets()

# ===== 2. SCI 风格 CSS =====
st.markdown("""
<style>
    .main { background-color: #FFFFFF !important; }
    .sci-title {
        color: #1A5276;
        font-family: 'Times New Roman', serif;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .sci-subtitle {
        color: #7F8C8D;
        font-family: 'Arial', sans-serif;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 30px;
    }
    .result-card {
        background-color: #F8F9FA;
        border: 1px solid #EAECEE;
        border-radius: 8px;
        padding: 20px;
        border-left: 6px solid #1A5276;
        margin-top: 15px;
    }
    .label-text { color: #2E4053; font-size: 0.8rem; text-transform: uppercase; font-weight: bold; }
    .value-text { color: #C0392B; font-family: 'Courier New', monospace; font-size: 3rem; font-weight: bold; margin: 5px 0; }
    .stNumberInput { margin-bottom: -10px; }
    /* 隐藏所有 matplotlib 生成图表的边框/背景网格细节 */
    iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)

# ===== 3. 标题 =====
st.markdown("<h1 class='sci-title'>ACL Injury Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<p class='sci-subtitle'>Precision Biomechanical Modeling for Clinical Decision Support</p>", unsafe_allow_html=True)

if model:
    # 使用 [1, 1] 比例让左右等宽，并使用垂直容器对齐
    col_left, col_right = st.columns([1, 1], gap="large")

    # -------- 左侧：参数输入 --------
    with col_left:
        st.markdown("### 📋 Input Parameters")
        
        # 使用多列布局紧凑化输入
        c1, c2 = st.columns(2)
        with c1:
            hfa = st.number_input("Hip Flexion (HFA)", value=21.20, format="%.2f")
            haa = st.number_input("Hip Adduction (HAA)", value=21.32, format="%.2f")
            kfa = st.number_input("Knee Flexion (KFA)", value=30.10, format="%.2f")
            kva = st.number_input("Knee Valgus (KVA)", value=0.22, format="%.2f")
        with c2:
            itr = st.number_input("Tibial Rotation (ITR)", value=6.00, format="%.2f")
            adf = st.number_input("Ankle Dorsifl. (ADF)", value=20.00, format="%.2f")
            fpa = st.number_input("Foot Prog. (FPA)", value=12.00, format="%.2f")
            tfa = st.number_input("Trunk Flexion (TFA)", value=24.00, format="%.2f")
        
        hq = st.slider("Hamstring/Quadriceps Ratio (H/Q)", 0.1, 1.5, 0.31)

        # 预测计算
        input_data = pd.DataFrame([[hfa, haa, kfa, itr, kva, adf, fpa, tfa, hq]], columns=feature_names)
        prediction = float(model.predict(input_data.values)[0])

        # 结果卡片
        st.markdown(f"""
            <div class="result-card">
                <div class="label-text">Predicted Stress Index</div>
                <div class="value-text">{prediction:.4f}</div>
                <div style="font-size: 0.85rem; color: #7F8C8D;">
                    Status: <span style="color:{'#C0392B' if prediction > 0.6 else '#27AE60'}; font-weight:bold;">
                    {'High Risk' if prediction > 0.6 else 'Normal Range'}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # 补充模型背景（用于平衡高度）
        with st.expander("ℹ️ Model Information"):
            st.caption("Algorithm: XGBoost Regressor")
            st.caption("Training Set: n=450 cases")
            st.caption("Validation RMSE: 0.042")
        
        csv = input_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Report", data=csv, file_name='acl_report.csv', mime='text/csv', use_container_width=True)

    # -------- 右侧：可视化 --------
    with col_right:
        st.markdown("### 🔍 Model Interpretation")
        
        shap_values = explainer(input_data)

        # 1. Waterfall Plot
        # 显式关闭网格并设置背景
        fig_wf, ax_wf = plt.subplots(figsize=(8, 4.5))
        shap.plots.waterfall(shap_values[0], show=False)
        ax_wf.grid(False) # 强制关闭网格
        plt.gca().xaxis.grid(False)
        plt.gca().yaxis.grid(False)
        st.pyplot(fig_wf, clear_figure=True)
        
        st.write("") # 间距

        # 2. Force Plot
        fig_fp = plt.figure(figsize=(10, 2.5))
        shap.force_plot(
            explainer.expected_value, 
            shap_values.values[0], 
            input_data.iloc[0], 
            matplotlib=True, 
            show=False,
            plot_cmap=['#1A5276', '#C0392B']
        )
        # Force plot matplotlib 版需要对当前轴进行清理
        plt.grid(False)
        ax = plt.gca()
        ax.grid(False)
        st.pyplot(plt.gcf(), clear_figure=True)
        
        st.caption("Fig 1. SHAP analysis quantifying the contribution of each kinematic variable.")

# ===== 5. 页脚 =====
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top: 1px solid #EEE; padding-top: 20px; color: #95A5A6; font-size: 0.8rem; font-family: 'Times New Roman', serif;">
    <strong>Reference:</strong> Zhang et al. (2026). Journal of Science and Medicine in Sport. DOI: 10.1016/j.jsams.2026.04.01
</div>
""", unsafe_allow_html=True)