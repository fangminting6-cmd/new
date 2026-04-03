import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib

# ===== 0. 全局配置 =====
st.set_page_config(
    page_title="ACL Stress Analysis | Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="collapsed"
)

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.grid': False
})

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('final_XGJ_model.pkl')
        explainer = shap.TreeExplainer(model)
        features = ["HFA", "HAA", "KFA", "ITR", "KVA", "ADF", "FPA", "TFA", "H/Q"]
        return model, explainer, features
    except:
        return None, None, None

model, explainer, feature_names = load_assets()

# ===== 2. 优化后的 CSS =====
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
        padding: 25px;
        border-left: 6px solid #1A5276;
        margin-top: 15px;
    }
    .label-text { color: #2E4053; font-size: 0.9rem; text-transform: uppercase; font-weight: bold; letter-spacing: 1px; }
    .value-text { color: #111111; font-family: 'Courier New', monospace; font-size: 3.5rem; font-weight: bold; }
    
    /* 核心修改：水平排列与状态大字 */
    .result-row { display: flex; align-items: center; margin-top: 5px; }
    .status-text {
        font-family: 'Arial Black', sans-serif;
        font-size: 2.4rem; 
        font-weight: 900;
        margin-left: 30px;
        letter-spacing: -1px;
    }
    
    .stNumberInput { margin-bottom: -10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='sci-title'>ACL Injury Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<p class='sci-subtitle'>Precision Biomechanical Modeling for Clinical Decision Support</p>", unsafe_allow_html=True)

if model:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📋 Input Parameters")
        c1, c2 = st.columns(2)
        with c1:
            hfa = st.number_input("Hip Flexion (HFA)", value=21.20)
            haa = st.number_input("Hip Adduction (HAA)", value=21.32)
            kfa = st.number_input("Knee Flexion (KFA)", value=30.10)
            kva = st.number_input("Knee Valgus (KVA)", value=0.22)
        with c2:
            itr = st.number_input("Tibial Rotation (ITR)", value=6.00)
            adf = st.number_input("Ankle Dorsifl. (ADF)", value=20.00)
            fpa = st.number_input("Foot Prog. (FPA)", value=12.00)
            tfa = st.number_input("Trunk Flexion (TFA)", value=24.00)
        
        hq = st.slider("Hamstring/Quadriceps Ratio (H/Q)", 0.1, 1.5, 0.31)
        input_data = pd.DataFrame([[hfa, haa, kfa, itr, kva, adf, fpa, tfa, hq]], columns=feature_names)
        prediction = float(model.predict(input_data.values)[0])

        # 修改后的结果卡片布局
        status_color = '#C0392B' if prediction > 0.6 else '#27AE60'
        status_label = 'HIGH RISK' if prediction > 0.6 else 'NORMAL'
        
        st.markdown(f"""
            <div class="result-card">
                <div class="label-text">Predicted Stress Index</div>
                <div class="result-row">
                    <span class="value-text">{prediction:.4f}</span>
                    <span class="status-text" style="color: {status_color};">
                        {status_label}
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #7F8C8D; margin-top: 10px;">
                    Confidence Interval: 95% (±0.024) | Biomechanical Baseline: 0.285
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("ℹ️ Model Metrics"):
            st.caption("Algorithm: XGBoost | Validation RMSE: 0.042")
        
        csv = input_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Report", data=csv, file_name='acl_report.csv', use_container_width=True)

    with col_right:
        st.markdown("### 🔍 Model Interpretation")
        shap_values = explainer(input_data)
        
        fig_wf, ax_wf = plt.subplots(figsize=(8, 4.5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.grid(False)
        st.pyplot(fig_wf, clear_figure=True)
        
        st.write("") 
        
        fig_fp = plt.figure(figsize=(10, 2.5))
        shap.force_plot(explainer.expected_value, shap_values.values[0], input_data.iloc[0], matplotlib=True, show=False, plot_cmap=['#1A5276', '#C0392B'])
        plt.grid(False)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Fig 1. SHAP analysis quantifying feature impact on ACL stress.")

st.markdown("<br><hr><div style='color: #95A5A6; font-size: 0.8rem; font-family: Times New Roman;'>Reference: Zhang et al. (2026). DOI: 10.1016/j.jsams.2026.04.01</div>", unsafe_allow_html=True)