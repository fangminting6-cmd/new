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
        # 加载50个重复划分模型用于预测区间（如有）
        ensemble_models = []
        import os
        for seed in range(50):
            path = f'XGB_seed{seed}.pkl'
            if os.path.exists(path):
                ensemble_models.append(joblib.load(path))
        return model, explainer, features, ensemble_models
    except:
        return None, None, None, []

model, explainer, feature_names, ensemble_models = load_assets()

# ===== 输入范围（来自描述性统计表的 Min/Max）=====
# ⚠️ 请将下方数值替换为你描述性统计表中的实际 Min 和 Max
FEATURE_RANGES = {
    "HFA":  (3.98,  52.99),   # (Min, Max) 请替换
    "HAA":  (-20.89,  27.95),
    "KFA":  (4.70,  46.77),
    "ITR":  (-19.84, 17.41),
    "KVA":  (-16.26, 18.38),
    "ADF":  (-22.85,  36.25),
    "FPA":  (-4.14, 22.26),
    "TFA":  (9.46,  45.09),
    "H/Q":  (0.35,  0.98),
}

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

def check_input_ranges(input_df):
    """检查输入是否超出训练数据范围，返回超出范围的特征列表"""
    warnings = []
    for feature, (lo, hi) in FEATURE_RANGES.items():
        val = float(input_df[feature].iloc[0])
        if val < lo or val > hi:
            warnings.append(f"**{feature}**: input {val:.2f} (valid range: {lo:.2f} – {hi:.2f})")
    return warnings

def get_ensemble_prediction(input_arr):
    """使用50个模型集成计算预测区间"""
    if len(ensemble_models) >= 2:
        preds = [float(m.predict(input_arr)[0]) for m in ensemble_models]
        return np.mean(preds), np.percentile(preds, 2.5), np.percentile(preds, 97.5)
    else:
        return None, None, None

st.markdown("<h1 class='sci-title'>Predicting ACL Loading in wide Lunge Movements</h1>", unsafe_allow_html=True)
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

        # ── 输入范围检查 ──────────────────────────────────────────
        range_warnings = check_input_ranges(input_data)
        if range_warnings:
            st.warning(
                "⚠️ **Out-of-range input detected.** The following features exceed the training data distribution. "
                "Predictions may be unreliable:\n\n" + "\n\n".join(f"- {w}" for w in range_warnings)
            )

        # ── 预测区间（50模型集成）────────────────────────────────
        ens_mean, ci_lo, ci_hi = get_ensemble_prediction(input_data.values)

        # ── 结果卡片 ─────────────────────────────────────────────
        status_color = '#C0392B' if prediction > 0.6 else '#27AE60'
        status_label = 'HIGH LOAD' if prediction > 0.6 else 'NORMAL'

        if ens_mean is not None:
            ci_str = f"95% Prediction Interval: [{ci_lo:.4f}, {ci_hi:.4f}] &nbsp;|&nbsp; Ensemble mean: {ens_mean:.4f}"
        else:
            ci_str = "95% Prediction Interval: not available (ensemble models not loaded) &nbsp;|&nbsp; Biomechanical Baseline: 0.285"

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
                    {ci_str}
                </div>
            </div>
        """, unsafe_allow_html=True)

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
