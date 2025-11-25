# 导入库
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 0. 页面设置：宽屏 =====
st.set_page_config(page_title="ACL Injury Risk Predictor", layout="wide")

# 调整所有 number_input 之间的上下间距
st.markdown("""
    <style>
        /* 整个页面使用 Times New Roman 字体 */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdown"] {
            font-family: "Times New Roman", "Times New Roman", serif;
        }

        /* 控制每个 st.number_input 外层容器的下边距 */
        div.stNumberInput {
            margin-bottom: 1.0rem;   /* 数值可以再调大一点比如 1.5rem */
        }
    </style>
""", unsafe_allow_html=True)

# ===================== 1. 加载模型 =====================
model = joblib.load('final_XGJ_model.pkl')  # 确保路径无误

# 如果暂时不用，可以先注释掉
X_test = pd.read_csv('X_test.csv')

# 特征名称（要与训练时一致）
feature_names = [
    "HFA",       # 髋屈曲
    "HAA",       # 髋外展 / 内收
    "KFA",       # 膝屈曲
    "ITR",       # 胫骨内旋
    "KAA",       # 膝外翻 / 内翻
    "AFA",       # 踝屈曲
    "FPA",       # 足外展
    "TFA",       # 躯干前倾
    "HQ_ratio",  # 腘绳肌/股四头肌
]

# ===================== 2. 布局：左输入 / 右预测 =====================
st.title("ACL Injury Risk Predictor")

# 左右同宽
left_col, right_col = st.columns(2)

# -------- 左侧：所有 st.number_input + Predict 按钮 --------
with left_col:
    col1, col2 = st.columns(2)

    with col1:
        HFA = st.number_input(
            "Hip flexion angle (HFA):",
            min_value=0.0, max_value=120.0, value=32.2, step=1.0
        )
        KFA = st.number_input(
            "Knee flexion angle (KFA):",
            min_value=0.0, max_value=120.0, value=11.0, step=1.0
        )
        KAA = st.number_input(
            "Knee valgus angle (KVA):",
            min_value=-15.0, max_value=30.0, value=11.29, step=1.0
        )
        FPA = st.number_input(
            "Foot progression angle (FPA):",
            min_value=-30.0, max_value=40.0, value=12.0, step=1.0
        )
        HAA = st.number_input(
            "Hip abduction angle (HAA):",
            min_value=-30.0, max_value=30.0, value=11.0, step=1.0
        )

    with col2:
        ITR = st.number_input(
            "Internal tibial rotation angle (ITR):",
            min_value=-30.0, max_value=30.0, value=6.0, step=1.0
        )
        AFA = st.number_input(
            "Ankle flexion angle (AFA):",
            min_value=-20.0, max_value=40.0, value=20.0, step=1.0
        )
        TFA = st.number_input(
            "Trunk flexion angle (TFA):",
            min_value=0.0, max_value=90.0, value=24.0, step=1.0
        )
        HQ_ratio = st.number_input(
            "H/Q:",
            min_value=0.0, max_value=3.0, value=0.58, step=0.01
        )

    # ⭐ 把按钮放在 H/Q 下面
    predict_clicked = st.button("Predict", use_container_width=True)

# -------- 右侧：组装输入 + 显示预测 + SHAP --------
with right_col:
    # 往上挪一点：加一个带负 margin-top 的占位 div
    st.markdown(
        "<div style='margin-top:-8rem;'></div>",
        unsafe_allow_html=True
    )

    st.subheader("Prediction & Explanation")

    # 组装成模型输入
    feature_values = [HFA, HAA, KFA, ITR, KAA, AFA, FPA, TFA, HQ_ratio]
    features = np.array([feature_values])  # shape = (1, 9)

    # ===================== 3. 点击按钮进行预测 =====================
    if predict_clicked:
        # ---------- 3.1 预测 ACL （假设输出单位为 ×BW） ----------
        acl_bw = float(np.asarray(model.predict(features)).ravel()[0])
        st.write(f"**Predicted ACL load (×BW):** {acl_bw:.2f}")

        # ---------- 3.2 风险分级 ----------
        HIGH_TH = 2.45

        # 建议：≥ 阈值为 High risk
        if acl_bw >= HIGH_TH:
            risk_label = "High risk"
            advice = (
                "- Increase knee flexion at initial contact and avoid excessive tibial internal rotation or toe-in, keeping the foot generally aligned with the direction of movement.\n"
                "- Strengthen the hamstrings and gluteal muscles, and use neuromuscular and sport-specific technique training to improve dynamic knee control.\n"
                "- Monitor training and competition load, and consult a sports medicine professional if knee pain or instability occurs."
)

        else:
            risk_label = "Low risk"
           advice = (
    "- Maintain current lunge technique with smooth deceleration, good trunk and hip control and avoid abrupt changes in footwork without technical supervision.\n"
    "- Continue lower-limb strength and neuromuscular training (hamstrings, gluteals, quadriceps and core), including single-leg balance and landing drills with proper alignment.\n"
    "- Monitor overall training and competition load, prioritize recovery and fatigue management, and seek early assessment if any new knee pain, swelling or instability appears."
)



        st.markdown(f"**Risk level:** {risk_label}")
        st.markdown("**Recommendations:**\n" + advice)

        # ===================== 4. SHAP 单样本解释 =====================
        st.markdown("### SHAP Force Plot")

        # 4.1 创建解释器
        explainer_shap = shap.TreeExplainer(model)

        # 4.2 把输入变成 DataFrame，列名与特征对应
        input_df = pd.DataFrame(features, columns=feature_names)

        # 4.3 计算当前这个样本的 SHAP 值（回归：shape = (1, n_features)）
        shap_values = explainer_shap.shap_values(input_df)

        # 4.4 画 force plot（Matplotlib 版本，便于保存/嵌入）
        plt.figure(figsize=(8, 2.5))
        shap.force_plot(
            explainer_shap.expected_value,  # baseline
            shap_values[0, :],              # 当前样本的 SHAP 值
            input_df.iloc[0, :],            # 当前样本的特征
            matplotlib=True,
            show=False
        )

        st.pyplot(plt.gcf())
        plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=300)
        plt.close()
