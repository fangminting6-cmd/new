# 导入库
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 设置页面为宽屏，标题居中 & 红色（类似截图）
st.set_page_config(layout="wide", page_title="Predicting Peak ACL Stress")
st.markdown(
    "<h1 style='text-align:center; color:#b30000;'>Predicting Peak ACL Stress in Cutting Movements</h1>",
    unsafe_allow_html=True
)

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

# 🔴 修改成左窄右宽：右侧预测 & 图像区域更大
left_col, right_col = st.columns([1, 2])

# -------- 左侧：所有 st.number_input --------
with left_col:
    col1, col2 = st.columns(2)

    with col1:
        HFA = st.number_input(
            "Hip Flexion Angle (HFA, °):",
            min_value=0.0, max_value=120.0, value=21.2, step=0.1
        )
        KFA = st.number_input(
            "Knee Flexion Angle (KFA, °):",
            min_value=0.0, max_value=120.0, value=30.1, step=0.1
        )
        HAA = st.number_input(
            "Hip Adduction/Abduction Angle (HAA, °):",
            min_value=-30.0, max_value=30.0, value=21.3, step=0.1
        )
        KAA = st.number_input(
            "Knee Valgus Angle (KAA, °):",
            min_value=-15.0, max_value=30.0, value=0.22, step=0.1
        )

    with col2:
        ITR = st.number_input(
            "Internal Tibial Rotation Angle (ITR, °):",
            min_value=-30.0, max_value=30.0, value=-10.2, step=0.1
        )
        AFA = st.number_input(
            "Ankle Flexion Angle (AFA, °):",
            min_value=-20.0, max_value=40.0, value=22.1, step=0.1
        )
        FPA = st.number_input(
            "Foot Progression Angle (FPA, °):",
            min_value=-30.0, max_value=40.0, value=2.06, step=0.1
        )
        TFA = st.number_input(
            "Trunk Flexion Angle (TFA, °):",
            min_value=0.0, max_value=90.0, value=22.12, step=0.1
        )
        HQ_ratio = st.number_input(
            "Hamstring/Quadriceps (H/Q):",
            min_value=0.0, max_value=3.0, value=0.31, step=0.01
        )

# -------- 右侧：组装输入 + 预测 + 图像（结构按截图） --------
with right_col:
    # 组装成模型输入
    feature_values = [HFA, HAA, KFA, ITR, KAA, AFA, FPA, TFA, HQ_ratio]
    features = np.array([feature_values])  # shape = (1, 9)

    # 按钮放在右侧顶部
    if st.button("Predict", use_container_width=True):
        # ---------- 3.1 预测 ACL （假设输出单位为 ×BW） ----------
        acl_bw = float(np.asarray(model.predict(features)).ravel()[0])

        # ========== 上半部分：Predicted Value + 风险等级 ==========
        st.markdown("---")
        st.markdown(
            "<h3 style='text-align:center; color:#008000;'>Predicted Value</h3>",
            unsafe_allow_html=True
        )
        # 大号蓝色数字（类似截图 2.271）
        st.markdown(
            f"<h1 style='text-align:center; color:#0000ff;'>{acl_bw:.3f}</h1>",
            unsafe_allow_html=True
        )

        # 风险分级
        HIGH_TH = 2.45
        if acl_bw >= HIGH_TH:
            risk_label = "High risk"
            advice = (
                "- Increase knee flexion angle at initial contact (≥30°) to avoid dynamic knee valgus.\n"
                "- Reduce excessive tibial internal rotation / toe-in; keep the foot progression angle around 10–20°.\n"
                "- Strengthen the hamstrings and gluteal muscles, and improve H/Q co-activation and hip control.\n"
                "- Incorporate sport-specific technique and neuromuscular training, and monitor training/competition load.\n"
                "- If instability or pain is present, consult a sports medicine professional."
            )
            risk_color = "#ff0000"
        else:
            risk_label = "Low risk"
            advice = (
                "- The current ACL load is relatively low; you may continue with your current training program.\n"
                "- Maintain lower-limb strength and neuromuscular control, and pay attention to movement quality under fatigue.\n"
                "- Reassess regularly to monitor changes in risk."
            )
            risk_color = "#008000"

        st.markdown(
            f"<h4 style='text-align:center; color:{risk_color};'>Risk level: {risk_label}</h4>",
            unsafe_allow_html=True
        )
        st.markdown("**Recommendations:**\n" + advice)

        # ========== 下半部分：Force Plot（SHAP） ==========
        st.markdown("---")
        st.markdown(
            "<h3 style='text-align:center;'>Force Plot</h3>",
            unsafe_allow_html=True
        )

        # 4.1 创建解释器
        explainer_shap = shap.TreeExplainer(model)

        # 4.2 把输入变成 DataFrame，列名与特征对应
        input_df = pd.DataFrame(features, columns=feature_names)

        # 4.3 计算当前这个样本的 SHAP 值（回归：shape = (1, n_features)）
        shap_values = explainer_shap.shap_values(input_df)

        # 4.4 画 force plot（Matplotlib 版本，便于保存/嵌入）
        plt.figure(figsize=(10, 2.8))
        shap.force_plot(
            explainer_shap.expected_value,   # baseline
            shap_values[0, :],              # 当前样本的 SHAP 值
            input_df.iloc[0, :],            # 当前样本的特征
            matplotlib=True,
            show=False
        )

        st.pyplot(plt.gcf())
        plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=300)
        plt.close()
        # 如果想再下面展示保存的图片，也可以：
        # st.image("shap_force_plot.png", caption="Force Plot (SHAP)")
