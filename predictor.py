# predictor.py

# ========= 0. 导入库 =========
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ========= 1. Streamlit 标题 =========
st.title("ACL Injury Risk Predictor")

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

# ========= 2. 定义一个“安全加载模型”的函数（懒加载） =========
@st.cache_resource
def load_model_safe():
    """
    懒加载 final_XGJ_model.pkl。
    如果缺少某个依赖库，抛出异常，在外面统一捕获并显示。
    注意：这里不直接 st.error/st.stop，只负责“抛错”。
    """
    return joblib.load("final_XGJ_model.pkl")


# ========= 3. UI：输入框 =========
HFA = st.number_input("髋屈曲 (°):",   min_value=0.0,  max_value=120.0, value=43.0, step=1.0)
HAA = st.number_input("髋外展 (°):",   min_value=-30.0, max_value=30.0,  value=3.0,  step=1.0)
KFA = st.number_input("膝屈曲 (°):",   min_value=0.0,  max_value=120.0, value=29.0, step=1.0)
ITR = st.number_input("胫骨内旋 (°):", min_value=-30.0, max_value=30.0,  value=8.0,  step=1.0)
KAA = st.number_input("膝外翻 (°):",   min_value=-15.0, max_value=30.0,  value=10.0, step=1.0)
AFA = st.number_input("踝屈曲 (°):",   min_value=-20.0, max_value=40.0,  value=21.0, step=1.0)
FPA = st.number_input("足外展 (°):",   min_value=-30.0, max_value=40.0,  value=13.0, step=1.0)
TFA = st.number_input("躯干前倾 (°):", min_value=0.0,  max_value=90.0,  value=38.0, step=1.0)
HQ_ratio = st.number_input("腘绳肌/股四头肌 H/Q:", min_value=0.0, max_value=3.0, value=0.71, step=0.01)

feature_values = [HFA, HAA, KFA, ITR, KAA, AFA, FPA, TFA, HQ_ratio]
features = np.array([feature_values])  # shape = (1, 9)

# ========= 4. 点击按钮：在这里才去加载模型 & 预测 =========
if st.button("Predict"):
    # 4.1 先尝试加载模型，如果缺库，用 st.error 显示真正报错信息
    try:
        model = load_model_safe()
    except ModuleNotFoundError as e:
        # 这里会显示类似: No module named 'imblearn.pipeline'
        st.error(f"加载模型失败，缺少依赖库：{e}")
        st.stop()
    except Exception as e:
        st.error(f"加载模型失败，其他错误：{e}")
        st.stop()

    # 4.2 预测 ACL 值（假设为 ×BW）
    acl_bw = float(np.asarray(model.predict(features)).ravel()[0])
    st.write(f"**Predicted ACL load (×BW):** {acl_bw:.2f}")

    # 4.3 风险分级：<2.00 低；2.00–2.45 中；≥2.45 高
    LOW_TH, HIGH_TH = 2.00, 2.45

    if acl_bw >= HIGH_TH:
        risk_label = "High risk"
        advice = (
            "- 增加触地时膝关节屈曲角度（≥30°），避免动态膝外翻。\n"
            "- 减少过度的胫骨内旋 / 足趾内扣，FPA 控制在大约 10–20°。\n"
            "- 加强腘绳肌和臀肌力量，改善 H/Q 共激活与髋控。\n"
            "- 进行专项技术与神经肌肉训练，并注意训练/比赛负荷管理。\n"
            "- 如出现不稳或疼痛，建议就诊运动医学专科。"
        )
    elif acl_bw >= LOW_TH:
        risk_label = "Moderate risk"
        advice = (
            "- 保持中等膝屈曲与近似中立的下肢对线。\n"
            "- 维持腘绳肌力量与耐力，重点加强髋—膝动态控制。\n"
            "- 结合弹跳/落地训练，逐步增加难度并监测疲劳。"
        )
    else:
        risk_label = "Low risk"
        advice = (
            "- 继续保持当前技术动作和神经肌肉训练。\n"
            "- 在高疲劳或更复杂动作条件下，建议定期复评 ACL 负荷。"
        )

    st.markdown(f"**Risk level:** {risk_label}")
    st.markdown("**Recommendations:**\n" + advice)

    # ========= 5. SHAP 单样本解释 =========
    st.subheader("SHAP Force Plot Explanation")

    explainer_shap = shap.TreeExplainer(model)
    input_df = pd.DataFrame(features, columns=feature_names)
    shap_values = explainer_shap.shap_values(input_df)

    plt.figure(figsize=(8, 2.5))
    shap.force_plot(
        explainer_shap.expected_value,
        shap_values[0, :],
        input_df.iloc[0, :],
        matplotlib=True,
        show=False,
    )
    st.pyplot(plt.gcf())
    plt.close()
