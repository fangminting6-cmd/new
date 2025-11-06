# 导入库
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


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

# ===================== 2. Streamlit UI =====================
st.title("ACL Injury Risk Predictor")

# 角度可以用 float 会更合理，这里给出一个常用范围示例，你可以根据数据再微调
HFA = st.number_input("hip flexion angle (HFA):",   min_value=0.0,  max_value=120.0, value=43.0, step=1.0)
HAA = st.number_input("hip abduction angle (HAA):",   min_value=-30.0, max_value=30.0,  value=3.0,  step=1.0)
KFA = st.number_input("knee flexion angle (KFA):",   min_value=0.0,  max_value=120.0, value=29.0, step=1.0)
ITR = st.number_input("internal tibial rotation angle (ITR):", min_value=-30.0, max_value=30.0,  value=8.0,  step=1.0)
KAA = st.number_input("knee valgus angle (KVA):",   min_value=-15.0, max_value=30.0,  value=10.0, step=1.0)
AFA = st.number_input("ankle flexion angle (AFA):",   min_value=-20.0, max_value=40.0,  value=21.0, step=1.0)
FPA = st.number_input("foot progression angle (FPA):",   min_value=-30.0, max_value=40.0,  value=13.0, step=1.0)
TFA = st.number_input("trunk flexion angle (TFA):", min_value=0.0,  max_value=90.0,  value=38.0, step=1.0)

# H/Q 比建议用 float 范围
HQ_ratio = st.number_input("H/Q:", min_value=0.0, max_value=3.0, value=0.71, step=0.01)

# 组装成模型输入
feature_values = [HFA, HAA, KFA, ITR, KAA, AFA, FPA, TFA, HQ_ratio]
features = np.array([feature_values])  # shape = (1, 9)

# ===================== 3. 点击按钮进行预测 =====================
if st.button("Predict"):
    # ---------- 3.1 预测 ACL （假设输出单位为 ×BW） ----------
    acl_bw = float(np.asarray(model.predict(features)).ravel()[0])
    st.write(f"**Predicted ACL load (×BW):** {acl_bw:.2f}")

    # ---------- 3.2 风险分级 ----------
    HIGH_TH = 2.45

    if acl_bw < HIGH_TH:
        risk_label = "High risk"
        advice = (
    "- Increase knee flexion angle at initial contact (≥30°) to avoid dynamic knee valgus.\n"
    "- Reduce excessive tibial internal rotation / toe-in; keep the foot progression angle around 10–20°.\n"
    "- Strengthen the hamstrings and gluteal muscles, and improve H/Q co-activation and hip control.\n"
    "- Incorporate sport-specific technique and neuromuscular training, and monitor training/competition load.\n"
    "- If instability or pain is present, consult a sports medicine professional."
)

    else:
    risk_label = "Low risk"
    advice = (
    "- The current ACL load is relatively low; you may continue with your existing training program.\n"
    "- Maintain lower-limb strength and neuromuscular control, and pay attention to movement quality under fatigue.\n"
    "- Reassess periodically to monitor changes in risk."
)


    st.markdown(f"**Risk level:** {risk_label}")
    st.markdown("**Recommendations:**\n" + advice)

    # ===================== 4. SHAP 单样本解释 =====================
    st.subheader("SHAP Force Plot Explanation")

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
        shap_values[0, :],             # 当前样本的 SHAP 值
        input_df.iloc[0, :],           # 当前样本的特征
        matplotlib=True,
        show=False                     # 不要自动 show
    )

    # 方式一：直接在 Streamlit 里显示
    st.pyplot(plt.gcf())

    # 若你还想保存成文件：
    plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=300)
    plt.close()
    # st.image("shap_force_plot.png", caption="SHAP Force Plot Explanation")
