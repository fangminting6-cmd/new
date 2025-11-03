# 导入 Streamlit 库，用于构建 Web 应用
import streamlit as st

# 导入 NumPy 库，用于数值计算
import numpy as np

# 导入 Pandas 库，用于数据处理和操作
import pandas as pd

# 导入 SHAP 库，用于解释机器学习模型的预测
import shap

# 导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt

# 加载训练好的模型
model = joblib.load('final_XGJ_model.pkl')

# 从 X_test.csv 文件加载测试数据，用于 LIME 解释器
X_test = pd.read_csv('X_test.csv')

# 定义特征名称，对应数据集中的列名
feature_names = [
    "HFA",       
    "HAA",        
    "KFA",  
    "ITR",      
    "KAA",      
    "AFA",   
    "FPA",   
    "TFA",     
    "HQ_ratio",   
]

# Streamlit 用户界面
st.title("ACL Injury Risk Predictor")  # 设置网页标题

HFA = st.number_input("髋屈曲:", min_value=0, max_value=100, value=43)

HAA = st.snumber_input("髋外展：", min_value=0, max_value=100, value=3)

KFA = st.number_input("膝屈曲:", min_value=0, max_value=100, value=29)

ITR = st.number_input("胫骨内旋:", min_value=0, max_value=100, value=8)

KAA = st.number_input("膝外翻:", min_value=0, max_value=100, value=10)

AFA = st.number_input("踝屈曲:", min_value=0, max_value=100, value=21)

FPA = st.number_input("足外展:", min_value=0, max_value=100, value=13)

TFA = st.number_input("躯干前倾：", min_value=0, max_value=100, value=38)

HQ_ratio = st.number_input("腘绳肌/股四头肌：", min_value=0, max_value=100, value=0.71)

feature_values = [
    HFA, HAA, KFA, ITR, KAA, AFA, FPA,
    TFA, HQ_ratio
]  # 将用户输入的特征值存入列表
features = np.array([feature_values])  # 转为 NumPy 数组，适用于模型输入

# 当用户点击 “Predict” 按钮时执行
if st.button("Predict"):
    # 1) 预测 ACL 数值（回归）
    predicted_acl = float(np.asarray(model.predict(features)).ravel()[0])  # 标量
    st.write(f"**Predicted ACL (N):** {predicted_acl:.1f}")

    # 3) 风险分级与建议（示例阈值，可按你的研究调整）
    LOW_TH, HIGH_TH = 2.45, 2.45   # <2.00 低；2.00–2.45 中；≥2.45 高
    if acl_bw >= HIGH_TH:
        risk_label = "High risk"
        advice = (
            "- Increase knee flexion at initial contact (≥30°); avoid dynamic valgus.\n"
            "- Reduce excessive tibial internal rotation / toe-in; keep FPA ~10–20°.\n"
            "- Strengthen hamstrings & gluteals; improve H/Q co-activation.\n"
            "- Technique coaching, neuromuscular training, and workload control.\n"
            "- If symptoms/instability present, consult sports medicine."
        )
    elif acl_bw >= LOW_TH:
        risk_label = "Moderate risk"
        advice = (
            "- Target moderate knee flexion and neutral alignment.\n"
            "- Maintain hamstring strength/endurance; emphasize hip control.\n"
            "- Use landing drills & progressive plyometrics; monitor fatigue."
        )
    else:
        risk_label = "Low risk"
        advice = (
            "- Maintain current technique & neuromuscular training.\n"
            "- Reassess under fatigue or higher task complexity."
        )

    st.markdown(f"**Risk level:** {risk_label}")
    st.markdown("**Recommendations:**\n" + advice)


    # SHAP 解释
st.subheader("SHAP Force Plot Explanation")

# 创建 SHAP 解释器（基于树模型，如随机森林/梯度提升）
explainer_shap = shap.TreeExplainer(model)

# 计算 SHAP 值，用于解释模型预测
# 将单条用户输入转换为 DataFrame，并指定列名
input_df = pd.DataFrame([feature_values], columns=feature_names)
shap_values = explainer_shap.shap_values(input_df)

# 根据预测类别显示 SHAP 力图
# 期望值（baseline） + 对应类别的 SHAP 值 + 特征值数据
# 使用 Matplotlib 后端以便保存为图片
if predicted_class == 1:
    shap.force_plot(
        explainer_shap.expected_value[1],
        shap_values[:, :, 1],
        input_df,
        matplotlib=True
    )
else:
    shap.force_plot(
        explainer_shap.expected_value[0],
        shap_values[:, :, 0],
        input_df,
        matplotlib=True
    )

# 保存图像
plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=1200)
st.image("shap_force_plot.png", caption="SHAP Force Plot Explanation")