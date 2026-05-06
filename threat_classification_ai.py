import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="APS Threat Classification AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ APS Threat Classification AI Model")
st.write("This AI model classifies an incoming threat as LOW, MEDIUM, or HIGH based on velocity, distance, and approach angle.")

def classify_rule(velocity, distance, angle):
    if velocity <= 0:
        tti = 999
    else:
        tti = distance / velocity

    if velocity > 1200 or tti < 0.05 or angle >= 60:
        return 2
    elif velocity > 500 or tti < 0.15 or angle >= 30:
        return 1
    else:
        return 0

@st.cache_resource
def train_model():
    data = []
    for _ in range(3000):
        velocity = np.random.uniform(100, 1800)
        distance = np.random.uniform(20, 500)
        angle = np.random.uniform(0, 90)
        label = classify_rule(velocity, distance, angle)
        data.append([velocity, distance, angle, label])

    df = pd.DataFrame(data, columns=["Velocity", "Distance", "Angle", "Threat_Class"])

    X = df[["Velocity", "Distance", "Angle"]]
    y = df["Threat_Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler, df

model, scaler, df = train_model()

st.sidebar.header("Input Threat Parameters")

velocity = st.sidebar.slider("Threat Velocity (m/s)", 100, 1800, 900)
distance = st.sidebar.slider("Distance from Vehicle (m)", 20, 500, 120)
angle = st.sidebar.slider("Approach Angle (degrees)", 0, 90, 45)

input_data = pd.DataFrame([[velocity, distance, angle]], columns=["Velocity", "Distance", "Angle"])
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0]

labels = {
    0: "LOW THREAT",
    1: "MEDIUM THREAT",
    2: "HIGH THREAT"
}

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Velocity", f"{velocity} m/s")

with col2:
    st.metric("Distance", f"{distance} m")

with col3:
    st.metric("Angle", f"{angle}°")

st.markdown("## Prediction Result")

if prediction == 2:
    st.error(f"🔴 {labels[prediction]}")
elif prediction == 1:
    st.warning(f"🟡 {labels[prediction]}")
else:
    st.success(f"🟢 {labels[prediction]}")

st.markdown("## Confidence Score")

confidence_df = pd.DataFrame({
    "Threat Level": ["LOW", "MEDIUM", "HIGH"],
    "Confidence (%)": [
        probability[0] * 100,
        probability[1] * 100,
        probability[2] * 100
    ]
})

st.bar_chart(confidence_df.set_index("Threat Level"))

st.markdown("## Technical Decision Parameters")

tti = distance / velocity

result_table = pd.DataFrame({
    "Parameter": [
        "Threat Velocity",
        "Distance from Vehicle",
        "Approach Angle",
        "Estimated Time-to-Impact",
        "Predicted Threat Class"
    ],
    "Value": [
        f"{velocity} m/s",
        f"{distance} m",
        f"{angle} degrees",
        f"{tti:.4f} seconds",
        labels[prediction]
    ]
})

st.table(result_table)

st.markdown("## Dataset Preview Used for AI Training")
st.dataframe(df.head(20))

st.markdown("## Thesis Explanation")
st.write("""
The developed AI model acts as a decision-support layer for an Active Protection System.
It classifies incoming threats based on velocity, distance, and angle of approach.
A synthetic dataset is generated using APS-based decision logic, and a neural network
classifier is trained to predict whether the threat level is LOW, MEDIUM, or HIGH.
This supports fast threat prioritization before interceptor launch.
""")
