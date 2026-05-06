import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="APS Threat Classification AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ APS Threat Classification AI with 3D Threat Animation")
st.write(
    "This dashboard classifies incoming threats and visualizes a 3D incoming threat trajectory "
    "toward a Main Battle Tank model."
)

# -----------------------------
# Rule-based threat label
# -----------------------------

def classify_rule(velocity, distance, angle):
    tti = distance / velocity if velocity > 0 else 999

    if velocity >= 600 or tti < 0.08 or angle >= 60:
        return 2
    elif velocity >= 200 or tti < 0.25 or angle >= 30:
        return 1
    else:
        return 0


@st.cache_resource
def train_model():
    data = []

    for _ in range(3000):
        velocity = np.random.uniform(100, 900)
        distance = np.random.uniform(20, 500)
        angle = np.random.uniform(0, 90)
        label = classify_rule(velocity, distance, angle)
        data.append([velocity, distance, angle, label])

    df = pd.DataFrame(
        data,
        columns=["Velocity", "Distance", "Angle", "Threat_Class"]
    )

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

# -----------------------------
# Sidebar inputs
# -----------------------------

st.sidebar.header("Threat Input Parameters")

threat_type = st.sidebar.selectbox(
    "Threat Type",
    ["HEAT / HESH", "FSAPDS / High-Velocity KE"]
)

if threat_type == "HEAT / HESH":
    velocity = st.sidebar.slider("Threat Velocity (m/s)", 100, 400, 200)
else:
    velocity = st.sidebar.slider("Threat Velocity (m/s)", 400, 900, 600)

distance = st.sidebar.slider("Initial Distance from Tank (m)", 20, 500, 150)
angle = st.sidebar.slider("Incoming Angle (degrees)", 0, 90, 30)

input_data = pd.DataFrame(
    [[velocity, distance, angle]],
    columns=["Velocity", "Distance", "Angle"]
)

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0]

labels = {
    0: "LOW THREAT",
    1: "MEDIUM THREAT",
    2: "HIGH THREAT"
}

tti = distance / velocity

# -----------------------------
# Output metrics
# -----------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Threat Type", threat_type)

with col2:
    st.metric("Velocity", f"{velocity} m/s")

with col3:
    st.metric("Distance", f"{distance} m")

with col4:
    st.metric("Time-to-Impact", f"{tti:.3f} s")

st.markdown("## AI Threat Classification Result")

if prediction == 2:
    st.error(f"🔴 {labels[prediction]}")
elif prediction == 1:
    st.warning(f"🟡 {labels[prediction]}")
else:
    st.success(f"🟢 {labels[prediction]}")

confidence_df = pd.DataFrame({
    "Threat Level": ["LOW", "MEDIUM", "HIGH"],
    "Confidence (%)": [
        probability[0] * 100,
        probability[1] * 100,
        probability[2] * 100
    ]
})

st.bar_chart(confidence_df.set_index("Threat Level"))

# -----------------------------
# 3D tank + incoming threat animation
# -----------------------------

st.markdown("## 3D Incoming Threat Animation Toward Main Battle Tank")

def create_cuboid(x0, x1, y0, y1, z0, z1, name):
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]

    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6, 6]
    j = [1, 2, 4, 2, 5, 3, 5, 6, 6, 1, 7, 2]
    k = [2, 3, 5, 5, 6, 7, 6, 7, 1, 0, 2, 3]

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.75,
        name=name
    )

# Tank model parts
hull = create_cuboid(-4, 4, -2, 2, 0, 1.2, "Tank Hull")
turret = create_cuboid(-1.6, 1.6, -1.2, 1.2, 1.2, 2.2, "Turret")
gun = create_cuboid(1.6, 6.0, -0.15, 0.15, 1.55, 1.85, "Gun Barrel")

# Threat trajectory
theta = np.radians(angle)

start_x = -distance * np.cos(theta) / 20
start_y = -distance * np.sin(theta) / 20
start_z = 4 if angle >= 45 else 1.5

end_x = 0
end_y = 0
end_z = 1.5

n_frames = 25

x_path = np.linspace(start_x, end_x, n_frames)
y_path = np.linspace(start_y, end_y, n_frames)
z_path = np.linspace(start_z, end_z, n_frames)

trajectory = go.Scatter3d(
    x=x_path,
    y=y_path,
    z=z_path,
    mode="lines",
    name="Threat Path",
    line=dict(width=5)
)

threat_marker = go.Scatter3d(
    x=[x_path[0]],
    y=[y_path[0]],
    z=[z_path[0]],
    mode="markers+text",
    marker=dict(size=8),
    text=[threat_type],
    textposition="top center",
    name="Incoming Threat"
)

frames = []

for idx in range(n_frames):
    frame = go.Frame(
        data=[
            go.Scatter3d(
                x=[x_path[idx]],
                y=[y_path[idx]],
                z=[z_path[idx]],
                mode="markers+text",
                marker=dict(size=8),
                text=[f"{threat_type}<br>{velocity} m/s"],
                textposition="top center",
                name="Incoming Threat"
            )
        ],
        name=str(idx)
    )
    frames.append(frame)

fig = go.Figure(
    data=[hull, turret, gun, trajectory, threat_marker],
    frames=frames
)

fig.update_layout(
    scene=dict(
        xaxis_title="X Direction",
        yaxis_title="Y Direction",
        zaxis_title="Height",
        aspectmode="data",
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.2)
        )
    ),
    height=650,
    title="3D APS Threat Scenario: Incoming Threat Toward MBT",
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="▶ Play Animation",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=120, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True
                        )
                    ]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate"
                        )
                    ]
                )
            ]
        )
    ]
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Technical table
# -----------------------------

st.markdown("## Technical Output Table")

result_table = pd.DataFrame({
    "Parameter": [
        "Threat Type",
        "Velocity",
        "Distance",
        "Incoming Angle",
        "Estimated Time-to-Impact",
        "Predicted Threat Class"
    ],
    "Value": [
        threat_type,
        f"{velocity} m/s",
        f"{distance} m",
        f"{angle} degrees",
        f"{tti:.4f} seconds",
        labels[prediction]
    ]
})

st.table(result_table)

# -----------------------------
# Thesis explanation
# -----------------------------

st.markdown("## Thesis Explanation")

st.write("""
The 3D animation represents a simplified Active Protection System engagement scenario.
A notional Main Battle Tank is placed at the origin, while an incoming threat approaches
from a user-defined distance and angle. The threat classification model uses velocity,
distance, and approach angle to classify the incoming object as LOW, MEDIUM, or HIGH threat.

For demonstration, HEAT/HESH threats are represented with lower velocity ranges around
200 m/s, while FSAPDS/high-velocity kinetic threats are represented with higher velocity
ranges around 600 m/s and above. The animation supports visual interpretation of
time-to-impact and engagement urgency, which is useful for APS decision-support studies.
""")
