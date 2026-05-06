import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="APS Threat Classification AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Based Threat Classification System for Active Protection System")
st.write(
    "This application classifies incoming threats and visualizes a 3D engagement scenario "
    "between a Main Battle Tank and an incoming projectile."
)


# ============================================================
# THREAT CLASSIFICATION LOGIC
# ============================================================

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

    for _ in range(4000):
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
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
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


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Threat Input Parameters")

threat_type = st.sidebar.selectbox(
    "Threat Type",
    [
        "HEAT / HESH",
        "FSAPDS / High-Velocity KE"
    ]
)

if threat_type == "HEAT / HESH":
    velocity = st.sidebar.slider(
        "Threat Velocity (m/s)",
        min_value=100,
        max_value=400,
        value=200
    )
else:
    velocity = st.sidebar.slider(
        "Threat Velocity (m/s)",
        min_value=400,
        max_value=900,
        value=600
    )

distance = st.sidebar.slider(
    "Initial Distance from Tank (m)",
    min_value=20,
    max_value=500,
    value=150
)

angle = st.sidebar.slider(
    "Incoming Angle (degrees)",
    min_value=0,
    max_value=90,
    value=30
)

tti = distance / velocity


# ============================================================
# AI PREDICTION
# ============================================================

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


# ============================================================
# OUTPUT METRICS
# ============================================================

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

st.markdown("## AI Confidence Score")
st.bar_chart(confidence_df.set_index("Threat Level"))


# ============================================================
# 3D MBT MODEL FUNCTIONS
# ============================================================

def cuboid(x0, x1, y0, y1, z0, z1, name="Part", opacity=0.85):
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]

    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6, 6]
    j = [1, 2, 4, 2, 5, 3, 5, 6, 6, 1, 7, 2]
    k = [2, 3, 5, 5, 6, 7, 6, 7, 1, 0, 2, 3]

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=opacity,
        name=name,
        color="olive"
    )


def cylinder_between(x0, x1, y0, y1, z0, z1, radius=0.12, name="Cylinder", n=28):
    t = np.linspace(0, 2 * np.pi, n)

    x = []
    y = []
    z = []

    for xi, yi, zi in [(x0, y0, z0), (x1, y1, z1)]:
        x.extend([xi] * n)
        y.extend(yi + radius * np.cos(t))
        z.extend(zi + radius * np.sin(t))

    faces_i, faces_j, faces_k = [], [], []

    for a in range(n - 1):
        faces_i.append(a)
        faces_j.append(a + 1)
        faces_k.append(a + n)

        faces_i.append(a + 1)
        faces_j.append(a + n + 1)
        faces_k.append(a + n)

    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=faces_i,
        j=faces_j,
        k=faces_k,
        opacity=0.9,
        name=name,
        color="darkolivegreen"
    )


def create_mbt_model():
    parts = []

    # Main hull
    parts.append(cuboid(-4.5, 4.5, -2.1, 2.1, 0.35, 1.25, "Main Hull"))

    # Front glacis
    parts.append(cuboid(-5.3, -4.3, -1.8, 1.8, 0.35, 0.95, "Sloped Front Hull", 0.8))

    # Rear deck
    parts.append(cuboid(3.0, 4.8, -1.8, 1.8, 1.25, 1.55, "Engine Deck"))

    # Turret
    parts.append(cuboid(-1.7, 1.8, -1.35, 1.35, 1.25, 2.10, "Turret Base"))
    parts.append(cuboid(-1.2, 1.2, -1.0, 1.0, 2.10, 2.45, "Turret Top"))

    # Gun mantlet and barrel
    parts.append(cuboid(-2.2, -1.55, -0.55, 0.55, 1.55, 2.05, "Gun Mantlet"))
    parts.append(cylinder_between(-2.2, -7.5, 0, 0, 1.78, 1.78, 0.16, "Main Gun Barrel"))
    parts.append(cylinder_between(-7.5, -8.6, 0, 0, 1.78, 1.78, 0.11, "Gun Muzzle"))

    # Tracks
    parts.append(cuboid(-4.8, 4.7, -2.55, -2.0, 0.0, 0.65, "Left Track", 0.95))
    parts.append(cuboid(-4.8, 4.7, 2.0, 2.55, 0.0, 0.65, "Right Track", 0.95))

    # Road wheels
    for xw in np.linspace(-3.8, 3.7, 6):
        parts.append(cylinder_between(xw, xw, -2.58, -2.62, 0.32, 0.32, 0.32, "Left Road Wheel"))
        parts.append(cylinder_between(xw, xw, 2.58, 2.62, 0.32, 0.32, 0.32, "Right Road Wheel"))

    # ERA / armor blocks on turret
    for xb in [-1.3, -0.5, 0.3, 1.1]:
        parts.append(cuboid(xb, xb + 0.45, -1.55, -1.25, 1.75, 2.1, "Left ERA Block", 0.9))
        parts.append(cuboid(xb, xb + 0.45, 1.25, 1.55, 1.75, 2.1, "Right ERA Block", 0.9))

    # Side armor
    for xb in np.linspace(-3.8, 2.8, 7):
        parts.append(cuboid(xb, xb + 0.55, -2.15, -1.85, 1.25, 1.55, "Left Side Armor", 0.9))
        parts.append(cuboid(xb, xb + 0.55, 1.85, 2.15, 1.25, 1.55, "Right Side Armor", 0.9))

    # Hatches
    parts.append(cylinder_between(0.3, 0.3, 0.0, 0.04, 2.52, 2.52, 0.45, "Commander Hatch"))
    parts.append(cylinder_between(-0.7, -0.7, 0.0, 0.04, 2.5, 2.5, 0.35, "Gunner Hatch"))

    # Antenna
    parts.append(go.Scatter3d(
        x=[1.5, 1.5],
        y=[1.0, 1.0],
        z=[2.3, 5.2],
        mode="lines",
        line=dict(width=5, color="black"),
        name="Antenna"
    ))

    # Rear fuel drum
    parts.append(cylinder_between(4.3, 4.3, -1.2, 1.2, 1.85, 1.85, 0.38, "Rear Fuel Drum"))

    # APS launcher pucks around hull
    launcher_positions = [
        (-4.2, -2.4, 1.35),
        (-4.2, 2.4, 1.35),
        (0.0, -2.4, 1.45),
        (0.0, 2.4, 1.45),
        (4.2, -2.4, 1.35),
        (4.2, 2.4, 1.35),
        (0.0, 0.0, 2.65)
    ]

    for lx, ly, lz in launcher_positions:
        parts.append(cylinder_between(lx, lx, ly, ly + 0.05, lz, lz, 0.22, "APS Launcher Module"))

    return parts


# ============================================================
# THREAT TRAJECTORY AND ANIMATION
# ============================================================

st.markdown("## 3D MBT Threat Engagement Animation")

theta = np.radians(angle)

start_x = -distance * np.cos(theta) / 20
start_y = -distance * np.sin(theta) / 20
start_z = 4.5 if angle >= 45 else 1.6

end_x = 0.0
end_y = 0.0
end_z = 1.55

n_frames = 32

x_path = np.linspace(start_x, end_x, n_frames)
y_path = np.linspace(start_y, end_y, n_frames)
z_path = np.linspace(start_z, end_z, n_frames)

trajectory = go.Scatter3d(
    x=x_path,
    y=y_path,
    z=z_path,
    mode="lines",
    name="Incoming Threat Path",
    line=dict(width=6, color="red")
)

threat_marker = go.Scatter3d(
    x=[x_path[0]],
    y=[y_path[0]],
    z=[z_path[0]],
    mode="markers+text",
    marker=dict(size=8, color="red"),
    text=[f"{threat_type}<br>{velocity} m/s"],
    textposition="top center",
    name="Incoming Threat"
)

frames = []

for idx in range(n_frames):
    frames.append(
        go.Frame(
            data=[
                go.Scatter3d(
                    x=[x_path[idx]],
                    y=[y_path[idx]],
                    z=[z_path[idx]],
                    mode="markers+text",
                    marker=dict(size=9, color="red"),
                    text=[f"{threat_type}<br>{velocity} m/s"],
                    textposition="top center",
                    name="Incoming Threat"
                )
            ],
            name=str(idx)
        )
    )

tank_parts = create_mbt_model()

fig = go.Figure(
    data=tank_parts + [trajectory, threat_marker],
    frames=frames
)

fig.update_layout(
    title="3D Threat Approach Scenario Toward Main Battle Tank",
    scene=dict(
        xaxis_title="X Direction",
        yaxis_title="Y Direction",
        zaxis_title="Height",
        aspectmode="data",
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.3)
        )
    ),
    height=700,
    showlegend=False,
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.05,
            y=1.05,
            buttons=[
                dict(
                    label="▶ Play Threat Animation",
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


# ============================================================
# TECHNICAL OUTPUT TABLE
# ============================================================

st.markdown("## Technical Output Table")

technical_table = pd.DataFrame({
    "Parameter": [
        "Threat Type",
        "Velocity",
        "Distance",
        "Incoming Angle",
        "Estimated Time-to-Impact",
        "AI Classified Threat Level"
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

st.table(technical_table)


# ============================================================
# THESIS EXPLANATION
# ============================================================

st.markdown("## Thesis Explanation")

st.write("""
This AI-based module classifies incoming threats for an Active Protection System using three critical kinematic parameters:
velocity, distance, and approach angle. The model uses a neural network trained on a synthetic APS-based dataset generated
from rule-based engagement logic. The 3D visualization represents a simplified Main Battle Tank equipped with distributed
APS launcher modules. The incoming threat path is animated based on the selected velocity, distance, and angle, allowing
visual interpretation of time-to-impact and engagement urgency.

HEAT/HESH threats are represented using lower velocity ranges around 200 m/s, while FSAPDS or high-velocity kinetic
energy threats are represented using higher velocity ranges around 600 m/s and above. This AI-assisted visualization
supports threat prioritization and can be integrated conceptually with the ETC-DMA launcher framework developed in this thesis.
""")
