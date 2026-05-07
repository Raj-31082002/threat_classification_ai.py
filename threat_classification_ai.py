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

st.title("🛡️ AI-Based APS Threat Classification with 3D Engagement Animation")
st.write(
    "This dashboard classifies incoming threats and visualizes a 3D APS engagement scenario "
    "with detection range, interception range, distributed launcher modules, and incoming threat animation."
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

st.sidebar.header("Threat Scenario Inputs")

threat_type = st.sidebar.selectbox(
    "Threat Type",
    [
        "HEAT / HESH",
        "FSAPDS / High-Velocity KE",
        "RPG / ATGM Type Threat"
    ]
)

if threat_type == "HEAT / HESH":
    velocity = st.sidebar.slider("Threat Velocity (m/s)", 100, 400, 200)
elif threat_type == "FSAPDS / High-Velocity KE":
    velocity = st.sidebar.slider("Threat Velocity (m/s)", 400, 900, 600)
else:
    velocity = st.sidebar.slider("Threat Velocity (m/s)", 150, 500, 300)

distance = st.sidebar.slider("Initial Threat Distance (m)", 20, 500, 180)
angle = st.sidebar.slider("Incoming Angle (degrees)", 0, 90, 25)

detection_range_m = st.sidebar.slider("Detection Range (m)", 100, 800, 350)
interception_range_m = st.sidebar.slider("Interception Range (m)", 20, 250, 120)

animation_frames = st.sidebar.slider("Animation Smoothness", 20, 70, 40)

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
# 3D GEOMETRY UTILITIES
# ============================================================

def cuboid(x0, x1, y0, y1, z0, z1, name="Part", color="olive", opacity=0.85):
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
        color=color,
        flatshading=True
    )


def cylinder_x(x0, x1, y, z, radius=0.12, name="Cylinder", color="darkolivegreen", n=32):
    theta = np.linspace(0, 2 * np.pi, n)
    x = []
    yy = []
    zz = []

    for xi in [x0, x1]:
        x.extend([xi] * n)
        yy.extend(y + radius * np.cos(theta))
        zz.extend(z + radius * np.sin(theta))

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
        y=yy,
        z=zz,
        i=faces_i,
        j=faces_j,
        k=faces_k,
        opacity=0.95,
        name=name,
        color=color
    )


def cylinder_y(x, y0, y1, z, radius=0.12, name="Cylinder", color="darkolivegreen", n=32):
    theta = np.linspace(0, 2 * np.pi, n)
    xx = []
    y = []
    zz = []

    for yi in [y0, y1]:
        xx.extend(x + radius * np.cos(theta))
        y.extend([yi] * n)
        zz.extend(z + radius * np.sin(theta))

    faces_i, faces_j, faces_k = [], [], []

    for a in range(n - 1):
        faces_i.append(a)
        faces_j.append(a + 1)
        faces_k.append(a + n)

        faces_i.append(a + 1)
        faces_j.append(a + n + 1)
        faces_k.append(a + n)

    return go.Mesh3d(
        x=xx,
        y=y,
        z=zz,
        i=faces_i,
        j=faces_j,
        k=faces_k,
        opacity=0.95,
        name=name,
        color=color
    )


def hemisphere(radius=10, name="Range Dome", color="lightblue", opacity=0.15, z_offset=0.0):
    phi = np.linspace(0, np.pi / 2, 35)
    theta = np.linspace(0, 2 * np.pi, 70)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi) + z_offset

    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=opacity,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name
    )


def range_ring(radius=10, z=0.02, name="Range Ring", color="cyan"):
    theta = np.linspace(0, 2 * np.pi, 200)

    return go.Scatter3d(
        x=radius * np.cos(theta),
        y=radius * np.sin(theta),
        z=np.full_like(theta, z),
        mode="lines",
        line=dict(color=color, width=5),
        name=name
    )


def create_ground(size=26):
    x = np.linspace(-size, size, 2)
    y = np.linspace(-size, size, 2)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x) - 0.02

    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.35,
        colorscale=[[0, "tan"], [1, "tan"]],
        showscale=False,
        name="Ground Plane"
    )


# ============================================================
# DETAILED 3D MBT MODEL WITH DMA LAUNCHERS
# ============================================================

def create_mbt_model():
    parts = []

    # Hull
    parts.append(cuboid(-4.6, 4.6, -2.1, 2.1, 0.35, 1.20, "Main Hull", "darkolivegreen"))
    parts.append(cuboid(-5.2, -4.3, -1.8, 1.8, 0.35, 0.95, "Front Glacis", "olivedrab", 0.85))
    parts.append(cuboid(3.2, 4.8, -1.8, 1.8, 1.20, 1.52, "Engine Deck", "darkkhaki", 0.85))

    # Turret
    parts.append(cuboid(-1.8, 1.7, -1.35, 1.35, 1.20, 2.05, "Turret", "olive"))
    parts.append(cuboid(-1.1, 1.1, -0.85, 0.85, 2.05, 2.40, "Turret Top", "darkolivegreen"))

    # Gun
    parts.append(cuboid(-2.3, -1.65, -0.55, 0.55, 1.50, 2.05, "Gun Mantlet", "darkkhaki"))
    parts.append(cylinder_x(-2.3, -8.2, 0, 1.78, 0.16, "Main Gun Barrel", "darkolivegreen"))
    parts.append(cylinder_x(-8.2, -9.0, 0, 1.78, 0.10, "Gun Muzzle", "black"))

    # Tracks
    parts.append(cuboid(-4.9, 4.8, -2.60, -2.05, 0.00, 0.65, "Left Track", "dimgray", 0.95))
    parts.append(cuboid(-4.9, 4.8, 2.05, 2.60, 0.00, 0.65, "Right Track", "dimgray", 0.95))

    # Road wheels
    for xw in np.linspace(-3.8, 3.8, 7):
        parts.append(cylinder_y(xw, -2.67, -2.60, 0.32, 0.30, "Left Road Wheel", "gray"))
        parts.append(cylinder_y(xw, 2.60, 2.67, 0.32, 0.30, "Right Road Wheel", "gray"))

    # Side armor blocks
    for xb in np.linspace(-3.8, 2.9, 8):
        parts.append(cuboid(xb, xb + 0.50, -2.15, -1.82, 1.20, 1.52, "Left Armor Block", "khaki", 0.92))
        parts.append(cuboid(xb, xb + 0.50, 1.82, 2.15, 1.20, 1.52, "Right Armor Block", "khaki", 0.92))

    # Turret ERA blocks
    for xb in [-1.35, -0.55, 0.25, 1.05]:
        parts.append(cuboid(xb, xb + 0.42, -1.55, -1.25, 1.72, 2.08, "Turret ERA Left", "khaki", 0.92))
        parts.append(cuboid(xb, xb + 0.42, 1.25, 1.55, 1.72, 2.08, "Turret ERA Right", "khaki", 0.92))

    # Hatches
    parts.append(cylinder_y(0.35, -0.04, 0.04, 2.52, 0.42, "Commander Hatch", "darkslategray"))
    parts.append(cylinder_y(-0.75, -0.04, 0.04, 2.48, 0.34, "Gunner Hatch", "darkslategray"))

    # Rear fuel drum
    parts.append(cylinder_y(4.35, -1.15, 1.15, 1.85, 0.38, "Rear Fuel Drum", "darkolivegreen"))

    # Antenna
    parts.append(go.Scatter3d(
        x=[1.5, 1.5],
        y=[1.0, 1.0],
        z=[2.25, 5.1],
        mode="lines",
        line=dict(width=5, color="black"),
        name="Antenna"
    ))

    # DMA launcher modules - distributed around tank
    launcher_positions = [
        (-4.5, -2.45, 1.42, "Front Left DMA"),
        (-4.5, 2.45, 1.42, "Front Right DMA"),
        (0.0, -2.45, 1.52, "Side Left DMA"),
        (0.0, 2.45, 1.52, "Side Right DMA"),
        (4.3, -2.45, 1.42, "Rear Left DMA"),
        (4.3, 2.45, 1.42, "Rear Right DMA"),
        (0.2, 0.0, 2.72, "Top DMA")
    ]

    for lx, ly, lz, lname in launcher_positions:
        parts.append(cylinder_y(lx, ly - 0.18, ly + 0.18, lz, 0.18, lname, "gold"))

    # Radar / sensor modules
    sensor_points = [
        (-4.7, 0, 1.35),
        (0, -2.7, 1.35),
        (0, 2.7, 1.35),
        (4.6, 0, 1.35),
        (0, 0, 2.85)
    ]

    for sx, sy, sz in sensor_points:
        parts.append(go.Scatter3d(
            x=[sx],
            y=[sy],
            z=[sz],
            mode="markers+text",
            marker=dict(size=5, color="cyan"),
            text=["Sensor"],
            textposition="top center",
            name="APS Sensor"
        ))

    return parts


# ============================================================
# 3D APS ENGAGEMENT ANIMATION
# ============================================================

st.markdown("## 3D APS Engagement Scenario")

# Scale real metres to plot units
scale = 18.0

det_radius = detection_range_m / scale
int_radius = interception_range_m / scale

theta = np.radians(angle)

# Threat starts from front-left side like demo video
start_x = -distance * np.cos(theta) / scale
start_y = -distance * np.sin(theta) / scale
start_z = 4.3 if angle >= 45 else 1.6

tank_center = np.array([0.0, 0.0, 1.45])
start_point = np.array([start_x, start_y, start_z])

# Intercept point: point on trajectory at interception range from tank
direction = tank_center - start_point
direction = direction / np.linalg.norm(direction)

intercept_point = tank_center - direction * int_radius

end_point = tank_center

n_frames = animation_frames

x_path = np.linspace(start_point[0], end_point[0], n_frames)
y_path = np.linspace(start_point[1], end_point[1], n_frames)
z_path = np.linspace(start_point[2], end_point[2], n_frames)

# Find closest animation frame to intercept point
dist_to_intercept = np.sqrt(
    (x_path - intercept_point[0]) ** 2 +
    (y_path - intercept_point[1]) ** 2 +
    (z_path - intercept_point[2]) ** 2
)
intercept_idx = int(np.argmin(dist_to_intercept))

trajectory_full = go.Scatter3d(
    x=x_path,
    y=y_path,
    z=z_path,
    mode="lines",
    name="Threat Trajectory",
    line=dict(width=7, color="red")
)

trajectory_before_intercept = go.Scatter3d(
    x=x_path[:intercept_idx + 1],
    y=y_path[:intercept_idx + 1],
    z=z_path[:intercept_idx + 1],
    mode="lines",
    name="Path Before Intercept",
    line=dict(width=8, color="orange")
)

intercept_marker = go.Scatter3d(
    x=[x_path[intercept_idx]],
    y=[y_path[intercept_idx]],
    z=[z_path[intercept_idx]],
    mode="markers+text",
    marker=dict(size=9, color="lime"),
    text=["Intercept Point"],
    textposition="top center",
    name="Intercept Point"
)

threat_marker = go.Scatter3d(
    x=[x_path[0]],
    y=[y_path[0]],
    z=[z_path[0]],
    mode="markers+text",
    marker=dict(size=8, color="red", symbol="diamond"),
    text=[f"{threat_type}<br>{velocity} m/s"],
    textposition="top center",
    name="Incoming Threat"
)

# Detection and interception domes
detection_dome = hemisphere(det_radius, "Detection Range Dome", "lightblue", 0.12, 0.0)
interception_dome = hemisphere(int_radius, "Interception Range Dome", "lightgreen", 0.20, 0.0)

detection_ring = range_ring(det_radius, 0.03, "Detection Range Ring", "cyan")
interception_ring = range_ring(int_radius, 0.06, "Interception Range Ring", "lime")

ground = create_ground(size=max(26, det_radius + 4))

tank_parts = create_mbt_model()

frames = []

for idx in range(n_frames):
    current_color = "red"
    current_text = f"{threat_type}<br>{velocity} m/s"

    if idx >= intercept_idx:
        current_color = "lime"
        current_text = "Intercepted"

    frames.append(
        go.Frame(
            data=[
                go.Scatter3d(
                    x=[x_path[idx]],
                    y=[y_path[idx]],
                    z=[z_path[idx]],
                    mode="markers+text",
                    marker=dict(size=10, color=current_color, symbol="diamond"),
                    text=[current_text],
                    textposition="top center",
                    name="Incoming Threat"
                )
            ],
            name=str(idx)
        )
    )

fig = go.Figure(
    data=[
        ground,
        detection_dome,
        interception_dome,
        detection_ring,
        interception_ring,
    ] + tank_parts + [
        trajectory_full,
        trajectory_before_intercept,
        intercept_marker,
        threat_marker
    ],
    frames=frames
)

fig.update_layout(
    title="3D APS Threat Detection and Interception Visualization",
    scene=dict(
        xaxis_title="Forward Direction",
        yaxis_title="Lateral Direction",
        zaxis_title="Height",
        aspectmode="data",
        camera=dict(
            eye=dict(x=-1.75, y=-1.85, z=1.15),
            center=dict(x=0.0, y=0.0, z=-0.05)
        ),
        xaxis=dict(showbackground=True, backgroundcolor="rgb(230,220,200)"),
        yaxis=dict(showbackground=True, backgroundcolor="rgb(230,220,200)"),
        zaxis=dict(showbackground=True, backgroundcolor="rgb(240,240,240)")
    ),
    height=750,
    showlegend=True,
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor="rgba(255,255,255,0.65)"
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.03,
            y=1.05,
            buttons=[
                dict(
                    label="▶ Play APS Animation",
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

st.info(
    "Blue transparent dome = detection range. Green transparent dome = interception range. "
    "Gold modules = Distributed Modular Architecture launcher positions."
)


# ============================================================
# TECHNICAL OUTPUT TABLE
# ============================================================

st.markdown("## Technical Output Table")

technical_table = pd.DataFrame({
    "Parameter": [
        "Threat Type",
        "Velocity",
        "Initial Distance",
        "Incoming Angle",
        "Detection Range",
        "Interception Range",
        "Estimated Time-to-Impact",
        "AI Classified Threat Level"
    ],
    "Value": [
        threat_type,
        f"{velocity} m/s",
        f"{distance} m",
        f"{angle} degrees",
        f"{detection_range_m} m",
        f"{interception_range_m} m",
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
The developed AI-based APS threat classification module evaluates incoming threats using velocity,
distance, and approach angle. A neural network classifier is trained on a synthetic APS-oriented dataset
generated from decision rules based on engagement urgency and time-to-impact. The 3D visualization
represents a notional Main Battle Tank equipped with Distributed Modular Architecture launcher modules.
The blue dome indicates the detection envelope, while the green dome indicates the interception envelope.
The incoming threat trajectory is animated toward the vehicle and highlights the estimated interception
point.

This module supports system-level APS decision making by linking threat classification, time-to-impact,
detection range, and interception range in a single user-friendly visualization. HEAT/HESH type threats
are represented with lower velocity values, while FSAPDS/high-velocity kinetic threats are represented
with higher velocity values. The tool complements the ETC launcher simulation by providing an AI-based
threat assessment and engagement visualization layer.
""")
