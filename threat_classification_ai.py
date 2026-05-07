import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="APS Threat Classification AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Based APS Threat Classification and 3D Interception Animation")
st.caption(
    "Clean 3D APS concept visualization: threat detection, DMA launcher selection, "
    "interceptor launch, and neutralization point."
)


# ============================================================
# AI THREAT CLASSIFICATION MODEL
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

    for _ in range(5000):
        velocity = np.random.uniform(100, 900)
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


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Scenario Inputs")

threat_type = st.sidebar.selectbox(
    "Threat Type",
    ["HEAT / HESH", "FSAPDS / High-Velocity KE", "RPG / ATGM Type Threat"]
)

if threat_type == "HEAT / HESH":
    velocity = st.sidebar.slider("Threat Velocity (m/s)", 100, 400, 200)
elif threat_type == "FSAPDS / High-Velocity KE":
    velocity = st.sidebar.slider("Threat Velocity (m/s)", 400, 900, 600)
else:
    velocity = st.sidebar.slider("Threat Velocity (m/s)", 150, 500, 300)

distance = st.sidebar.slider("Initial Distance (m)", 20, 500, 180)
angle = st.sidebar.slider("Incoming Angle (degrees)", 0, 75, 25)

detection_range_m = st.sidebar.slider("Detection Range (m)", 100, 800, 350)
interception_range_m = st.sidebar.slider("Interception Range (m)", 20, 250, 120)

interceptor_velocity = st.sidebar.slider("Interceptor Velocity (m/s)", 300, 1200, 700)
sensor_delay_ms = st.sidebar.slider("Sensor + Processing Delay (ms)", 1, 30, 8)
animation_frames = st.sidebar.slider("Animation Smoothness", 25, 80, 45)

# Main timing calculations
time_to_impact = distance / velocity
time_to_detection = max((distance - detection_range_m) / velocity, 0)
time_to_interception_zone = max((distance - interception_range_m) / velocity, 0)

sensor_delay_s = sensor_delay_ms / 1000
interceptor_flight_time = interception_range_m / interceptor_velocity
interceptor_launch_time = max(time_to_interception_zone - interceptor_flight_time, 0)
interception_time = interceptor_launch_time + interceptor_flight_time
time_margin = time_to_impact - interception_time


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
# OUTPUT SUMMARY
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Threat Type", threat_type)

with c2:
    st.metric("Velocity", f"{velocity} m/s")

with c3:
    st.metric("Initial Distance", f"{distance} m")

with c4:
    st.metric("Time-to-Impact", f"{time_to_impact:.4f} s")

st.markdown("### AI Classified Threat Level")

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

with st.expander("Show AI Confidence Score"):
    st.bar_chart(confidence_df.set_index("Threat Level"))


# ============================================================
# 3D SHAPE HELPERS
# ============================================================

def cuboid(x0, x1, y0, y1, z0, z1, name="Part", color="olive", opacity=1.0):
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]

    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6, 6]
    j = [1, 2, 4, 2, 5, 3, 5, 6, 6, 1, 7, 2]
    k = [2, 3, 5, 5, 6, 7, 6, 7, 1, 0, 2, 3]

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        flatshading=True,
        hoverinfo="skip"
    )


def cylinder_x(x0, x1, y, z, radius=0.12, name="Cylinder", color="darkolivegreen", n=24):
    theta = np.linspace(0, 2 * np.pi, n)

    x, yy, zz = [], [], []
    for xi in [x0, x1]:
        x.extend([xi] * n)
        yy.extend(y + radius * np.cos(theta))
        zz.extend(z + radius * np.sin(theta))

    ii, jj, kk = [], [], []
    for a in range(n - 1):
        ii += [a, a + 1]
        jj += [a + 1, a + n + 1]
        kk += [a + n, a + n]

    return go.Mesh3d(
        x=x, y=yy, z=zz,
        i=ii, j=jj, k=kk,
        color=color,
        opacity=1.0,
        name=name,
        hoverinfo="skip"
    )


def cylinder_y(x, y0, y1, z, radius=0.12, name="Cylinder", color="darkolivegreen", n=24):
    theta = np.linspace(0, 2 * np.pi, n)

    xx, y, zz = [], [], []
    for yi in [y0, y1]:
        xx.extend(x + radius * np.cos(theta))
        y.extend([yi] * n)
        zz.extend(z + radius * np.sin(theta))

    ii, jj, kk = [], [], []
    for a in range(n - 1):
        ii += [a, a + 1]
        jj += [a + 1, a + n + 1]
        kk += [a + n, a + n]

    return go.Mesh3d(
        x=xx, y=y, z=zz,
        i=ii, j=jj, k=kk,
        color=color,
        opacity=1.0,
        name=name,
        hoverinfo="skip"
    )


def range_ring(radius, z, name, color, width=5):
    t = np.linspace(0, 2 * np.pi, 250)

    return go.Scatter3d(
        x=radius * np.cos(t),
        y=radius * np.sin(t),
        z=np.full_like(t, z),
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        hoverinfo="skip"
    )


def simple_dome(radius, name, color, opacity=0.08):
    # Low-resolution transparent upper dome, clean and non-cluttered
    phi = np.linspace(0, np.pi / 2, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=opacity,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name,
        hoverinfo="skip"
    )


def create_ground(size=18):
    x = np.linspace(-size, size, 2)
    y = np.linspace(-size, size, 2)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x) - 0.03

    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.5,
        colorscale=[[0, "rgb(190,175,145)"], [1, "rgb(190,175,145)"]],
        showscale=False,
        hoverinfo="skip"
    )


def create_mbt():
    parts = []

    # Hull and turret
    parts.append(cuboid(-4.5, 4.5, -2.0, 2.0, 0.35, 1.20, "Hull", "darkolivegreen"))
    parts.append(cuboid(-5.2, -4.2, -1.7, 1.7, 0.35, 0.95, "Front Armor", "olivedrab"))
    parts.append(cuboid(2.7, 4.7, -1.7, 1.7, 1.20, 1.50, "Engine Deck", "darkkhaki"))
    parts.append(cuboid(-1.8, 1.8, -1.25, 1.25, 1.20, 2.05, "Turret", "olive"))
    parts.append(cuboid(-1.0, 1.0, -0.8, 0.8, 2.05, 2.35, "Turret Top", "darkolivegreen"))

    # Main gun
    parts.append(cuboid(-2.3, -1.65, -0.5, 0.5, 1.50, 2.00, "Mantlet", "darkkhaki"))
    parts.append(cylinder_x(-2.3, -8.4, 0, 1.75, 0.14, "Main Gun", "darkolivegreen"))
    parts.append(cylinder_x(-8.4, -9.0, 0, 1.75, 0.09, "Muzzle", "black"))

    # Tracks
    parts.append(cuboid(-4.8, 4.7, -2.55, -2.05, 0.0, 0.65, "Left Track", "dimgray"))
    parts.append(cuboid(-4.8, 4.7, 2.05, 2.55, 0.0, 0.65, "Right Track", "dimgray"))

    # Wheels
    for xw in np.linspace(-3.8, 3.8, 7):
        parts.append(cylinder_y(xw, -2.65, -2.58, 0.32, 0.28, "Wheel", "gray"))
        parts.append(cylinder_y(xw, 2.58, 2.65, 0.32, 0.28, "Wheel", "gray"))

    # Armor blocks
    for xb in np.linspace(-3.6, 2.8, 7):
        parts.append(cuboid(xb, xb + 0.45, -2.12, -1.82, 1.18, 1.48, "Left Side Armor", "khaki"))
        parts.append(cuboid(xb, xb + 0.45, 1.82, 2.12, 1.18, 1.48, "Right Side Armor", "khaki"))

    # DMA modules, clean and visible
    dma_modules = [
        (-4.4, -2.35, 1.45),
        (-4.4, 2.35, 1.45),
        (0.0, -2.35, 1.55),
        (0.0, 2.35, 1.55),
        (4.2, -2.35, 1.45),
        (4.2, 2.35, 1.45),
        (0.1, 0.0, 2.58)
    ]

    for lx, ly, lz in dma_modules:
        parts.append(cylinder_y(lx, ly - 0.20, ly + 0.20, lz, 0.16, "DMA Launcher", "gold"))

    # Sensor marker only one label, no clutter
    parts.append(go.Scatter3d(
        x=[0],
        y=[0],
        z=[2.85],
        mode="markers+text",
        marker=dict(size=6, color="cyan"),
        text=["APS Sensor Suite"],
        textposition="top center",
        name="APS Sensor Suite",
        hoverinfo="skip"
    ))

    return parts, dma_modules


def create_missile_mesh(cx, cy, cz, length=1.2, radius=0.12, color="red", name="Incoming Missile"):
    # Missile aligned roughly along X axis for clean appearance
    body = cylinder_x(cx - length / 2, cx + length / 2, cy, cz, radius, name, color, n=20)

    nose = go.Cone(
        x=[cx + length / 2],
        y=[cy],
        z=[cz],
        u=[0.7],
        v=[0],
        w=[0],
        sizemode="absolute",
        sizeref=0.35,
        anchor="tip",
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name,
        hoverinfo="skip"
    )

    fins = [
        cuboid(cx - length / 2 - 0.05, cx - length / 2 + 0.20, cy - 0.35, cy - 0.25, cz - 0.04, cz + 0.04, "Fin", color),
        cuboid(cx - length / 2 - 0.05, cx - length / 2 + 0.20, cy + 0.25, cy + 0.35, cz - 0.04, cz + 0.04, "Fin", color),
        cuboid(cx - length / 2 - 0.05, cx - length / 2 + 0.20, cy - 0.04, cy + 0.04, cz + 0.25, cz + 0.35, "Fin", color),
    ]

    return [body, nose] + fins


def create_interceptor_marker(x, y, z):
    return go.Cone(
        x=[x],
        y=[y],
        z=[z],
        u=[-0.7],
        v=[0.25],
        w=[0.05],
        sizemode="absolute",
        sizeref=0.45,
        anchor="tip",
        colorscale=[[0, "lime"], [1, "lime"]],
        showscale=False,
        name="Interceptor",
        hoverinfo="skip"
    )


# ============================================================
# CLEAN 3D ANIMATION
# ============================================================

st.markdown("## Clean 3D Detection and Interception Animation")

# Scale metres into plot units
scale = 25.0
det_radius = detection_range_m / scale
int_radius = interception_range_m / scale

# Start missile from left/front, like reference image
theta = np.radians(angle)
start = np.array([
    -distance / scale,
    -np.tan(theta) * (distance / scale) * 0.45,
    2.2 + (angle / 75) * 3.0
])

tank_center = np.array([0.0, 0.0, 1.45])
direction = tank_center - start
direction = direction / np.linalg.norm(direction)

# Point where threat is neutralized on interception ring
intercept_point = tank_center - direction * int_radius
end = tank_center

missile_path = np.linspace(start, end, animation_frames)

# DMA launcher selected nearest to intercept path
_, dma_modules = create_mbt()
dma_array = np.array(dma_modules)
closest_idx = np.argmin(np.linalg.norm(dma_array - intercept_point, axis=1))
launcher_point = dma_array[closest_idx]

# Interceptor path from selected launcher to intercept point
interceptor_path = np.linspace(launcher_point, intercept_point, animation_frames)

# Determine intercept frame
distance_to_intercept = np.linalg.norm(missile_path - intercept_point, axis=1)
intercept_idx = int(np.argmin(distance_to_intercept))

# Base traces
ground = create_ground(size=max(18, det_radius + 3))
detection_dome = simple_dome(det_radius, "Detection Dome", "lightblue", 0.06)
interception_dome = simple_dome(int_radius, "Interception Dome", "lightgreen", 0.10)
detection_ring = range_ring(det_radius, 0.04, "Detection Range", "cyan", 6)
interception_ring = range_ring(int_radius, 0.08, "Interception Range", "lime", 7)

tank_parts, dma_modules = create_mbt()

threat_line = go.Scatter3d(
    x=missile_path[:, 0],
    y=missile_path[:, 1],
    z=missile_path[:, 2],
    mode="lines",
    line=dict(color="red", width=6),
    name="Incoming Threat Path",
    hoverinfo="skip"
)

interceptor_line = go.Scatter3d(
    x=[launcher_point[0], intercept_point[0]],
    y=[launcher_point[1], intercept_point[1]],
    z=[launcher_point[2], intercept_point[2]],
    mode="lines",
    line=dict(color="lime", width=7, dash="dash"),
    name="Interceptor Path",
    hoverinfo="skip"
)

intercept_marker = go.Scatter3d(
    x=[intercept_point[0]],
    y=[intercept_point[1]],
    z=[intercept_point[2]],
    mode="markers+text",
    marker=dict(size=7, color="lime"),
    text=["Neutralization Point"],
    textposition="top center",
    name="Neutralization Point",
    hoverinfo="skip"
)

selected_launcher_marker = go.Scatter3d(
    x=[launcher_point[0]],
    y=[launcher_point[1]],
    z=[launcher_point[2]],
    mode="markers+text",
    marker=dict(size=8, color="yellow"),
    text=["Selected DMA Launcher"],
    textposition="top center",
    name="Selected DMA Launcher",
    hoverinfo="skip"
)

initial_missile = create_missile_mesh(
    missile_path[0, 0],
    missile_path[0, 1],
    missile_path[0, 2],
    length=1.1,
    radius=0.11,
    color="red",
    name="Incoming Missile"
)

initial_interceptor = create_interceptor_marker(
    launcher_point[0],
    launcher_point[1],
    launcher_point[2]
)

base_data = [
    ground,
    detection_dome,
    interception_dome,
    detection_ring,
    interception_ring,
] + tank_parts + [
    threat_line,
    interceptor_line,
    intercept_marker,
    selected_launcher_marker,
] + initial_missile + [
    initial_interceptor
]

frames = []
for i in range(animation_frames):
    missile_position = missile_path[min(i, intercept_idx)]

    if i < intercept_idx:
        interceptor_progress_idx = min(i, animation_frames - 1)
        interceptor_position = interceptor_path[interceptor_progress_idx]
        missile_color = "red"
    else:
        interceptor_position = intercept_point
        missile_position = intercept_point
        missile_color = "orange"

    frame_missile = create_missile_mesh(
        missile_position[0],
        missile_position[1],
        missile_position[2],
        length=1.1,
        radius=0.11,
        color=missile_color,
        name="Incoming Missile"
    )

    frame_interceptor = create_interceptor_marker(
        interceptor_position[0],
        interceptor_position[1],
        interceptor_position[2]
    )

    frames.append(
        go.Frame(
            data=frame_missile + [frame_interceptor],
            traces=list(range(len(base_data) - 4, len(base_data))),
            name=str(i)
        )
    )

fig = go.Figure(data=base_data, frames=frames)

fig.update_layout(
    title="APS Threat Detection → DMA Launcher Selection → Interception",
    height=760,
    showlegend=False,
    margin=dict(l=0, r=0, t=50, b=0),
    scene=dict(
        xaxis=dict(
            title="Forward Direction",
            range=[-18, 10],
            showbackground=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Lateral Direction",
            range=[-10, 10],
            showbackground=False,
            showgrid=False,
            zeroline=False
        ),
        zaxis=dict(
            title="Height",
            range=[0, 9],
            showbackground=False,
            showgrid=False,
            zeroline=False
        ),
        aspectmode="manual",
        aspectratio=dict(x=1.7, y=1.0, z=0.55),
        camera=dict(
            eye=dict(x=-1.85, y=-1.35, z=0.85),
            center=dict(x=-0.15, y=0.0, z=-0.15)
        )
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.02,
            y=0.98,
            buttons=[
                dict(
                    label="▶ Play Animation",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=100, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
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

st.success(
    "Clean view enabled: cyan ring = detection range, green ring = interception range, "
    "yellow marker = selected DMA launcher, red missile = incoming threat, green interceptor = countermeasure."
)


# ============================================================
# REQUIRED OUTPUT TABLE
# ============================================================

st.markdown("## Final Engagement Output")

output_table = pd.DataFrame({
    "Parameter": [
        "Threat Type",
        "Velocity",
        "Initial Distance",
        "Incoming Angle",
        "Detection Range",
        "Interception Range",
        "Estimated Time-to-Impact",
        "Detection Time",
        "Interceptor Launch Time",
        "Interception / Neutralization Time",
        "Interceptor Flight Time",
        "Time Margin After Interception",
        "AI Classified Threat Level"
    ],
    "Value": [
        threat_type,
        f"{velocity} m/s",
        f"{distance} m",
        f"{angle} degrees",
        f"{detection_range_m} m",
        f"{interception_range_m} m",
        f"{time_to_impact:.4f} seconds",
        f"{time_to_detection:.4f} seconds",
        f"{interceptor_launch_time:.4f} seconds",
        f"{interception_time:.4f} seconds",
        f"{interceptor_flight_time:.4f} seconds",
        f"{time_margin:.4f} seconds",
        labels[prediction]
    ]
})

st.table(output_table)

st.markdown("## Thesis Explanation")

st.write("""
This module provides a clean 3D APS engagement visualization. The incoming missile is represented
as a simple 3D projectile model approaching the Main Battle Tank from the selected angle. The cyan
ring indicates the detection range, while the green ring indicates the interception range. The DMA
architecture is represented through distributed launcher modules placed around the vehicle. Based on
the threat direction, a suitable DMA launcher is selected, and an interceptor trajectory is shown from
that module to the predicted neutralization point.

The AI model classifies the threat using velocity, distance, and incoming angle. The output also
calculates estimated time-to-impact, detection time, interceptor launch time, interceptor flight time,
and neutralization time. This provides a system-level decision-support visualization for APS studies
and complements the ETC launcher simulation work.
""")
