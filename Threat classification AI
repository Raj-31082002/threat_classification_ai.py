# ============================================================
# THREAT CLASSIFICATION AI MODEL
# For Active Protection System Decision Support
# Model: Neural Network Classifier
# Inputs: Velocity, Distance, Angle
# Output: LOW / MEDIUM / HIGH Threat
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------------------------------------------
# 1. RULE-BASED LABEL GENERATION
# ------------------------------------------------------------

def classify_threat_rule(velocity, distance, angle):
    """
    Creates threat labels based on simplified APS logic.

    velocity : incoming threat velocity in m/s
    distance : distance from vehicle in m
    angle    : approach angle in degrees

    Output:
    0 = LOW threat
    1 = MEDIUM threat
    2 = HIGH threat
    """

    # Time-to-impact approximation
    if velocity <= 0:
        tti = 999
    else:
        tti = distance / velocity

    # Normalize angle severity
    # Direct/frontal or steep top-attack angles may be more critical
    angle_risk = 0

    if angle >= 60:
        angle_risk = 2
    elif angle >= 30:
        angle_risk = 1
    else:
        angle_risk = 0

    # Threat classification logic
    if velocity > 1200 or tti < 0.05 or angle_risk == 2:
        return 2  # HIGH

    elif velocity > 500 or tti < 0.15 or angle_risk == 1:
        return 1  # MEDIUM

    else:
        return 0  # LOW


# ------------------------------------------------------------
# 2. SYNTHETIC DATASET GENERATION
# ------------------------------------------------------------

def generate_dataset(samples=2000):
    """
    Generates synthetic APS threat data.
    This is used because real threat data is restricted/unavailable.
    """

    data = []

    for _ in range(samples):
        velocity = np.random.uniform(100, 1800)     # m/s
        distance = np.random.uniform(20, 500)       # meters
        angle = np.random.uniform(0, 90)            # degrees

        threat_class = classify_threat_rule(velocity, distance, angle)

        data.append([velocity, distance, angle, threat_class])

    df = pd.DataFrame(
        data,
        columns=["Velocity_mps", "Distance_m", "Angle_deg", "Threat_Class"]
    )

    return df


# ------------------------------------------------------------
# 3. TRAIN NEURAL NETWORK MODEL
# ------------------------------------------------------------

def train_model(df):
    """
    Trains a Neural Network classifier.
    """

    X = df[["Velocity_mps", "Distance_m", "Angle_deg"]]
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
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("\n================ MODEL PERFORMANCE ================\n")
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["LOW", "MEDIUM", "HIGH"]
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler


# ------------------------------------------------------------
# 4. PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_threat(model, scaler, velocity, distance, angle):
    """
    Predicts threat level using trained neural network.
    """

    input_data = pd.DataFrame(
        [[velocity, distance, angle]],
        columns=["Velocity_mps", "Distance_m", "Angle_deg"]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    threat_names = {
        0: "LOW THREAT",
        1: "MEDIUM THREAT",
        2: "HIGH THREAT"
    }

    print("\n================ THREAT CLASSIFICATION RESULT ================\n")
    print(f"Input Velocity : {velocity:.2f} m/s")
    print(f"Input Distance : {distance:.2f} m")
    print(f"Input Angle    : {angle:.2f} degrees")

    print("\nPredicted Threat Level:", threat_names[prediction])

    print("\nPrediction Confidence:")
    print(f"LOW    : {probability[0] * 100:.2f}%")
    print(f"MEDIUM : {probability[1] * 100:.2f}%")
    print(f"HIGH   : {probability[2] * 100:.2f}%")

    return prediction, probability


# ------------------------------------------------------------
# 5. GRAPH GENERATION FOR THESIS
# ------------------------------------------------------------

def plot_results(df):
    """
    Creates plots that can be used in thesis/PPT.
    """

    threat_labels = {
        0: "LOW",
        1: "MEDIUM",
        2: "HIGH"
    }

    df["Threat_Label"] = df["Threat_Class"].map(threat_labels)

    # Plot 1: Threat class distribution
    plt.figure(figsize=(7, 5))
    df["Threat_Label"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).plot(kind="bar")
    plt.xlabel("Threat Class")
    plt.ylabel("Number of Samples")
    plt.title("Threat Class Distribution")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("threat_class_distribution.png", dpi=300)
    plt.show()

    # Plot 2: Velocity vs Distance
    plt.figure(figsize=(8, 5))
    for class_id, label in threat_labels.items():
        subset = df[df["Threat_Class"] == class_id]
        plt.scatter(
            subset["Distance_m"],
            subset["Velocity_mps"],
            label=label,
            alpha=0.5
        )

    plt.xlabel("Distance (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Threat Classification Based on Velocity and Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("velocity_distance_threat_map.png", dpi=300)
    plt.show()

    # Plot 3: Angle vs Velocity
    plt.figure(figsize=(8, 5))
    for class_id, label in threat_labels.items():
        subset = df[df["Threat_Class"] == class_id]
        plt.scatter(
            subset["Angle_deg"],
            subset["Velocity_mps"],
            label=label,
            alpha=0.5
        )

    plt.xlabel("Angle of Approach (deg)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Threat Classification Based on Angle and Velocity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("angle_velocity_threat_map.png", dpi=300)
    plt.show()


# ------------------------------------------------------------
# 6. MAIN PROGRAM
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\nGenerating synthetic APS threat dataset...")
    df = generate_dataset(samples=3000)

    print("\nFirst five rows of dataset:")
    print(df.head())

    df.to_csv("aps_threat_classification_dataset.csv", index=False)
    print("\nDataset saved as: aps_threat_classification_dataset.csv")

    model, scaler = train_model(df)

    plot_results(df)

    # --------------------------------------------------------
    # Example Prediction
    # Change these values during presentation/demo
    # --------------------------------------------------------

    example_velocity = 1400   # m/s
    example_distance = 80     # m
    example_angle = 65        # degrees

    predict_threat(
        model,
        scaler,
        example_velocity,
        example_distance,
        example_angle
    )
