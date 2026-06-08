# ============================================================
# FINAL EKF SENSOR FUSION MODULE
# ============================================================
#
# PURPOSE:
# Generate ONE unified EKF fused dataset
# for BOTH:
#
# 1. Random Forest Geofencing
# 2. Isolation Forest Anomaly Detection
#
# ============================================================
#
# OUTPUT FEATURES:
#
# time
# fused_x
# fused_y
# fused_vx
# fused_vy
# fused_speed
# acceleration_magnitude
# movement_direction
#
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

DATASET_FOLDER = "datasets"

DATASETS = [
    "standardized_dataset_1.csv",
    "standardized_dataset_2.csv",
    "standardized_dataset_3.csv"
]

OUTPUT_FOLDER = "results"

Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

EARTH_RADIUS = 6378137

# ============================================================
# EKF STABILITY LIMITS  (prevent velocity blowup on sparse GPS)
# ============================================================
#
# When GPS packets are far apart (e.g. 13-60 s on LoRaWAN),
# integrating a single accelerometer sample over the whole gap
# makes velocity explode. These two limits keep the filter stable:
#
#   MAX_DT     : cap the integration window — a single accel reading
#                is only valid for a short interval, not a 60 s gap.
#   MAX_SPEED  : hard ceiling on velocity magnitude (m/s).
#                ~15 m/s = 54 km/h — above any child on foot, still
#                lets genuine vehicle-speed anomalies through.

MAX_DT    = 2.0     # seconds — cap the accel integration window
MAX_SPEED = 15.0    # m/s — physical velocity ceiling

# ============================================================
# GPS TO LOCAL XY CONVERSION
# ============================================================

def gps_to_local_xy(lat, lon):

    lat0 = np.radians(lat.iloc[0])
    lon0 = np.radians(lon.iloc[0])

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = (lon_rad - lon0) * EARTH_RADIUS * np.cos(lat0)
    y = (lat_rad - lat0) * EARTH_RADIUS

    return x, y

# ============================================================
# EKF MEASUREMENT FUNCTION
# ============================================================

def Hx(x_state):

    return np.array([
        x_state[0],
        x_state[1]
    ])

# ============================================================
# EKF JACOBIAN
# ============================================================

def HJacobian(x_state):

    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

# ============================================================
# MAIN EKF FUNCTION
# ============================================================

def run_ekf(dataset_path):

    print("\n================================================")
    print(f"PROCESSING: {dataset_path}")
    print("================================================")

    # ========================================================
    # LOAD DATASET
    # ========================================================

    df = pd.read_csv(dataset_path)

    # ========================================================
    # HANDLE TIME
    # ========================================================

    try:

        df['time'] = pd.to_datetime(df['time'])

        df['time_sec'] = (
            df['time'] - df['time'].iloc[0]
        ).dt.total_seconds()

    except:

        df['time_sec'] = df['time']

    # ========================================================
    # REMOVE INVALID GPS
    # ========================================================

    df = df[
        (df['latitude'] != 0) &
        (df['longitude'] != 0)
    ]

    df = df.dropna()

    df = df.reset_index(drop=True)

    if len(df) < 10:

        print("Dataset too small.")
        return

    # ========================================================
    # CONVERT GPS TO LOCAL XY
    # ========================================================

    raw_x, raw_y = gps_to_local_xy(
        df['latitude'],
        df['longitude']
    )

    # ========================================================
    # EKF INITIALIZATION
    # ========================================================

    ekf = ExtendedKalmanFilter(
        dim_x=4,
        dim_z=2
    )

    # State vector:
    #
    # x = [x_position,
    #      y_position,
    #      x_velocity,
    #      y_velocity]

    ekf.x = np.array([
        raw_x.iloc[0],
        raw_y.iloc[0],
        0,
        0
    ])

    # Initial covariance

    ekf.P *= 10

    # Measurement noise
    # Higher = trust GPS less

    ekf.R = np.array([
        [5, 0],
        [0, 5]
    ])

    # Process noise
    # Higher = trust IMU less

    ekf.Q = np.eye(4) * 0.1

    # ========================================================
    # STORAGE
    # ========================================================

    timestamps = []

    fused_x = []
    fused_y = []

    fused_vx = []
    fused_vy = []

    fused_speed = []

    acceleration_magnitude = []

    movement_direction = []

    # ========================================================
    # EKF LOOP
    # ========================================================

    prev_time = df['time_sec'].iloc[0]

    for i in range(1, len(df)):

        current_time = df['time_sec'].iloc[i]

        dt = current_time - prev_time

        if dt <= 0:
            dt = 0.01

        # Cap the integration window. A single accel sample is not
        # representative of a long GPS gap, so integrating it over the
        # full dt (e.g. 60 s) makes velocity explode. Limit it.
        if dt > MAX_DT:
            dt = MAX_DT

        prev_time = current_time

        # ====================================================
        # IMU INPUT
        # ====================================================

        ax = df['ax'].iloc[i]
        ay = df['ay'].iloc[i]

        # ====================================================
        # ACCELERATION MAGNITUDE
        # ====================================================

        accel_mag = np.sqrt(
            ax**2 + ay**2
        )

        # ====================================================
        # STATE TRANSITION MATRIX
        # ====================================================

        ekf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # ====================================================
        # CONTROL INPUT MATRIX
        # ====================================================

        B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])

        u = np.array([ax, ay])

        # ====================================================
        # PREDICTION STEP
        # ====================================================

        ekf.x = ekf.F @ ekf.x + B @ u

        ekf.P = (
            ekf.F @ ekf.P @ ekf.F.T
            + ekf.Q
        )

        # ====================================================
        # GPS MEASUREMENT
        # ====================================================

        z = np.array([
            raw_x.iloc[i],
            raw_y.iloc[i]
        ])

        # ====================================================
        # UPDATE STEP
        # ====================================================

        ekf.update(
            z,
            HJacobian,
            Hx
        )

        # ====================================================
        # VELOCITY CLAMP  (hard safety net vs divergence)
        # ====================================================

        speed_now = np.sqrt(ekf.x[2]**2 + ekf.x[3]**2)
        if speed_now > MAX_SPEED:
            scale = MAX_SPEED / speed_now
            ekf.x[2] *= scale
            ekf.x[3] *= scale

        # ====================================================
        # EXTRACT EKF STATES
        # ====================================================

        x_pos = ekf.x[0]
        y_pos = ekf.x[1]

        vx = ekf.x[2]
        vy = ekf.x[3]

        # ====================================================
        # CALCULATE SPEED
        # ====================================================

        speed = np.sqrt(
            vx**2 + vy**2
        )

        # ====================================================
        # MOVEMENT DIRECTION
        # ====================================================

        direction = np.degrees(
            np.arctan2(vy, vx)
        )

        # ====================================================
        # STORE RESULTS
        # ====================================================

        timestamps.append(df['time'].iloc[i])

        fused_x.append(x_pos)
        fused_y.append(y_pos)

        fused_vx.append(vx)
        fused_vy.append(vy)

        fused_speed.append(speed)

        acceleration_magnitude.append(accel_mag)

        movement_direction.append(direction)

    # ========================================================
    # CREATE FINAL FUSED DATASET
    # ========================================================

    fused_dataset = pd.DataFrame({

        "time": timestamps,

        "fused_x": fused_x,
        "fused_y": fused_y,

        "fused_vx": fused_vx,
        "fused_vy": fused_vy,

        "fused_speed": fused_speed,

        "acceleration_magnitude":
            acceleration_magnitude,

        "movement_direction":
            movement_direction

    })

    # ========================================================
    # SAVE DATASET
    # ========================================================

    dataset_name = Path(dataset_path).stem

    output_csv = (
        f"{OUTPUT_FOLDER}/"
        f"{dataset_name}_EKF_FUSED.csv"
    )

    fused_dataset.to_csv(
        output_csv,
        index=False
    )

    print("\nFUSED DATASET SAVED:")
    print(output_csv)

    # ========================================================
    # TRAJECTORY PLOT
    # ========================================================

    plt.figure(figsize=(10, 8))

    # RAW GPS
    plt.plot(
        raw_x,
        raw_y,
        linestyle='dotted',
        linewidth=2,
        label='Raw GPS'
    )

    # EKF FUSION
    plt.plot(
        fused_x,
        fused_y,
        linewidth=2,
        label='EKF Fusion'
    )

    plt.title(
        f"EKF Sensor Fusion - {dataset_name}"
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")

    plt.legend()

    plt.grid(True)

    plt.axis("equal")

    plot_path = (
        f"{OUTPUT_FOLDER}/"
        f"{dataset_name}_trajectory.png"
    )

    plt.savefig(plot_path)

    print("\nTRAJECTORY PLOT SAVED:")
    print(plot_path)

    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for dataset in DATASETS:

        dataset_path = (
            f"{DATASET_FOLDER}/{dataset}"
        )

        if Path(dataset_path).exists():

            run_ekf(dataset_path)

        else:

            print(
                f"Dataset not found: {dataset_path}"
            )

print("\nALL DATASETS COMPLETED.")