"""
anomaly_service.py
==================
Two-step sensor-fusion + anomaly-detection pipeline.

  STEP 1 — SENSOR FUSION
    Reads raw sensor data from:   ttn_uplinks/{deviceId}/events
    Computes 7 derived features:  fused_x, fused_y, fused_vx, fused_vy,
                                  fused_speed, acceleration_magnitude, movement_direction
    Writes fused vectors to:      fusion_data/{deviceId}/events

  STEP 2 — ISOLATION FOREST
    Reads fused data from:        fusion_data/{deviceId}/events
    Runs IsolationForest model
    Writes results to:            anomaly_results/{deviceId}

Firestore collections:
  ttn_uplinks/{deviceId}/events    raw IoT uploads  (read-only by this service)
  fusion_data/{deviceId}/events    fused features   (written by STEP 1)
  anomaly_results/{deviceId}       model output     (written by STEP 2)

Flask API — http://localhost:5000
  GET  /health
  POST /fuse/<device_id>      STEP 1 only  — raw → fusion_data
  POST /fuse/all              STEP 1 for all devices
  POST /predict/<device_id>   STEP 2 only  — fusion_data → anomaly_results
  POST /process/<device_id>   STEP 1 + 2  combined
  POST /process/all           STEP 1 + 2  for all devices
  GET  /fusion/<device_id>    inspect fusion_data events
  GET  /results/<device_id>   inspect anomaly_results

Requirements:
  pip install flask flask-cors scikit-learn joblib numpy firebase-admin filterpy

Usage:
  python anomaly_service.py
"""

import os
import math
import threading
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore

try:
    from filterpy.kalman import ExtendedKalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    print("[anomaly_service] WARNING: filterpy not installed — "
          "run: pip install filterpy")

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "isolation_forest_model.pkl")
SERVICE_ACCOUNT_PATH = os.path.join(
    os.path.dirname(__file__),
    "fulcrum-86a7b-firebase-adminsdk-fbsvc-d66d28891c.json",
)
PORT = 5000

RAW_COLLECTION    = "ttn_uplinks"     # IoT raw uploads
FUSION_COLLECTION = "fusion_data"     # fused feature vectors (written by Step 1)
RESULT_COLLECTION = "anomaly_results" # model predictions    (written by Step 2)

DEVICE_IDS = [
    "dtestt1", "dtestt2", "node3", "node4", "node5",
    "node6",   "node7",   "node8", "node9", "node10",
]

# ── Load model ────────────────────────────────────────────────────────────────

print(f"[anomaly_service] Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"[anomaly_service] Model loaded — "
      f"n_features={model.n_features_in_}, n_estimators={model.n_estimators}")

# ── Firebase init ─────────────────────────────────────────────────────────────

if os.path.exists(SERVICE_ACCOUNT_PATH):
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    FIREBASE_AVAILABLE = True
    print("[anomaly_service] Firebase initialised.")
else:
    FIREBASE_AVAILABLE = False
    db = None
    print(f"[anomaly_service] WARNING: {SERVICE_ACCOUNT_PATH} not found — "
          "Firestore disabled. REST API still works for direct inference.")

# ── Geometry helpers ──────────────────────────────────────────────────────────

_EARTH_R = 6_371_000.0  # metres

# ── EKF stability limits (MUST match sensorfuison.py) ─────────────────────────
# Prevent velocity blowup on sparse GPS (13-60 s gaps on LoRaWAN).
#   MAX_DT    : cap the accel integration window (a single sample is not
#               valid over a long gap).
#   MAX_SPEED : hard ceiling on velocity magnitude (m/s). ~15 m/s = 54 km/h.
MAX_DT    = 2.0     # seconds
MAX_SPEED = 15.0    # m/s

# Per-device EKF state for the real-time listener.
# Each entry: {"ekf": EKF, "ref_lat": float, "ref_lon": float, "prev_time": float|None}
_device_ekf: dict = {}


# ── EKF helpers (from sensorfuison.py) ───────────────────────────────────────

def _init_ekf() -> "ExtendedKalmanFilter":
    """Create a fresh EKF with the same noise tuning as sensorfuison.py."""
    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
    # State: [x_pos, y_pos, x_vel, y_vel]
    ekf.x = np.zeros(4)
    ekf.P = np.eye(4) * 10     # initial covariance
    ekf.R = np.array([[5.0, 0.0], [0.0, 5.0]])   # GPS measurement noise
    ekf.Q = np.eye(4) * 0.1   # process noise
    return ekf


def _Hx(x_state: np.ndarray) -> np.ndarray:
    """Measurement function — observe only position."""
    return np.array([x_state[0], x_state[1]])


def _HJacobian(x_state: np.ndarray) -> np.ndarray:
    """Jacobian of measurement function."""
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0]])


def _gps_to_xy(lat: float, lon: float,
               ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Equirectangular GPS → local XY metres (same as sensorfuison.py)."""
    x = math.radians(lon - ref_lon) * _EARTH_R * math.cos(math.radians(ref_lat))
    y = math.radians(lat - ref_lat) * _EARTH_R
    return x, y


def _xy_to_gps(x: float, y: float,
               ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Local XY metres → GPS lat/lon."""
    lat = ref_lat + math.degrees(y / _EARTH_R)
    lon = ref_lon + math.degrees(x / (_EARTH_R * math.cos(math.radians(ref_lat))))
    return lat, lon


def _ekf_step(ekf: "ExtendedKalmanFilter",
              ax: float, ay: float,
              gps_x: float, gps_y: float,
              dt: float) -> None:
    """One EKF predict + update cycle (in-place)."""
    # Use the REAL dt for position prediction (F) so velocity is inferred
    # correctly from GPS movement over the true elapsed time.
    # Cap dt only for the ACCEL integration (B): a single accel sample is not
    # valid over a long GPS gap (e.g. 60 s) and would otherwise blow up velocity.
    dt_b = min(dt, MAX_DT)

    # State transition (constant-velocity model) — real dt
    ekf.F = np.array([[1, 0, dt, 0],
                      [0, 1, 0,  dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=float)
    # Control input (accel) — capped dt
    B = np.array([[0.5 * dt_b**2, 0],
                  [0, 0.5 * dt_b**2],
                  [dt_b, 0],
                  [0,  dt_b]], dtype=float)
    # Predict
    ekf.x = ekf.F @ ekf.x + B @ np.array([ax, ay])
    ekf.P = ekf.F @ ekf.P @ ekf.F.T + ekf.Q
    # Update with GPS
    ekf.update(np.array([gps_x, gps_y]), _HJacobian, _Hx)

    # Velocity clamp — hard safety net against divergence.
    speed_now = math.sqrt(ekf.x[2]**2 + ekf.x[3]**2)
    if speed_now > MAX_SPEED:
        scale = MAX_SPEED / speed_now
        ekf.x[2] *= scale
        ekf.x[3] *= scale


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SENSOR FUSION  (EKF — from sensorfuison.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_raw(ev: dict) -> tuple:
    """Pull raw sensor values from a Firestore event dict."""
    decoded = ev.get("decoded") or ev
    lat = float(decoded.get("lat") or decoded.get("latitude") or 0.0)
    lon = float(decoded.get("lon") or decoded.get("longitude") or 0.0)
    ax  = float(decoded.get("ax")  or decoded.get("accel_x")  or 0.0)
    ay  = float(decoded.get("ay")  or decoded.get("accel_y")  or 0.0)
    az  = float(decoded.get("az")  or decoded.get("accel_z")  or 9.8)
    server_time = ev.get("serverTime")
    try:
        t = server_time.timestamp() if hasattr(server_time, "timestamp") else None
    except Exception:
        t = None
    return lat, lon, ax, ay, az, server_time, t


def _build_feature_row(ekf: "ExtendedKalmanFilter",
                       ax: float, ay: float, az: float,
                       ref_lat: float, ref_lon: float,
                       server_time, doc_id: str) -> dict:
    """Convert EKF state + raw IMU into the 7-feature dict written to Firestore.

    fused_x / fused_y are kept as local XY metres (same as sensorfuison.py
    training output) so that the Isolation Forest model receives the same
    feature scale it was trained on.
    lat / lon are stored separately for Flutter display.
    """
    fused_x_m = float(ekf.x[0])   # metres from EKF origin
    fused_y_m = float(ekf.x[1])   # metres from EKF origin
    fused_vx  = float(ekf.x[2])
    fused_vy  = float(ekf.x[3])
    fused_speed = math.sqrt(fused_vx**2 + fused_vy**2)

    # Convert EKF XY back to lat/lon for Flutter display only
    fused_lat, fused_lon = _xy_to_gps(fused_x_m, fused_y_m, ref_lat, ref_lon)

    acc_mag = math.sqrt(ax**2 + ay**2 + az**2)
    movement_direction = (math.degrees(math.atan2(fused_vx, fused_vy)) + 360.0) % 360.0

    return {
        # 7 model features — fused_x/fused_y in METRES (matches training data)
        "fused_x":               fused_x_m,
        "fused_y":               fused_y_m,
        "fused_vx":              fused_vx,
        "fused_vy":              fused_vy,
        "fused_speed":           fused_speed,
        "acceleration_magnitude": acc_mag,
        "movement_direction":    movement_direction,
        # flutter-compatible aliases
        "acc_mag":   acc_mag,
        "acc_delta": 0.0,
        "speed":     fused_speed,
        # lat/lon kept separately for Flutter map display
        "ax": ax, "ay": ay, "az": az,
        "lat": fused_lat,
        "lon": fused_lon,
        "doc_id":      doc_id,
        "server_time": server_time,
    }


def compute_features(events: list[dict]) -> list[dict]:
    """
    BATCH EKF sensor fusion (mirrors sensorfuison.py).

    Initialises a fresh EKF from the first valid GPS fix, then for every
    subsequent event:
      - Predict: uses ax/ay as control input (constant-velocity model)
      - Update:  corrects with the GPS measurement
    Returns one feature row per event (first event is skipped — EKF needs
    at least one prior step to produce meaningful velocity).
    """
    if not FILTERPY_AVAILABLE:
        raise RuntimeError("filterpy is required — pip install filterpy")

    # Filter events that have a valid GPS fix
    valid = []
    for ev in events:
        lat, lon, ax, ay, az, server_time, t = _extract_raw(ev)
        if lat != 0.0 or lon != 0.0:
            valid.append((ev, lat, lon, ax, ay, az, server_time, t))

    if len(valid) < 2:
        return []

    # Reference origin = first GPS fix
    _, ref_lat, ref_lon, _, _, _, _, _ = valid[0]
    prev_t = valid[0][7]

    # Initialise EKF at origin
    ekf = _init_ekf()
    x0, y0 = _gps_to_xy(ref_lat, ref_lon, ref_lat, ref_lon)  # (0, 0)
    ekf.x = np.array([x0, y0, 0.0, 0.0])

    results = []
    for ev, lat, lon, ax, ay, az, server_time, curr_t in valid[1:]:
        dt = (curr_t - prev_t) if (curr_t and prev_t and curr_t > prev_t) else 1.0
        prev_t = curr_t

        gps_x, gps_y = _gps_to_xy(lat, lon, ref_lat, ref_lon)
        _ekf_step(ekf, ax, ay, gps_x, gps_y, dt)

        results.append(_build_feature_row(
            ekf, ax, ay, az, ref_lat, ref_lon,
            server_time, ev.get("_doc_id", ""),
        ))

    return results


def compute_single_ekf(device_id: str, event: dict) -> dict | None:
    """
    REAL-TIME EKF sensor fusion — one event at a time.

    Maintains per-device EKF state in _device_ekf so velocity and position
    are properly estimated across consecutive live events.
    Returns None on the first event for a device (EKF just initialised,
    no velocity estimate yet).
    """
    if not FILTERPY_AVAILABLE:
        raise RuntimeError("filterpy is required — pip install filterpy")

    lat, lon, ax, ay, az, server_time, curr_t = _extract_raw(event)
    if lat == 0.0 and lon == 0.0:
        return None

    state = _device_ekf.get(device_id)

    if state is None:
        # First event — initialise EKF, no output yet
        ekf = _init_ekf()
        ekf.x = np.array([0.0, 0.0, 0.0, 0.0])
        _device_ekf[device_id] = {
            "ekf": ekf,
            "ref_lat": lat,
            "ref_lon": lon,
            "prev_t":  curr_t,
        }
        return None

    ekf     = state["ekf"]
    ref_lat = state["ref_lat"]
    ref_lon = state["ref_lon"]
    prev_t  = state["prev_t"]

    dt = (curr_t - prev_t) if (curr_t and prev_t and curr_t > prev_t) else 1.0
    state["prev_t"] = curr_t

    gps_x, gps_y = _gps_to_xy(lat, lon, ref_lat, ref_lon)
    _ekf_step(ekf, ax, ay, gps_x, gps_y, dt)

    return _build_feature_row(
        ekf, ax, ay, az, ref_lat, ref_lon,
        server_time, event.get("_doc_id", ""),
    )


def _fusion_doc(device_id: str, row: dict) -> dict:
    """Build the Firestore document written into fusion_data/{deviceId}/events."""
    return {
        "deviceId":              device_id,
        "sourceDocId":           row.get("doc_id", ""),
        "serverTime":            row.get("server_time") or firestore.SERVER_TIMESTAMP,
        "fusedAt":               firestore.SERVER_TIMESTAMP,
        # 7 model features
        "fused_x":               row["fused_x"],
        "fused_y":               row["fused_y"],
        "fused_vx":              row["fused_vx"],
        "fused_vy":              row["fused_vy"],
        "fused_speed":           row["fused_speed"],
        "acceleration_magnitude": row["acceleration_magnitude"],
        "movement_direction":    row["movement_direction"],
        # raw values (kept for traceability)
        "lat": row["lat"], "lon": row["lon"],
        "ax":  row["ax"],  "ay":  row["ay"],  "az": row["az"],
    }


def write_fusion_event(device_id: str, row: dict) -> None:
    """Write a single fused event to fusion_data/{deviceId}/events."""
    (db.collection(FUSION_COLLECTION)
       .document(device_id)
       .collection("events")
       .add(_fusion_doc(device_id, row)))


def write_fusion_batch(device_id: str, rows: list[dict]) -> None:
    """Batch-write fused events to fusion_data/{deviceId}/events (chunked ≤ 500)."""
    col   = (db.collection(FUSION_COLLECTION)
               .document(device_id)
               .collection("events"))
    valid = [r for r in rows if r["lat"] != 0.0 or r["lon"] != 0.0]
    for i in range(0, max(len(valid), 1), 500):
        chunk = valid[i: i + 500]
        if not chunk:
            break
        batch = db.batch()
        for row in chunk:
            batch.set(col.document(), _fusion_doc(device_id, row))
        batch.commit()
    print(f"[anomaly_service] [STEP 1] Wrote {len(valid)} fused events for {device_id}")


def fetch_raw_events(device_id: str, hours: int = 24) -> list[dict]:
    """Read raw events from ttn_uplinks/{deviceId}/events."""
    from datetime import datetime, timezone, timedelta
    since_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    docs = (
        db.collection(RAW_COLLECTION)
          .document(device_id)
          .collection("events")
          .where("serverTime", ">=", since_dt)
          .order_by("serverTime")
          .stream()
    )
    events = []
    for doc in docs:
        d = doc.to_dict()
        d["_doc_id"] = doc.id
        events.append(d)
    return events


def fetch_fusion_events(device_id: str, hours: int = 24) -> list[dict]:
    """Read fused events from fusion_data/{deviceId}/events."""
    from datetime import datetime, timezone, timedelta
    since_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    docs = (
        db.collection(FUSION_COLLECTION)
          .document(device_id)
          .collection("events")
          .where("serverTime", ">=", since_dt)
          .order_by("serverTime")
          .stream()
    )
    events = []
    for doc in docs:
        d = doc.to_dict()
        d["_doc_id"] = doc.id
        events.append(d)
    return events


def run_fusion(device_id: str, hours: int = 24) -> int:
    """
    STEP 1 — Sensor fusion (batch):
      1. Read raw events from ttn_uplinks/{deviceId}/events
      2. Compute 7 features per event
      3. Write fused vectors to fusion_data/{deviceId}/events
    Returns the number of events written.
    """
    raw_events = fetch_raw_events(device_id, hours)
    if not raw_events:
        print(f"[anomaly_service] [STEP 1] No raw data for {device_id}")
        return 0
    feature_rows = compute_features(raw_events)
    write_fusion_batch(device_id, feature_rows)
    return len([r for r in feature_rows if r["lat"] != 0.0 or r["lon"] != 0.0])


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ISOLATION FOREST (reads from fusion_data)
# ═══════════════════════════════════════════════════════════════════════════════

def _severity(prediction: int, score: float) -> str:
    return "normal" if prediction == 1 else "anomaly"


def _fusion_docs_to_feature_rows(fusion_docs: list[dict]) -> list[dict]:
    """Convert fusion_data documents into feature dicts ready for the model."""
    rows = []
    for doc in fusion_docs:
        rows.append({
            "fused_x":               float(doc.get("fused_x", 0)),
            "fused_y":               float(doc.get("fused_y", 0)),
            "fused_vx":              float(doc.get("fused_vx", 0)),
            "fused_vy":              float(doc.get("fused_vy", 0)),
            "fused_speed":           float(doc.get("fused_speed", 0)),
            "acceleration_magnitude": float(doc.get("acceleration_magnitude", 0)),
            "movement_direction":    float(doc.get("movement_direction", 0)),
            "acc_mag":               float(doc.get("acceleration_magnitude", 0)),
            "acc_delta":             0.0,
            "speed":                 float(doc.get("fused_speed", 0)),
            "ax":  float(doc.get("ax", 0)),
            "ay":  float(doc.get("ay", 0)),
            "az":  float(doc.get("az", 9.8)),
            "lat": float(doc.get("lat", 0)),
            "lon": float(doc.get("lon", 0)),
            "doc_id":     doc.get("sourceDocId", doc.get("_doc_id", "")),
            "server_time": doc.get("serverTime"),
        })
    return rows


def predict_features(feature_rows: list[dict]) -> list[dict]:
    """Run Isolation Forest on a list of feature dicts. Returns enriched dicts."""
    if not feature_rows:
        return []
    X = np.array([[
        r["fused_x"], r["fused_y"], r["fused_vx"], r["fused_vy"],
        r["fused_speed"], r["acceleration_magnitude"], r["movement_direction"],
    ] for r in feature_rows])
    predictions = model.predict(X)
    scores      = model.decision_function(X)
    for i, row in enumerate(feature_rows):
        row["prediction"]    = int(predictions[i])
        row["anomaly_score"] = float(scores[i])
        row["is_anomaly"]    = predictions[i] == -1
        row["severity"]      = _severity(predictions[i], scores[i])
    return feature_rows


def write_anomaly_result(device_id: str, result_rows: list[dict]) -> None:
    """Write anomaly results to anomaly_results/{deviceId}."""
    anomalies = [r for r in result_rows if r["is_anomaly"]]
    db.collection(RESULT_COLLECTION).document(device_id).set({
        "deviceId":       device_id,
        "totalAnalysed":  len(result_rows),
        "totalAnomalies": len(anomalies),
        "updatedAt":   firestore.SERVER_TIMESTAMP,
        "events": [
            {
                "docId":             r["doc_id"],
                "lat":               r["lat"],
                "lon":               r["lon"],
                "accMag":            r["acc_mag"],
                "accDelta":          r["acc_delta"],
                "speed":             r["speed"],
                "fusedVx":           r["fused_vx"],
                "fusedVy":           r["fused_vy"],
                "movementDirection": r["movement_direction"],
                "anomalyScore":      r["anomaly_score"],
                "severity":          r["severity"],
                "isAnomaly":         r["is_anomaly"],
                "serverTime":        r["server_time"],
            }
            for r in anomalies
        ],
    })
    print(f"[anomaly_service] [STEP 2] {len(anomalies)} anomalies written for {device_id}")


def run_algorithm(device_id: str, hours: int = 24) -> dict:
    """
    STEP 2 — Isolation Forest (batch):
      1. Read fused events from fusion_data/{deviceId}/events
      2. Run IsolationForest model
      3. Write results to anomaly_results/{deviceId}
    """
    fusion_docs = fetch_fusion_events(device_id, hours)
    if not fusion_docs:
        print(f"[anomaly_service] [STEP 2] No fusion data for {device_id} — run /fuse first")
        return {"deviceId": device_id, "status": "no_fusion_data", "eventsAnalysed": 0}

    feature_rows = _fusion_docs_to_feature_rows(fusion_docs)
    result_rows  = predict_features(feature_rows)
    write_anomaly_result(device_id, result_rows)

    anomalies = [r for r in result_rows if r["is_anomaly"]]
    return {
        "deviceId":       device_id,
        "status":         "ok",
        "eventsAnalysed": len(result_rows),
        "anomaliesFound": len(anomalies),
    }


def process_device(device_id: str, hours: int = 24) -> dict:
    """Run STEP 1 then STEP 2 for one device."""
    fused = run_fusion(device_id, hours)
    if fused == 0:
        return {"deviceId": device_id, "status": "no_data", "eventsAnalysed": 0}
    return run_algorithm(device_id, hours)


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-TIME LISTENER (one event at a time)
# ═══════════════════════════════════════════════════════════════════════════════

def start_realtime_listener():
    """
    Watches ttn_uplinks/{deviceId}/events for new documents.
    For each new event:
      STEP 1  compute fused features (using the previous event for velocity)
              write to fusion_data/{deviceId}/events
      STEP 2  run IsolationForest on the fused vector
              write a live_alert if an anomaly is detected
    """
    if not FIREBASE_AVAILABLE:
        return

    print("[anomaly_service] Starting real-time listeners...")

    def on_snapshot(_col_snapshot, changes, _read_time, device_id):
        for change in changes:
            if change.type.name != "ADDED":
                continue

            doc = change.document.to_dict()
            doc["_doc_id"] = change.document.id

            # ── STEP 1: EKF sensor fusion ────────────────────────────────────
            current_row = compute_single_ekf(device_id, doc)
            if current_row is None:
                continue   # first event for this device — EKF initialised, skip

            write_fusion_event(device_id, current_row)

            # ── STEP 2: Isolation Forest ─────────────────────────────────────
            result_rows = predict_features([current_row])
            if result_rows and result_rows[0]["is_anomaly"]:
                r = result_rows[0]
                print(f"[anomaly_service] ** ANOMALY on {device_id}: "
                      f"severity={r['severity']}, score={r['anomaly_score']:.4f}")
                db.collection(RESULT_COLLECTION).document(device_id) \
                  .collection("live_alerts").add({
                    "deviceId":          device_id,
                    "severity":          r["severity"],
                    "anomalyScore":      r["anomaly_score"],
                    "accMag":            r["acc_mag"],
                    "accDelta":          r["acc_delta"],
                    "speed":             r["speed"],
                    "fusedVx":           r["fused_vx"],
                    "fusedVy":           r["fused_vy"],
                    "movementDirection": r["movement_direction"],
                    "lat":               r["lat"],
                    "lon":               r["lon"],
                    "detectedAt":        firestore.SERVER_TIMESTAMP,
                })

    from datetime import datetime, timezone
    # Only watch events written AFTER this service started.
    # This prevents historical events from polluting the EKF state.
    service_start = datetime.now(timezone.utc)
    print(f"[anomaly_service] Ignoring events older than: {service_start}")

    watchers = []
    for device_id in DEVICE_IDS:
        col_ref = (db.collection(RAW_COLLECTION)
                     .document(device_id)
                     .collection("events")
                     .where("serverTime", ">=", service_start))
        watcher = col_ref.on_snapshot(
            lambda snap, changes, rt, did=device_id: on_snapshot(snap, changes, rt, did)
        )
        watchers.append(watcher)
        print(f"[anomaly_service]   Listening on {RAW_COLLECTION}/{device_id}/events")

    return watchers


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK REST API
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model":  "IsolationForest",
        "n_features": model.n_features_in_,
        "features": [
            "fused_x", "fused_y", "fused_vx", "fused_vy",
            "fused_speed", "acceleration_magnitude", "movement_direction",
        ],
        "collections": {
            "raw":    RAW_COLLECTION,
            "fusion": FUSION_COLLECTION,
            "result": RESULT_COLLECTION,
        },
    })


@app.route("/fuse/<device_id>", methods=["POST"])
def api_fuse_single(device_id):
    """STEP 1: read raw events → compute features → write to fusion_data."""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firestore not configured"}), 503
    hours = int(request.args.get("hours", 24))
    fused = run_fusion(device_id, hours)
    return jsonify({
        "deviceId":    device_id,
        "status":      "ok" if fused > 0 else "no_data",
        "fusedEvents": fused,
    })


@app.route("/fuse/all", methods=["POST"])
def api_fuse_all():
    """STEP 1 for all configured devices."""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firestore not configured"}), 503
    summaries = []
    for device_id in DEVICE_IDS:
        fused = run_fusion(device_id)
        summaries.append({
            "deviceId":    device_id,
            "status":      "ok" if fused > 0 else "no_data",
            "fusedEvents": fused,
        })
    return jsonify({"devices": summaries})


@app.route("/predict/<device_id>", methods=["POST"])
def api_predict_single(device_id):
    """STEP 2: read fusion_data → run model → write anomaly_results."""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firestore not configured"}), 503
    hours = int(request.args.get("hours", 24))
    return jsonify(run_algorithm(device_id, hours))


@app.route("/process/<device_id>", methods=["POST"])
def api_process_single(device_id):
    """STEP 1 + STEP 2 combined for one device."""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firestore not configured"}), 503
    hours = int(request.args.get("hours", 24))
    return jsonify(process_device(device_id, hours))


@app.route("/process/all", methods=["POST"])
def api_process_all():
    """STEP 1 + STEP 2 for all devices."""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firestore not configured"}), 503
    summaries = []
    for device_id in DEVICE_IDS:
        result = process_device(device_id)
        summaries.append({
            "deviceId":       result["deviceId"],
            "status":         result["status"],
            "eventsAnalysed": result.get("eventsAnalysed", 0),
            "anomaliesFound": result.get("anomaliesFound", 0),
        })
    return jsonify({"devices": summaries})


@app.route("/fusion/<device_id>", methods=["GET"])
def api_get_fusion(device_id):
    """Inspect fused events stored in fusion_data/{deviceId}/events."""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firestore not configured"}), 503
    hours  = int(request.args.get("hours", 24))
    events = fetch_fusion_events(device_id, hours)
    return jsonify({"deviceId": device_id, "total": len(events), "events": events})


@app.route("/results/<device_id>", methods=["GET"])
def api_get_results(device_id):
    """Inspect anomaly results stored in anomaly_results/{deviceId}."""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firestore not configured"}), 503
    doc = db.collection(RESULT_COLLECTION).document(device_id).get()
    if not doc.exists:
        return jsonify({"deviceId": device_id, "status": "no_results"})
    return jsonify(doc.to_dict())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if FIREBASE_AVAILABLE:
        listener_thread = threading.Thread(target=start_realtime_listener, daemon=True)
        listener_thread.start()

    print(f"\n[anomaly_service] >> Flask API running on http://localhost:{PORT}")
    print(f"[anomaly_service] Pipeline:")
    print(f"  ttn_uplinks  →  [STEP 1 fusion]  →  fusion_data  →  [STEP 2 model]  →  anomaly_results")
    print(f"[anomaly_service] Endpoints:")
    print(f"  GET  /health")
    print(f"  POST /fuse/<device_id>     STEP 1: raw → fusion_data")
    print(f"  POST /fuse/all             STEP 1 for all devices")
    print(f"  POST /predict/<device_id>  STEP 2: fusion_data → anomaly_results")
    print(f"  POST /process/<device_id>  STEP 1 + 2 combined")
    print(f"  POST /process/all          STEP 1 + 2 for all devices")
    print(f"  GET  /fusion/<device_id>   inspect fusion_data events")
    print(f"  GET  /results/<device_id>  inspect anomaly_results\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
