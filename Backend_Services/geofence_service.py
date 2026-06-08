"""
geofence_service.py
===================
Python microservice for Random Forest geofence classification.

Architecture:
  - Loads random_forest_geofence_model_v2.pkl (RandomForestClassifier, 8 features, 300 estimators)
  - Accepts POST /classify with a device point + geofence polygon + optional motion data
  - Engineers 8 features matching the model:
      fused_x            = latitude
      fused_y            = longitude
      fused_vx           = velocity east  (m/s) — from request or derived from GPS diff
      fused_vy           = velocity north (m/s) — from request or derived from GPS diff
      fused_speed        = total speed (m/s)
      acceleration_magnitude = sqrt(ax^2 + ay^2 + az^2)
      movement_direction = compass bearing 0-360 deg
      distance_to_boundary   = min distance to polygon edge (metres)
  - Model classes: "IN" | "NEAR_BOUNDARY" | "OUT"
    mapped to Flutter-friendly: "inside" | "nearBoundary" | "outside"
  - Falls back to pure geometric algorithm if model file is missing

Requirements:
  pip install flask flask-cors scikit-learn joblib numpy

Usage:
  Run:  python geofence_service.py
  API available at http://localhost:5001

Endpoints:
  GET  /health                -- service status + model info
  POST /classify              -- classify a single point vs polygon
  POST /classify/batch        -- classify multiple points in one call
"""

import os
import math
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Feature column order — must match what the model was trained on
FEATURE_COLS = [
    "fused_x", "fused_y", "fused_vx", "fused_vy",
    "fused_speed", "acceleration_magnitude",
    "movement_direction", "distance_to_boundary",
]

# --- Configuration -----------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest_geofence_model_v2.pkl")
PORT = 5001

# Near-boundary threshold (metres) used by the geometric fallback only
NEAR_THRESHOLD_M = 5.0

# Model class labels -> Flutter-friendly labels
LABEL_MAP = {
    "IN":            "inside",
    "NEAR_BOUNDARY": "nearBoundary",
    "OUT":           "outside",
    # pass-through in case model already uses friendly names
    "inside":        "inside",
    "nearBoundary":  "nearBoundary",
    "outside":       "outside",
}

MODEL_CLASSES = ["inside", "nearBoundary", "outside"]

# --- Load model --------------------------------------------------------------

model = None

try:
    import joblib
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, "predict"), "Loaded object has no .predict() method"
        print(f"[geofence_service] [OK] Model loaded: {MODEL_PATH}")
        print(f"[geofence_service]    Classes     : {[str(c) for c in model.classes_]}")
        print(f"[geofence_service]    n_features  : {model.n_features_in_}")
        print(f"[geofence_service]    n_estimators: {model.n_estimators}")
        if hasattr(model, "feature_names_in_"):
            print(f"[geofence_service]    Features    : {list(model.feature_names_in_)}")
    else:
        print(f"[geofence_service] [WARN] Model not found at: {MODEL_PATH}")
        print("[geofence_service]    Geometric fallback will be used.")
except Exception as e:
    print(f"[geofence_service] [WARN] Could not load model: {e}")
    print("[geofence_service]    Geometric fallback active.")

# --- Geometry helpers --------------------------------------------------------

_EARTH_R = 6_371_000.0  # metres


def _to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _point_to_metres(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Equirectangular projection: returns (x_m east, y_m north) relative to ref point."""
    cos_lat = math.cos(_to_rad(ref_lat))
    x = _to_rad(lon - ref_lon) * _EARTH_R * cos_lat
    y = _to_rad(lat - ref_lat) * _EARTH_R
    return x, y


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing in degrees (0=N, 90=E, 180=S, 270=W)."""
    dlam = _to_rad(lon2 - lon1)
    lat1r, lat2r = _to_rad(lat1), _to_rad(lat2)
    x = math.sin(dlam) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def is_inside_ray_cast(lat: float, lon: float, polygon: list[dict]) -> bool:
    """Standard ray-casting inside-polygon test. lon=x axis, lat=y axis."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]["lon"], polygon[i]["lat"]
        xj, yj = polygon[j]["lon"], polygon[j]["lat"]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def dist_to_boundary_m(lat: float, lon: float, polygon: list[dict]) -> float:
    """Minimum distance in metres from point to any edge of the polygon."""
    n = len(polygon)
    if n == 0:
        return float("inf")
    ref_lat = polygon[0]["lat"]
    ref_lon = polygon[0]["lon"]
    px, py = _point_to_metres(lat, lon, ref_lat, ref_lon)
    min_sq = float("inf")
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        ax, ay = _point_to_metres(a["lat"], a["lon"], ref_lat, ref_lon)
        bx, by = _point_to_metres(b["lat"], b["lon"], ref_lat, ref_lon)
        dx, dy = bx - ax, by - ay
        len_sq = dx * dx + dy * dy
        t = 0.0 if len_sq == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
        cx, cy = ax + t * dx, ay + t * dy
        sq = (px - cx) ** 2 + (py - cy) ** 2
        if sq < min_sq:
            min_sq = sq
    return math.sqrt(min_sq) if min_sq < float("inf") else float("inf")


def geometric_status(lat: float, lon: float, polygon: list[dict]) -> str:
    """Pure geometric classification: inside / nearBoundary / outside."""
    inside = is_inside_ray_cast(lat, lon, polygon)
    if not inside:
        return "outside"
    dist = dist_to_boundary_m(lat, lon, polygon)
    return "nearBoundary" if dist <= NEAR_THRESHOLD_M else "inside"


# --- Feature engineering -----------------------------------------------------

def build_features(
    lat: float,
    lon: float,
    polygon: list[dict],
    vx: float = 0.0,
    vy: float = 0.0,
    speed: float = 0.0,
    ax: float = 0.0,
    ay: float = 0.0,
    az: float = 9.8,
    direction: float = 0.0,
) -> np.ndarray:
    """
    Build the 8-feature vector matching random_forest_geofence_model.pkl:
      [fused_x, fused_y, fused_vx, fused_vy, fused_speed,
       acceleration_magnitude, movement_direction, distance_to_boundary]
    """
    acc_mag = math.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    fused_speed = speed if speed > 0.0 else math.sqrt(vx ** 2 + vy ** 2)
    d_boundary = dist_to_boundary_m(lat, lon, polygon)

    return np.array([[
        lat,          # fused_x
        lon,          # fused_y
        vx,           # fused_vx
        vy,           # fused_vy
        fused_speed,  # fused_speed
        acc_mag,      # acceleration_magnitude
        direction,    # movement_direction
        d_boundary,   # distance_to_boundary
    ]], dtype=np.float64)


def classify_point(
    lat: float,
    lon: float,
    polygon: list[dict],
    vx: float = 0.0,
    vy: float = 0.0,
    speed: float = 0.0,
    ax: float = 0.0,
    ay: float = 0.0,
    az: float = 9.8,
    direction: float = 0.0,
) -> dict:
    """
    Classify a point using the RF model if available, else geometric fallback.
    Returns dict with: status, confidence, method, features.
    """
    acc_mag = math.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    fused_speed = speed if speed > 0.0 else math.sqrt(vx ** 2 + vy ** 2)
    d_boundary = dist_to_boundary_m(lat, lon, polygon)

    feature_dict = {
        "fused_x": lat,
        "fused_y": lon,
        "fused_vx": round(vx, 4),
        "fused_vy": round(vy, 4),
        "fused_speed": round(fused_speed, 4),
        "acceleration_magnitude": round(acc_mag, 4),
        "movement_direction": round(direction, 2),
        "distance_to_boundary": round(d_boundary, 3),
    }

    if model is not None:
        X = pd.DataFrame([[
            lat, lon, vx, vy, fused_speed, acc_mag, direction, d_boundary
        ]], columns=FEATURE_COLS)

        raw_label = str(model.predict(X)[0])
        friendly_label = LABEL_MAP.get(raw_label, raw_label)

        confidence = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = {
                LABEL_MAP.get(str(cls), str(cls)): round(float(p), 4)
                for cls, p in zip(model.classes_, proba)
            }

        return {
            "status": friendly_label,
            "raw_label": raw_label,
            "confidence": confidence,
            "method": "random_forest",
            "features": feature_dict,
        }
    else:
        geo_status = geometric_status(lat, lon, polygon)
        return {
            "status": geo_status,
            "raw_label": geo_status,
            "confidence": {},
            "method": "geometric_fallback",
            "features": feature_dict,
        }


# --- Flask API ---------------------------------------------------------------

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    model_classes = [LABEL_MAP.get(str(c), str(c)) for c in model.classes_] if model is not None else MODEL_CLASSES
    raw_classes = [str(c) for c in model.classes_] if model is not None else []
    return jsonify({
        "status": "ok",
        "model": "RandomForestClassifier" if model is not None else None,
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "n_features": model.n_features_in_ if model is not None else 0,
        "n_estimators": model.n_estimators if model is not None else 0,
        "features": [
            "fused_x", "fused_y", "fused_vx", "fused_vy",
            "fused_speed", "acceleration_magnitude",
            "movement_direction", "distance_to_boundary",
        ],
        "classes": model_classes,
        "raw_classes": raw_classes,
        "near_threshold_m": NEAR_THRESHOLD_M,
        "fallback_active": model is None,
    })


@app.route("/classify", methods=["POST"])
def classify():
    """
    POST /classify
    Body (JSON):
    {
      "lat": 3.2541,
      "lon": 101.7291,
      "polygon": [{"lat": 3.2540, "lon": 101.7288}, ...],

      -- Optional motion fields (improves accuracy) --
      "vx":        0.5,    -- velocity east  (m/s)
      "vy":        0.3,    -- velocity north (m/s)
      "speed":     0.58,   -- total speed (m/s)
      "ax":        0.07,   -- accelerometer x
      "ay":        0.08,   -- accelerometer y
      "az":        9.8,    -- accelerometer z
      "direction": 45.0    -- movement bearing (degrees)
    }
    Response:
    {
      "status":    "inside" | "nearBoundary" | "outside",
      "raw_label": "IN"    | "NEAR_BOUNDARY" | "OUT",
      "confidence": {"inside": 0.85, "nearBoundary": 0.10, "outside": 0.05},
      "method":    "random_forest" | "geometric_fallback",
      "features":  { ... }
    }
    """
    data = request.get_json(force=True)

    lat = data.get("lat")
    lon = data.get("lon")
    polygon = data.get("polygon", [])

    if lat is None or lon is None:
        return jsonify({"error": "Missing 'lat' or 'lon'"}), 400
    if len(polygon) < 3:
        return jsonify({"error": "Polygon must have at least 3 points"}), 400

    try:
        result = classify_point(
            float(lat), float(lon), polygon,
            vx=float(data.get("vx", 0.0)),
            vy=float(data.get("vy", 0.0)),
            speed=float(data.get("speed", 0.0)),
            ax=float(data.get("ax", 0.0)),
            ay=float(data.get("ay", 0.0)),
            az=float(data.get("az", 9.8)),
            direction=float(data.get("direction", 0.0)),
        )
        print(f"[classify] lat={lat:.6f} lon={lon:.6f} -> {result['status']} ({result['method']})")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/classify/batch", methods=["POST"])
def classify_batch():
    """
    POST /classify/batch
    Body:
    {
      "points": [
        {
          "device_id": "dtestt1",
          "lat": 3.2541, "lon": 101.7291,
          -- optional: vx, vy, speed, ax, ay, az, direction --
        }
      ],
      "polygon": [{"lat": ..., "lon": ...}, ...]
    }
    Response:
    {
      "results": [
        {"device_id": "dtestt1", "status": "inside", "confidence": {...}, "method": "..."}
      ],
      "total": 1
    }
    """
    data = request.get_json(force=True)
    points = data.get("points", [])
    polygon = data.get("polygon", [])

    if len(polygon) < 3:
        return jsonify({"error": "Polygon must have at least 3 points"}), 400
    if not points:
        return jsonify({"error": "No points provided"}), 400

    results = []
    for p in points:
        lat = p.get("lat")
        lon = p.get("lon")
        device_id = p.get("device_id", "")
        if lat is None or lon is None:
            results.append({"device_id": device_id, "error": "Missing lat/lon"})
            continue
        try:
            r = classify_point(
                float(lat), float(lon), polygon,
                vx=float(p.get("vx", 0.0)),
                vy=float(p.get("vy", 0.0)),
                speed=float(p.get("speed", 0.0)),
                ax=float(p.get("ax", 0.0)),
                ay=float(p.get("ay", 0.0)),
                az=float(p.get("az", 9.8)),
                direction=float(p.get("direction", 0.0)),
            )
            r["device_id"] = device_id
            results.append(r)
        except Exception as e:
            results.append({"device_id": device_id, "error": str(e)})

    return jsonify({"results": results, "total": len(results)})


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    print(f"[geofence_service] >> Starting on http://localhost:{PORT}")
    print(f"[geofence_service] Model loaded: {model is not None}")
    print(f"[geofence_service] Endpoints:")
    print(f"  GET  /health")
    print(f"  POST /classify           -- classify single point vs polygon")
    print(f"  POST /classify/batch     -- classify multiple devices at once")
    app.run(host="0.0.0.0", port=PORT, debug=False)
