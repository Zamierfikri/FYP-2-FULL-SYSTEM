# Fulcrum System
 
**A three-node wearable IoT platform for sensor-fusion-based crowd geofencing and anomaly detection.**
 
The Fulcrum System is a Final Year Project (FYP2) developed at the Kulliyyah of Information and Communication Technology (KICT), International Islamic University Malaysia (IIUM). It extends a single-node geofencing algorithm (FYP1) into a distributed, three-node wearable platform that fuses GNSS and accelerometer data, classifies each wearer's position relative to a defined safe zone, and flags abnormal movement in near real time.
 
The system tracks three wearable nodes worn by individuals moving through a monitored area, streams their location and motion data over LoRaWAN to the cloud, and runs two machine-learning services that answer two questions continuously: *Is this person inside, near, or outside the safe zone?* and *Is this person's movement anomalous?*
 
---
 
## Table of Contents
 
- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Hardware](#hardware)
- [Communication Layer](#communication-layer)
- [Data Pipeline & Firestore Schema](#data-pipeline--firestore-schema)
- [Backend Services](#backend-services)
- [Algorithms](#algorithms)
- [Dashboard](#dashboard)
- [Field Testing & Results](#field-testing--results)
- [Setup & Installation](#setup--installation)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
---
 
## System Overview
 
The platform is organised into four layers that communicate in one direction, from the body-worn sensors up to the dashboard:
 
1. **Edge (wearable nodes)** — Three Heltec Wireless Tracker boards collect GNSS position and 3-axis acceleration, then transmit a compact payload over LoRaWAN roughly every 13 seconds.
2. **Network (LoRaWAN)** — A Heltec HT-M2802 gateway relays uplinks to The Things Network (TTN) V3, which forwards them via webhook to the cloud backend.
3. **Cloud (ingestion + processing)** — A Firebase Cloud Function writes raw uplinks into Firestore. Two Python microservices read from Firestore, run sensor fusion and machine-learning inference, and write results back.
4. **Presentation (dashboard)** — A Flutter web dashboard subscribes to Firestore and visualises live positions, safe zones, history, device health, and anomalies.
A defining design choice is that **all three software components communicate exclusively through Firestore** rather than calling each other directly. Firestore acts as the shared state and message bus, which keeps the services decoupled and the data flow easy to inspect.
 
---
 
## Architecture
 
```
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Fulcrum-A│   │ Fulcrum-B│   │ Fulcrum-C│      Three wearable nodes
  │  (Alia)  │   │  (Hana)  │   │  (Faris) │      ESP32-S3 + GNSS + IMU + LoRa
  └────┬─────┘   └────┬─────┘   └────┬─────┘
       │              │              │            LoRaWAN AU915 / AS923, SF7
       └──────────────┼──────────────┘            14-byte payload, ~13 s cycle
                      │
              ┌───────▼────────┐
              │ HT-M2802 Gateway│
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │   TTN V3 (AU1) │                  Webhook integration
              └───────┬────────┘
                      │
              ┌───────▼────────────┐
              │ Firebase Cloud Fn  │               Writes raw uplinks
              └───────┬────────────┘
                      │
              ┌───────▼────────────────────────────┐
              │           Firestore                 │   Shared state / message bus
              │  ttn_uplinks → fusion_data →        │
              │  anomaly_results                    │
              └───┬───────────────┬─────────────┬───┘
                  │               │             │
        ┌─────────▼──────┐ ┌──────▼────────┐ ┌──▼──────────────┐
        │ anomaly_service│ │geofence_service│ │ Flutter web      │
        │ (port 5000)    │ │ (port 5001)    │ │ dashboard        │
        │ EKF + Isolation│ │ Random Forest  │ │ 5 tabs           │
        │ Forest         │ │ classifier     │ │                  │
        └────────────────┘ └────────────────┘ └──────────────────┘
```
 
---
 
## Hardware
 
Each of the three nodes is built on the same hardware platform:
 
| Component | Part | Role |
|-----------|------|------|
| MCU / Radio board | Heltec Wireless Tracker (ESP32-S3) | Compute, Wi-Fi/BLE, board integration |
| GNSS | UC6580 | Multi-constellation positioning |
| Accelerometer | ADXL345 (3-axis) | Motion / acceleration sensing |
| LoRa transceiver | SX1262 | LoRaWAN uplink radio |
 
The three nodes are identified throughout the system by three naming conventions that all refer to the same physical devices:
 
| Node | Device ID | Wearer |
|------|-----------|--------|
| Fulcrum-A | `dtestt1` | Alia |
| Fulcrum-B | `node7` | Hana |
| Fulcrum-C | `node8` | Faris |
 
The network side uses a single **Heltec HT-M2802** LoRaWAN gateway.
 
---
 
## Communication Layer
 
- **Protocol:** LoRaWAN, AU915 / AS923 frequency plan
- **Spreading factor:** SF7. Adaptive Data Rate (ADR) was deliberately pushed down from SF9 to SF7 during field trials to shorten airtime.
- **Active channels:** Two AS923 gateway channels (923.2 MHz and 923.4 MHz) were observed in use. This is normal AS923 frequency hopping across the channel plan, not two distinct uplink types.
- **Payload:** A fixed-width **14-byte** binary payload containing node ID, latitude, longitude, three accelerometer axes, and a timestamp.
- **Uplink cadence:** Target ~13–15 seconds per uplink. In the field, the observed cadence was closer to ~60 seconds; this is explained by nodes stopping at different times during a run rather than by coverage gaps.
- **Routing:** Gateway → TTN V3 (AU1 cluster) → Firebase Cloud Function webhook → Firestore.
Per-node link quality differed during testing. Faris (Fulcrum-C) had the weakest link, with a mean RSSI of roughly −106 dBm attributable to body shadowing. Alia (Fulcrum-A) produced fewer points because of early run termination and GPS warm-up periods that emitted `(0,0)` coordinates. Hana (Fulcrum-B) provided the cleanest baseline.
 
---
 
## Data Pipeline & Firestore Schema
 
Data flows through three Firestore collections, each keyed by device:
 
| Collection | Path | Written by | Contents |
|-----------|------|-----------|----------|
| Raw uplinks | `ttn_uplinks/{deviceId}/events` | Cloud Function | Decoded TTN uplinks (raw GNSS + accel + timestamp) |
| Fused state | `fusion_data/{deviceId}/events` | `anomaly_service.py` | EKF state outputs and derived features |
| Live alerts | `anomaly_results/{deviceId}/live_alerts` | services | Geofence classifications and anomaly flags |
 
This structure makes the pipeline traceable end to end: a single uplink can be followed from raw ingestion, through fusion, to its final classification.
 
---
 
## Backend Services
 
Two independent Python microservices run the processing logic. Both read from and write to Firestore; neither calls the other directly.
 
### `anomaly_service.py` (port 5000)
Performs Extended Kalman Filter (EKF) sensor fusion and runs the Isolation Forest anomaly detector. It consumes raw uplinks, produces fused state, and emits anomaly flags.
 
### `geofence_service.py` (port 5001)
Runs the Random Forest geofence classifier, labelling each fused position as `INSIDE`, `NEAR_BOUNDARY`, or `OUTSIDE` relative to the configured safe zone.
 
A lightweight **client-side fallback** also runs in the Flutter dashboard (in Dart), providing rule-based and Z-score detection when the cloud services are unavailable (see Algorithms below).
 
---
 
## Algorithms
 
### Extended Kalman Filter (sensor fusion)
 
The EKF smooths noisy GNSS positions using a constant-velocity model.
 
- **State vector:** `[x_pos, y_pos, vx, vy]`
- **Noise parameters:** `R = 5` (GPS measurement noise), `Q = 0.1` (process noise)
- **Time step (`dt`):** Computed from real server timestamps, not assumed constant.
- **Initialisation:** The first event per device only initialises the filter; no fused output is produced until the second event arrives.
- **Operating mode:** The filter runs on **one snapshot per uplink** (~13 s apart). It is not a high-rate dead-reckoning filter — the low sampling cadence is an inherent characteristic of the LoRaWAN link.
- **Evaluation:** Measured via trajectory path-length reduction (a proxy for smoothing): Fulcrum-A 11.2%, Fulcrum-B 6.2%, Fulcrum-C 1.9%.
**Feature provenance note (important for interpretation):** Only `fused_x`, `fused_y`, `fused_vx`, and `fused_vy` are true EKF state outputs. `fused_speed` and `movement_direction` are trigonometric post-processing of those outputs, and `acceleration_magnitude` is raw IMU data with no EKF involvement.
 
### Random Forest (geofence classification)
 
- **Features (8):** `fused_x`, `fused_y`, `fused_vx`, `fused_vy`, `fused_speed`, `acceleration_magnitude`, `movement_direction`, `distance_to_boundary`
- **Classes (3):** `INSIDE`, `NEAR_BOUNDARY`, `OUTSIDE`, with a ±5 m boundary band
- **Labels:** Auto-generated via a signed-distance function relative to the safe-zone polygon
- **Training/testing:** Trained on Datasets 1 + 2 (1,198 samples), tested on a held-out Dataset 3 (598 samples)
- **Results:** **95.74% accuracy, 89.80% macro F1, zero INSIDE↔OUTSIDE hard misclassifications**
- **Top features by importance:** `fused_x` (0.347), `distance_to_boundary` (0.343), `fused_y` (0.208)
### Isolation Forest (anomaly detection)
 
- **Features (7):** The Random Forest feature set minus `distance_to_boundary`
- **Contamination:** `0.03`
- **Training:** Dataset 2 only (a walking "normal" baseline)
- **Per-dataset detection:** D1 (running) 527/599 = 88.0% flagged anomalous; D2 (walking) 32/599 = 5.3% false positives; D3 (walking) 56/598 = 9.4% false positives
- **Overall:** 91.09% accuracy, 85.69% precision, 87.98% recall, 86.82% F1
### Client-side fallback (Dart)
 
A rule-based and Z-score detector in the dashboard for offline resilience:
 
- **HIGH:** speed > 25 km/h
- **MEDIUM:** speed > 15 km/h, or a GPS jump > 300 m in < 20 s, or no movement for > 45 min
- **LOW:** Z-score > 2.5
---
 
## Dashboard
 
A Flutter web dashboard provides the operator view, organised into five tabs:
 
1. **Map** — Live node positions over the monitored area
2. **Safe Zones** — Defined geofence polygons
3. **History** — Past tracks and events
4. **Devices** — Per-node health and link status
5. **Anomalies** — Flagged abnormal-movement events
The dashboard reads from Firestore in real time and applies the client-side fallback detector when needed.
 
---
 
## Field Testing & Results
 
Field testing was conducted at the **KICT Roundabout**, using five checkpoints labelled A–E.
 
- **Scenario 1 (complete loop):** Real captured data. This is the basis for the reported results.

**End-to-end latency** (uplink to alert):
 
| Metric | Value |
|--------|-------|
| Mean | ~1.11 s (skewed by a cold-start outlier) |
| Mean (excluding outlier) | ~0.79 s |
| Median | ~0.68 s |
| Ingestion (mean / median) | 0.328 s / 0.233 s |
| Fusion (mean / median) | 0.674 s / 0.237 s |
| Alert | ~0.10 s |
 
 
## Setup & Installation
 
> **Before anything:** never commit secrets. The Firebase service-account JSON, TTN API keys, and LoRaWAN DevEUI/AppKey must stay out of the repo. Commit `config.h.example` and `.env.example` with placeholder values instead.
 
### 1. Firmware
Open `firmware/fulcrum-node/` in the Arduino IDE or PlatformIO, copy `config.h.example` to `config.h`, fill in your node ID and LoRaWAN keys, then flash to the Heltec Wireless Tracker.
 
### 2. Backend services
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
 
# Provide Firebase credentials via environment variable (do not commit the file)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
 
python anomaly_service.py     # port 5000
python geofence_service.py    # port 5001
```
 
### 3. Cloud Function
Deploy the TTN webhook function in `cloud-functions/ttn-webhook/` to Firebase, then point your TTN V3 application's webhook integration at its URL.
 
### 4. Dashboard
```bash
cd dashboard/flutter-web
flutter pub get
flutter run -d chrome
```
 
> **Reproducibility:** pin your `scikit-learn` version in `requirements.txt`. Isolation Forest and Random Forest behaviour can shift across versions, and the reported metrics (95.74% / 91.09%) are tied to the version used during evaluation.
 
---
 
## Limitations
 
- **EKF cadence:** The filter operates on a single snapshot per ~13 s uplink, so it provides position smoothing rather than fine-grained dead reckoning.
- **Isolation Forest feature mismatch:** Some features were designed assuming a high sampling rate that is inconsistent with the live uplink cadence (~13 s). This is a known limitation.
- **Link asymmetry:** Body shadowing and GPS warm-up produced uneven point counts across nodes, with Fulcrum-C the weakest link.
- **Single gateway:** All testing used one gateway; multi-gateway coverage was not evaluated.
---
 
## Future Work
 
- Complete Scenario 3 (mixed running/walking testing).
- Compare inter-node dispersion during separate (non-grouped) walks.
- Revisit anomaly-detection features to better match the real uplink cadence.
---
 
## Acknowledgements
 
Developed by **Muhammad Zamir Fikri** (Mechatronics Engineering, KICT, IIUM) as a Final Year Project, under the supervision of **Dr. Muhammad Afif Husman**.
