import 'dart:math';
import 'package:latlong2/latlong.dart';

// ─────────────────────────────────────────────────────────────────────────────
// ANOMALY DETECTOR — Pure Dart, statistical (Z-score + rule-based)
//
// Algorithm overview:
//   1. Haversine formula → distance between consecutive GPS points
//   2. Speed (km/h) = distance / elapsed time
//   3. Z-score on speed distribution → flag statistical outliers
//   4. Hard rule checks (impossible speed, GPS teleport, device abandoned)
// ─────────────────────────────────────────────────────────────────────────────

enum AnomalySeverity { low, medium, high }

class AnomalyResult {
  final String deviceId;
  final DateTime timestamp;
  final LatLng location;
  final double speedKmh;
  final double distanceM;
  final bool isAnomaly;
  final AnomalySeverity severity;
  final String reason;

  const AnomalyResult({
    required this.deviceId,
    required this.timestamp,
    required this.location,
    required this.speedKmh,
    required this.distanceM,
    required this.isAnomaly,
    required this.severity,
    required this.reason,
  });
}

class DeviceLocationPoint {
  final String deviceId;
  final LatLng location;
  final DateTime timestamp;

  const DeviceLocationPoint({
    required this.deviceId,
    required this.location,
    required this.timestamp,
  });
}

class AnomalyDetector {
  // ── Tunable thresholds ────────────────────────────────────────────────────

  /// Above this → suspicious, the tracked device is likely being carried or in a vehicle.
  static const double _impossibleSpeedKmh = 15.0;

  /// Above this → clearly impossible on foot, likely in a vehicle.
  static const double _highSeveritySpeedKmh = 25.0;

  /// GPS teleport: large distance in a very short time (< 20 s) → glitch.
  static const double _teleportDistanceM = 300.0;
  static const int _teleportWindowSec = 20;

  /// Device abandonment: no movement for this many minutes.
  static const int _abandonmentMinutes = 45;

  /// Minimum displacement (metres) to count as movement (filter GPS noise).
  static const double _movementNoiseM = 5.0;

  /// Z-score threshold to flag a point as a statistical anomaly.
  static const double _zScoreThreshold = 2.5;

  // ── Public API ─────────────────────────────────────────────────────────────

  /// Analyse a chronologically-ordered list of location points
  /// (oldest first) and return one [AnomalyResult] per gap.
  static List<AnomalyResult> analyze(List<DeviceLocationPoint> history) {
    if (history.length < 2) return [];

    // Sort ascending (oldest → newest)
    final sorted = List<DeviceLocationPoint>.from(history)
      ..sort((a, b) => a.timestamp.compareTo(b.timestamp));

    // ── Step 1: compute per-segment speed & distance ──────────────────────
    final List<_Segment> segments = [];
    for (int i = 1; i < sorted.length; i++) {
      final prev = sorted[i - 1];
      final curr = sorted[i];
      final distM = _haversine(prev.location, curr.location);
      final elapsedSec =
          curr.timestamp.difference(prev.timestamp).inSeconds.toDouble();
      final speedKmh =
          elapsedSec > 0 ? (distM / 1000) / (elapsedSec / 3600) : 0.0;

      segments.add(_Segment(
        deviceId: curr.deviceId,
        from: prev,
        to: curr,
        distanceM: distM,
        elapsedSec: elapsedSec,
        speedKmh: speedKmh,
      ));
    }

    // ── Step 2: Z-score on the speed distribution ─────────────────────────
    final speeds = segments.map((s) => s.speedKmh).toList();
    final mean = _mean(speeds);
    final std = _std(speeds, mean);

    // ── Step 3: classify each segment ─────────────────────────────────────
    final List<AnomalyResult> results = [];

    for (final seg in segments) {
      final zScore = std > 0 ? (seg.speedKmh - mean) / std : 0.0;
      final isSmallMovement = seg.distanceM < _movementNoiseM;

      String? reason;
      AnomalySeverity severity = AnomalySeverity.low;
      bool isAnomaly = false;

      // Rule 1: Impossible speed
      if (seg.speedKmh > _highSeveritySpeedKmh) {
        isAnomaly = true;
        severity = AnomalySeverity.high;
        reason = 'Speed too high — may be in a vehicle (${seg.speedKmh.toStringAsFixed(1)} km/h)';
      } else if (seg.speedKmh > _impossibleSpeedKmh) {
        isAnomaly = true;
        severity = AnomalySeverity.medium;
        reason = 'Speed too fast for walking pace (${seg.speedKmh.toStringAsFixed(1)} km/h) — possibly running or in a vehicle';
      }
      // Rule 2: GPS teleport
      else if (seg.distanceM > _teleportDistanceM &&
          seg.elapsedSec < _teleportWindowSec) {
        isAnomaly = true;
        severity = AnomalySeverity.medium;
        reason =
            'Sudden location jump of ${seg.distanceM.toStringAsFixed(0)} m in ${seg.elapsedSec.toStringAsFixed(0)} s — possible GPS glitch';
      }
      // Rule 3: Statistical outlier (Z-score)
      else if (!isSmallMovement && zScore > _zScoreThreshold) {
        isAnomaly = true;
        severity = AnomalySeverity.low;
        reason =
            'Unusual movement spike (z=${zScore.toStringAsFixed(2)}, ${seg.speedKmh.toStringAsFixed(1)} km/h)';
      }

      if (isAnomaly) {
        results.add(AnomalyResult(
          deviceId: seg.deviceId,
          timestamp: seg.to.timestamp,
          location: seg.to.location,
          speedKmh: seg.speedKmh,
          distanceM: seg.distanceM,
          isAnomaly: true,
          severity: severity,
          reason: reason!,
        ));
      }
    }

    // ── Step 4: Device abandonment check ─────────────────────────────────
    final lastPoint = sorted.last;
    final minutesSinceLast =
        DateTime.now().difference(lastPoint.timestamp).inMinutes;

    if (minutesSinceLast >= _abandonmentMinutes) {
      results.add(AnomalyResult(
        deviceId: lastPoint.deviceId,
        timestamp: lastPoint.timestamp,
        location: lastPoint.location,
        speedKmh: 0,
        distanceM: 0,
        isAnomaly: true,
        severity: AnomalySeverity.medium,
        reason:
            'No movement detected for $minutesSinceLast minutes — device may have been left behind',
      ));
    }

    // Return most recent first
    return results..sort((a, b) => b.timestamp.compareTo(a.timestamp));
  }

  /// Summary stats for display (mean speed, max speed, total distance).
  static Map<String, double> stats(List<DeviceLocationPoint> history) {
    if (history.length < 2) return {'mean': 0, 'max': 0, 'totalKm': 0};

    final sorted = List<DeviceLocationPoint>.from(history)
      ..sort((a, b) => a.timestamp.compareTo(b.timestamp));

    double totalKm = 0;
    final List<double> speeds = [];

    for (int i = 1; i < sorted.length; i++) {
      final distM = _haversine(sorted[i - 1].location, sorted[i].location);
      final elapsedSec =
          sorted[i].timestamp.difference(sorted[i - 1].timestamp).inSeconds.toDouble();
      final speedKmh =
          elapsedSec > 0 ? (distM / 1000) / (elapsedSec / 3600) : 0.0;
      totalKm += distM / 1000;
      speeds.add(speedKmh);
    }

    final m = _mean(speeds);
    final maxS = speeds.reduce(max);
    return {'mean': m, 'max': maxS, 'totalKm': totalKm};
  }

  // ── Math helpers ──────────────────────────────────────────────────────────

  /// Haversine great-circle distance in metres.
  static double _haversine(LatLng a, LatLng b) {
    const r = 6371000.0; // Earth radius in metres
    final dLat = _deg2rad(b.latitude - a.latitude);
    final dLon = _deg2rad(b.longitude - a.longitude);
    final sinDLat = sin(dLat / 2);
    final sinDLon = sin(dLon / 2);
    final h = sinDLat * sinDLat +
        cos(_deg2rad(a.latitude)) *
            cos(_deg2rad(b.latitude)) *
            sinDLon *
            sinDLon;
    return 2 * r * asin(sqrt(h));
  }

  static double _deg2rad(double deg) => deg * pi / 180;

  static double _mean(List<double> values) {
    if (values.isEmpty) return 0;
    return values.reduce((a, b) => a + b) / values.length;
  }

  static double _std(List<double> values, double mean) {
    if (values.length < 2) return 0;
    final variance =
        values.map((v) => pow(v - mean, 2)).reduce((a, b) => a + b) /
            (values.length - 1);
    return sqrt(variance);
  }
}

// ── Internal helper ──────────────────────────────────────────────────────────

class _Segment {
  final String deviceId;
  final DeviceLocationPoint from;
  final DeviceLocationPoint to;
  final double distanceM;
  final double elapsedSec;
  final double speedKmh;

  const _Segment({
    required this.deviceId,
    required this.from,
    required this.to,
    required this.distanceM,
    required this.elapsedSec,
    required this.speedKmh,
  });
}
