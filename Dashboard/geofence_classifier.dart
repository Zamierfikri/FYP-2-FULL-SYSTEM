// lib/services/geofence_classifier.dart
//
// Geofence classification service.
//
// Priority:
//   1. POST to geofence_service.py (Random Forest model)
//   2. Geometric fallback (ray-casting + boundary distance) if service
//      is unreachable or returns an error
//
// Usage:
//   final status = await GeofenceClassifier.classify(point, polygonPoints);

import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:latlong2/latlong.dart';
import '../main.dart' show GeofenceStatus, LocationHelpers;

class GeofenceClassifier {
  // ── Configuration ───────────────────────────────────────────────────────────

  /// URL of the running geofence_service.py Flask server.
  static const String _serviceUrl = 'http://localhost:5001';

  /// HTTP timeout — keep short so the UI stays responsive.
  static const Duration _timeout = Duration(seconds: 3);

  // ── Public API ──────────────────────────────────────────────────────────────

  /// Classify [point] against [polygon].
  ///
  /// Pass the fused motion features from fusion_data for best RF accuracy.
  /// Falls back to [LocationHelpers.geofenceStatus] if the service is
  /// unavailable or returns an error.
  static Future<GeofenceStatus> classify(
    LatLng point,
    List<LatLng> polygon, {
    double vx = 0.0,
    double vy = 0.0,
    double speed = 0.0,
    double ax = 0.0,
    double ay = 0.0,
    double az = 9.8,
    double direction = 0.0,
  }) async {
    if (polygon.length < 3) return GeofenceStatus.outside;

    try {
      final body = jsonEncode({
        'lat': point.latitude,
        'lon': point.longitude,
        'polygon': polygon
            .map((p) => {'lat': p.latitude, 'lon': p.longitude})
            .toList(),
        // fused motion features — improves RF model accuracy
        'vx': vx,
        'vy': vy,
        'speed': speed,
        'ax': ax,
        'ay': ay,
        'az': az,
        'direction': direction,
      });

      final response = await http
          .post(
            Uri.parse('$_serviceUrl/classify'),
            headers: {'Content-Type': 'application/json'},
            body: body,
          )
          .timeout(_timeout);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final statusStr = data['status'] as String? ?? '';
        return _parseStatus(statusStr);
      }
    } catch (e) {
      // Service unreachable or timed out — use geometric fallback silently.
      if (kDebugMode) {
        debugPrint('[GeofenceClassifier] Service unavailable, using geometric fallback: $e');
      }
    }

    // Geometric fallback
    return LocationHelpers.geofenceStatus(point, polygon);
  }

  /// Batch classify multiple [points] against a single [polygon].
  ///
  /// More efficient than calling [classify] in a loop when using the RF service,
  /// since it sends one HTTP request for all devices.
  static Future<Map<String, GeofenceStatus>> classifyBatch({
    required Map<String, LatLng> devicePoints, // deviceId → LatLng
    required List<LatLng> polygon,
  }) async {
    if (polygon.length < 3) {
      return {for (final id in devicePoints.keys) id: GeofenceStatus.outside};
    }

    // Try the batch endpoint first
    try {
      final body = jsonEncode({
        'points': devicePoints.entries
            .map((e) => {
                  'device_id': e.key,
                  'lat': e.value.latitude,
                  'lon': e.value.longitude,
                })
            .toList(),
        'polygon': polygon
            .map((p) => {'lat': p.latitude, 'lon': p.longitude})
            .toList(),
      });

      final response = await http
          .post(
            Uri.parse('$_serviceUrl/classify/batch'),
            headers: {'Content-Type': 'application/json'},
            body: body,
          )
          .timeout(_timeout);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final results = data['results'] as List<dynamic>;
        final statusMap = <String, GeofenceStatus>{};
        for (final r in results) {
          final id = r['device_id'] as String? ?? '';
          final statusStr = r['status'] as String? ?? '';
          statusMap[id] = _parseStatus(statusStr);
        }
        return statusMap;
      }
    } catch (e) {
      if (kDebugMode) {
        debugPrint('[GeofenceClassifier] Batch service unavailable, using geometric fallback: $e');
      }
    }

    // Geometric fallback for all devices
    return {
      for (final entry in devicePoints.entries)
        entry.key: LocationHelpers.geofenceStatus(entry.value, polygon)
    };
  }

  /// Check if the geofence service is reachable.
  static Future<bool> isServiceAvailable() async {
    try {
      final response = await http
          .get(Uri.parse('$_serviceUrl/health'))
          .timeout(_timeout);
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  static GeofenceStatus _parseStatus(String s) {
    switch (s) {
      case 'inside':       return GeofenceStatus.inside;
      case 'nearBoundary': return GeofenceStatus.nearBoundary;
      default:             return GeofenceStatus.outside;
    }
  }
}
