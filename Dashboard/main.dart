import 'dart:async';
import 'dart:math' as dm;
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'firebase_options.dart';
import 'models/anomaly_detector.dart';
import 'services/geofence_classifier.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tracking System',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.light,
        ),
        cardTheme: CardThemeData(
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      home: const MainNavigation(),
    );
  }
}

// ─────────────────────────────────────────────
// MODELS
// ─────────────────────────────────────────────

class DeviceInfo {
  final String id;
  final String name;
  final Color color;
  bool isVisible;

  DeviceInfo({
    required this.id,
    required this.name,
    required this.color,
    this.isVisible = true,
  });
}

class DeviceLocation {
  final String deviceId;
  final LatLng location;
  final DateTime timestamp;

  DeviceLocation({
    required this.deviceId,
    required this.location,
    required this.timestamp,
  });
}

class GeofenceInfo {
  final String docId;
  final String name;
  final Color color;
  final IconData icon;
  // Optional hardcoded boundary — if set, Firestore is not needed
  final List<LatLng>? hardcodedPoints;

  const GeofenceInfo({
    required this.docId,
    required this.name,
    required this.color,
    required this.icon,
    this.hardcodedPoints,
  });
}

// ─────────────────────────────────────────────
// SHARED DEVICE LIST (app-wide singleton)
// ─────────────────────────────────────────────

class AppState extends ChangeNotifier {
  static final AppState _instance = AppState._internal();
  factory AppState() => _instance;
  AppState._internal();

  late List<DeviceInfo> devices = [
    DeviceInfo(id: 'dtestt1', name: 'Alia',   color: Colors.blue,   isVisible: true),
    DeviceInfo(id: 'dtestt2', name: 'Ali',    color: Colors.red,    isVisible: true),
    DeviceInfo(id: 'node3',   name: 'Danial', color: Colors.green,  isVisible: true),
    DeviceInfo(id: 'node4',   name: 'Sarah',  color: Colors.orange, isVisible: true),
    DeviceInfo(id: 'node5',   name: 'Farah',  color: Colors.purple, isVisible: true),
    DeviceInfo(id: 'node6',   name: 'Rizqi',  color: Colors.pink,   isVisible: true),
    DeviceInfo(id: 'node7',   name: 'Hana',   color: Colors.teal,   isVisible: true),
    DeviceInfo(id: 'node8',   name: 'Faris',  color: Colors.cyan,   isVisible: true),
    DeviceInfo(id: 'node9',   name: 'Umar',   color: Colors.amber,  isVisible: true),
    DeviceInfo(id: 'node10',  name: 'Julia',  color: Colors.indigo, isVisible: true),
  ];

  // 🔧 CONFIGURE YOUR GEOFENCES HERE
  final List<GeofenceInfo> geofences = [
    GeofenceInfo(
      docId: 'geofence_kict',
      name: 'KICT',
      color: Colors.green,
      icon: Icons.park,
      // Coordinates from kictboundary.geojson (GeoJSON [lng,lat] → LatLng(lat,lng))
      hardcodedPoints: [
        LatLng(3.2540636290594733, 101.72883886407192),
        LatLng(3.2536255090355723, 101.72890155371113),
        LatLng(3.2531731641349637, 101.72952560057178),
        LatLng(3.2534149837611466, 101.73003851580103),
        LatLng(3.253946986733922,  101.73029782294412),
        LatLng(3.2545273532931986, 101.73023513330492),
        LatLng(3.2547691725935834, 101.7296766256128),
        LatLng(3.254498903959231,  101.72954554727568),
        LatLng(3.2542001859109178, 101.72922070096405),
        LatLng(3.2540636290594733, 101.72883886407192),
      ],
    ),
  ];

  void toggleDevice(int index) {
    devices[index].isVisible = !devices[index].isVisible;
    notifyListeners();
  }

  void notifyUpdate() => notifyListeners();
}

// ─────────────────────────────────────────────
// SHARED HELPERS
// ─────────────────────────────────────────────

// ── Geofence proximity status ────────────────────────────────────────────────
enum GeofenceStatus { inside, nearBoundary, outside }

class LocationHelpers {
  static bool isInsideGeofence(LatLng point, List<LatLng> geofence) {
    if (geofence.length < 3) return false;
    bool inside = false;
    int j = geofence.length - 1;
    for (int i = 0; i < geofence.length; i++) {
      if ((geofence[i].latitude > point.latitude) !=
              (geofence[j].latitude > point.latitude) &&
          (point.longitude <
              (geofence[j].longitude - geofence[i].longitude) *
                      (point.latitude - geofence[i].latitude) /
                      (geofence[j].latitude - geofence[i].latitude) +
                  geofence[i].longitude)) {
        inside = !inside;
      }
      j = i;
    }
    return inside;
  }

  /// Minimum distance in metres from [point] to any edge of the polygon.
  /// Uses equirectangular projection — accurate enough at sub-km scales.
  static double distanceToPolygonBoundary(LatLng point, List<LatLng> polygon) {
    if (polygon.isEmpty) return double.infinity;
    double minDist = double.infinity;
    const double R = 6371000; // Earth radius in metres
    double toRad(double d) => d * 3.141592653589793 / 180.0;

    final cosLatProper = _cos(point.latitude);
    final lat0 = toRad(point.latitude);
    final lon0 = toRad(point.longitude);

    double px = lon0 * R * cosLatProper;
    double py = lat0 * R;

    int n = polygon.length;
    for (int i = 0; i < n; i++) {
      final a = polygon[i];
      final b = polygon[(i + 1) % n];
      double ax = toRad(a.longitude) * R * cosLatProper;
      double ay = toRad(a.latitude) * R;
      double bx = toRad(b.longitude) * R * cosLatProper;
      double by = toRad(b.latitude) * R;

      double dx = bx - ax, dy = by - ay;
      double lenSq = dx * dx + dy * dy;
      double t = lenSq == 0 ? 0 : ((px - ax) * dx + (py - ay) * dy) / lenSq;
      t = t < 0 ? 0 : (t > 1 ? 1 : t);
      double cx = ax + t * dx, cy = ay + t * dy;
      double dist = ((px - cx) * (px - cx) + (py - cy) * (py - cy));
      if (dist < minDist) minDist = dist;
    }
    return minDist < 0 ? 0 : minDist < double.infinity ? _sqrt(minDist) : double.infinity;
  }

  static double _cos(double deg) {
    final r = deg * 3.141592653589793 / 180.0;
    // Taylor series cos(r) – sufficient for latitudes near 0-90°
    double r2 = r * r;
    return 1 - r2 / 2 + r2 * r2 / 24 - r2 * r2 * r2 / 720;
  }

  static double _sqrt(double x) {
    if (x <= 0) return 0;
    double g = x / 2;
    for (int i = 0; i < 20; i++) {
      g = (g + x / g) / 2;
    }
    return g;
  }

  /// Returns [GeofenceStatus] for a point relative to a polygon.
  /// [nearThresholdMetres] defaults to 5 m.
  static GeofenceStatus geofenceStatus(
      LatLng point, List<LatLng> polygon,
      {double nearThresholdMetres = 5.0}) {
    if (polygon.length < 3) return GeofenceStatus.outside;
    final inside = isInsideGeofence(point, polygon);
    if (!inside) return GeofenceStatus.outside;
    final dist = distanceToPolygonBoundary(point, polygon);
    return dist <= nearThresholdMetres
        ? GeofenceStatus.nearBoundary
        : GeofenceStatus.inside;
  }

  /// Convenience: colour for a [GeofenceStatus].
  static Color statusColor(GeofenceStatus status) {
    switch (status) {
      case GeofenceStatus.inside:       return Colors.green;
      case GeofenceStatus.nearBoundary: return Colors.amber;
      case GeofenceStatus.outside:      return Colors.red;
    }
  }

  static DateTime? parseIsoTrimNanos(String raw) {
    try {
      final hasZ = raw.endsWith('Z');
      var s = hasZ ? raw.substring(0, raw.length - 1) : raw;
      if (s.contains('.')) {
        final parts = s.split('.');
        final head = parts[0];
        final tail = parts[1];
        final digits = RegExp(r'^\d+').firstMatch(tail)?.group(0) ?? '';
        final frac = digits.length > 6 ? digits.substring(0, 6) : digits;
        s = '$head.$frac';
      }
      final fixed = hasZ ? '${s}Z' : s;
      return DateTime.parse(fixed);
    } catch (_) {
      return null;
    }
  }

  static List<LatLng> parseCoordinates(Map<String, dynamic> data) {
    final coordinates = data['coordinates'];
    if (coordinates is! List || coordinates.isEmpty) return [];
    final points = <LatLng>[];
    for (final item in coordinates) {
      try {
        if (item is GeoPoint) {
          points.add(LatLng(item.latitude, item.longitude));
        } else if (item is String) {
          final regex = RegExp(r'([-+]?[0-9]*\.?[0-9]+)');
          final matches = regex.allMatches(item).toList();
          if (matches.length >= 2) {
            points.add(LatLng(
              double.parse(matches[0].group(0)!),
              double.parse(matches[1].group(0)!),
            ));
          }
        } else if (item is Map) {
          final lat = item['latitude'] ?? item['lat'];
          final lng = item['longitude'] ?? item['lng'];
          if (lat != null && lng != null) {
            points.add(LatLng(
              (lat is num) ? lat.toDouble() : double.parse(lat.toString()),
              (lng is num) ? lng.toDouble() : double.parse(lng.toString()),
            ));
          }
        }
      } catch (_) {}
    }
    return points;
  }

  static double haversineMetres(LatLng a, LatLng b) {
    const double r = 6371000.0;
    final dLat = (b.latitude - a.latitude) * dm.pi / 180.0;
    final dLon = (b.longitude - a.longitude) * dm.pi / 180.0;
    final sinHLat = dm.sin(dLat / 2);
    final sinHLon = dm.sin(dLon / 2);
    final h = sinHLat * sinHLat +
        dm.cos(a.latitude * dm.pi / 180.0) *
            dm.cos(b.latitude * dm.pi / 180.0) *
            sinHLon * sinHLon;
    return 2 * r * dm.asin(dm.sqrt(h.clamp(0.0, 1.0)));
  }

  static Stream<DeviceLocation?> deviceLocationStream(String deviceId) {
    return FirebaseFirestore.instance
        .collection('ttn_uplinks')
        .doc(deviceId)
        .collection('events')
        .orderBy('serverTime', descending: true)
        .limit(1)
        .snapshots()
        .map((querySnapshot) {
      if (querySnapshot.docs.isEmpty) return null;
      final data = querySnapshot.docs.first.data();
      final decoded = data['decoded'];
      if (decoded == null || decoded is! Map) return null;
      final lat = decoded['lat'] ?? decoded['latitude'];
      final lon = decoded['lon'] ?? decoded['longitude'];
      if (lat == null || lon == null) return null;
      final location = LatLng(
        lat is num ? lat.toDouble() : double.tryParse(lat.toString()) ?? 0.0,
        lon is num ? lon.toDouble() : double.tryParse(lon.toString()) ?? 0.0,
      );
      DateTime? timestamp;
      final serverTime = data['serverTime'];
      if (serverTime is Timestamp) {
        timestamp = serverTime.toDate().toLocal();
      } else {
        final receivedAt = data['receivedAt'];
        if (receivedAt is String) {
          timestamp = parseIsoTrimNanos(receivedAt)?.toLocal();
        }
      }
      return DeviceLocation(deviceId: deviceId, location: location, timestamp: timestamp ?? DateTime.now());
    });
  }

  static Stream<List<DeviceLocation>> combinedDeviceStream(List<DeviceInfo> visibleDevices) {
    final visible = visibleDevices.where((d) => d.isVisible).toList();
    if (visible.isEmpty) return Stream.value([]);

    // Cache-based approach: each device has its own independent listener.
    // When ANY device sends new data the map refreshes with ALL cached locations.
    // This is correct for live IoT devices that send at different times.
    final cache = <String, DeviceLocation?>{};
    final subs  = <StreamSubscription>[];
    late StreamController<List<DeviceLocation>> controller;

    controller = StreamController<List<DeviceLocation>>(
      onListen: () {
        for (final device in visible) {
          final sub = deviceLocationStream(device.id).listen((loc) {
            cache[device.id] = loc;
            if (!controller.isClosed) {
              controller.add(cache.values.whereType<DeviceLocation>().toList());
            }
          });
          subs.add(sub);
        }
      },
      onCancel: () {
        for (final sub in subs) { sub.cancel(); }
      },
    );

    return controller.stream;
  }
}

// ─────────────────────────────────────────────
// MAIN NAVIGATION
// ─────────────────────────────────────────────

class MainNavigation extends StatefulWidget {
  const MainNavigation({super.key});

  @override
  State<MainNavigation> createState() => _MainNavigationState();
}

class _MainNavigationState extends State<MainNavigation> {
  int _currentIndex = 0;
  final AppState _appState = AppState();

  // Shared active geofence for Map page (set from Safe Zones page)
  String? _activeGeofenceDocId;

  void _navigateToGeofenceOnMap(String docId) {
    setState(() {
      _activeGeofenceDocId = docId;
      _currentIndex = 0; // go to Map tab
    });
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      LiveMapPage(
        appState: _appState,
        activeGeofenceDocId: _activeGeofenceDocId,
        onGeofenceViewed: () => setState(() => _activeGeofenceDocId = null),
        onViewAnomalies: () => setState(() => _currentIndex = 4),
      ),
      SafeZonesPage(
        appState: _appState,
        onViewOnMap: _navigateToGeofenceOnMap,
      ),
      HistoryPage(appState: _appState),
      DevicesPage(appState: _appState),
      AnomaliesPage(appState: _appState),
    ];

    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: pages,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (i) => setState(() => _currentIndex = i),
        backgroundColor: Colors.white,
        indicatorColor: Colors.deepPurple.shade100,
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.map_outlined),
            selectedIcon: Icon(Icons.map, color: Colors.deepPurple),
            label: 'Map',
          ),
          NavigationDestination(
            icon: Icon(Icons.shield_outlined),
            selectedIcon: Icon(Icons.shield, color: Colors.deepPurple),
            label: 'Safe Zones',
          ),
          NavigationDestination(
            icon: Icon(Icons.history_outlined),
            selectedIcon: Icon(Icons.history, color: Colors.deepPurple),
            label: 'History',
          ),
          NavigationDestination(
            icon: Icon(Icons.devices_outlined),
            selectedIcon: Icon(Icons.devices, color: Colors.deepPurple),
            label: 'Devices',
          ),
          NavigationDestination(
            icon: Icon(Icons.analytics_outlined),
            selectedIcon: Icon(Icons.analytics, color: Colors.deepPurple),
            label: 'Anomalies',
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
// TAB 1: LIVE MAP PAGE
// ─────────────────────────────────────────────

class LiveMapPage extends StatefulWidget {
  final AppState appState;
  final String? activeGeofenceDocId;
  final VoidCallback? onGeofenceViewed;
  final VoidCallback? onViewAnomalies;

  const LiveMapPage({
    super.key,
    required this.appState,
    this.activeGeofenceDocId,
    this.onGeofenceViewed,
    this.onViewAnomalies,
  });

  @override
  State<LiveMapPage> createState() => _LiveMapPageState();
}

class _LiveMapPageState extends State<LiveMapPage> {
  final MapController _mapController = MapController();
  bool _isLoading = true;
  String? _errorMessage;
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  // Which geofences are visible on map (all by default)
  late Set<String> _visibleGeofenceIds;

  // RF geofence status per device — updated from fusion_data in real-time.
  // Falls back to geometric check when not yet populated.
  final Map<String, GeofenceStatus> _rfGeofenceStatus = {};
  final List<StreamSubscription> _fusionSubs = [];
  final List<StreamSubscription> _anomalySubs = [];

  // Dart-side speed anomaly detection (works without anomaly_service.py)
  final List<StreamSubscription> _dartAnomalySubs = [];
  final Map<String, DateTime?> _lastAlertTime = {};
  final Map<String, DeviceLocation?> _prevDeviceLocation = {};
  bool _dartAnomalyListenerReady = false;

  @override
  void initState() {
    super.initState();
    _visibleGeofenceIds = widget.appState.geofences.map((g) => g.docId).toSet();
    _startFusionGeofenceListeners();
    _startAnomalyAlertListeners();
    _startDartAnomalyListeners();
  }

  @override
  void dispose() {
    for (final sub in _fusionSubs) { sub.cancel(); }
    for (final sub in _anomalySubs) { sub.cancel(); }
    for (final sub in _dartAnomalySubs) { sub.cancel(); }
    super.dispose();
  }

  /// Listens to the latest event in fusion_data for each device.
  /// Calls GeofenceClassifier with the fused motion features so the
  /// Random Forest model (geofence_service.py) produces the status,
  /// then caches the result in [_rfGeofenceStatus].
  void _startFusionGeofenceListeners() {
    for (final device in widget.appState.devices) {
      final sub = FirebaseFirestore.instance
          .collection('fusion_data')
          .doc(device.id)
          .collection('events')
          .orderBy('serverTime', descending: true)
          .limit(1)
          .snapshots()
          .listen((snap) async {
        if (snap.docs.isEmpty) return;
        final data = snap.docs.first.data();

        final lat = (data['lat'] as num?)?.toDouble();
        final lon = (data['lon'] as num?)?.toDouble();
        if (lat == null || lon == null || lat == 0.0 && lon == 0.0) return;

        final point = LatLng(lat, lon);

        // Determine worst status across all geofences for this device
        GeofenceStatus worstStatus = GeofenceStatus.outside;
        for (final g in widget.appState.geofences) {
          final pts = g.hardcodedPoints ?? [];
          if (pts.length < 3) continue;

          final status = await GeofenceClassifier.classify(
            point,
            pts,
            vx:        (data['fused_vx']              as num?)?.toDouble() ?? 0.0,
            vy:        (data['fused_vy']              as num?)?.toDouble() ?? 0.0,
            speed:     (data['fused_speed']           as num?)?.toDouble() ?? 0.0,
            ax:        (data['ax']                    as num?)?.toDouble() ?? 0.0,
            ay:        (data['ay']                    as num?)?.toDouble() ?? 0.0,
            az:        (data['az']                    as num?)?.toDouble() ?? 9.8,
            direction: (data['movement_direction']    as num?)?.toDouble() ?? 0.0,
          );

          if (status == GeofenceStatus.inside) {
            worstStatus = GeofenceStatus.inside;
            break;
          } else if (status == GeofenceStatus.nearBoundary) {
            worstStatus = GeofenceStatus.nearBoundary;
          }
        }

        if (mounted) {
          setState(() => _rfGeofenceStatus[device.id] = worstStatus);
        }
      });
      _fusionSubs.add(sub);
    }
  }

  /// Listens to anomaly_results/{deviceId}/live_alerts for new Isolation
  /// Forest anomalies written by anomaly_service.py. Skips documents that
  /// existed before the app started so only new alerts trigger the banner.
  void _startAnomalyAlertListeners() {
    for (final device in widget.appState.devices) {
      // Watch only the newest alert (limit 1). On first snapshot we record
      // the current newest doc id as baseline; any later change to a new id
      // means a fresh anomaly was written → fire the popup.
      String? latestId;
      bool baselineSet = false;

      final sub = FirebaseFirestore.instance
          .collection('anomaly_results')
          .doc(device.id)
          .collection('live_alerts')
          .orderBy('detectedAt', descending: true)
          .limit(1)
          .snapshots()
          .listen((snap) {
        if (snap.docs.isEmpty) {
          baselineSet = true;
          return;
        }
        final newestId = snap.docs.first.id;
        if (!baselineSet) {
          latestId = newestId;
          baselineSet = true;
          debugPrint('[AnomalyAlert] ${device.id} baseline=$newestId');
          return;
        }
        if (newestId != latestId) {
          latestId = newestId;
          debugPrint('[AnomalyAlert] ${device.id} NEW anomaly → popup');
          if (mounted) _showAnomalyAlert(device);
        }
      }, onError: (e) {
        debugPrint('[AnomalyAlert] listener error for ${device.id}: $e');
      });
      _anomalySubs.add(sub);
    }
  }

  void _showAnomalyAlert(DeviceInfo device, {String? reason}) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          backgroundColor: Colors.deepOrange.shade700,
          duration: const Duration(seconds: 8),
          behavior: SnackBarBehavior.floating,
          margin: const EdgeInsets.all(16),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          content: Row(
            children: [
              const Icon(Icons.warning_amber_rounded, color: Colors.white, size: 22),
              const SizedBox(width: 10),
              Expanded(
                child: Text('Anomaly detected — ${device.name}',
                    style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 14)),
              ),
            ],
          ),
          action: SnackBarAction(
            label: 'VIEW',
            textColor: Colors.white,
            onPressed: () => widget.onViewAnomalies?.call(),
          ),
        ),
      );
  }

  /// Dart-side real-time anomaly detection from ttn_uplinks.
  /// Triggers a map snackbar when a device exceeds 25 km/h (vehicle-level
  /// speed). Works entirely without anomaly_service.py running.
  void _startDartAnomalyListeners() {
    // Wait 5 s after app start so that the first snapshot (historical data)
    // is stored as a baseline before we start comparing consecutive points.
    Future.delayed(const Duration(seconds: 5), () {
      if (mounted) _dartAnomalyListenerReady = true;
    });

    for (final device in widget.appState.devices) {
      final sub = LocationHelpers.deviceLocationStream(device.id).listen((current) {
        if (!_dartAnomalyListenerReady || current == null) return;

        final prev = _prevDeviceLocation[device.id];
        _prevDeviceLocation[device.id] = current;

        if (prev == null) return; // need at least 2 points

        final distM = LocationHelpers.haversineMetres(prev.location, current.location);
        final elapsedSec =
            current.timestamp.difference(prev.timestamp).inSeconds.toDouble();
        if (elapsedSec <= 0) return;

        final speedKmh = (distM / 1000.0) / (elapsedSec / 3600.0);

        // 10 km/h — too fast for walking pace, triggers the live popup alert
        if (speedKmh > 10.0 && mounted) {
          // One alert per device per 60 s to avoid spamming
          final last = _lastAlertTime[device.id];
          final now = DateTime.now();
          if (last == null || now.difference(last).inSeconds >= 60) {
            _lastAlertTime[device.id] = now;
            _showAnomalyAlert(
              device,
              reason:
                  'High speed detected — ${speedKmh.toStringAsFixed(1)} km/h '
                  '(may be in a vehicle)',
            );
          }
        }
      });
      _dartAnomalySubs.add(sub);
    }
  }

  @override
  void didUpdateWidget(LiveMapPage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.activeGeofenceDocId != null &&
        widget.activeGeofenceDocId != oldWidget.activeGeofenceDocId) {
      // Show only this geofence and fly to it
      setState(() {
        _visibleGeofenceIds = {widget.activeGeofenceDocId!};
      });
      widget.onGeofenceViewed?.call();
    }
  }

  Stream<Map<String, List<LatLng>>> get allGeofencesStream {
    // If all geofences have hardcoded points, skip Firestore entirely
    final allHardcoded = widget.appState.geofences.every(
      (g) => g.hardcodedPoints != null && g.hardcodedPoints!.isNotEmpty,
    );

    if (allHardcoded) {
      WidgetsBinding.instance.addPostFrameCallback(
        (_) => setState(() => _isLoading = false),
      );
      return Stream.value({
        for (final g in widget.appState.geofences)
          g.docId: g.hardcodedPoints!,
      });
    }

    // Fallback: fetch from Firestore for geofences without hardcoded points
    final streams = widget.appState.geofences.map((g) {
      if (g.hardcodedPoints != null && g.hardcodedPoints!.isNotEmpty) {
        return Stream.value(MapEntry(g.docId, g.hardcodedPoints!));
      }
      return FirebaseFirestore.instance
          .collection('geofence')
          .doc(g.docId)
          .snapshots()
          .map((snap) {
        if (!snap.exists || snap.data() == null) return MapEntry(g.docId, <LatLng>[]);
        final pts = LocationHelpers.parseCoordinates(snap.data()!);
        return MapEntry(g.docId, pts);
      });
    }).toList();

    if (streams.isEmpty) return Stream.value({});

    return streams.first.asyncMap((firstEntry) async {
      final result = <String, List<LatLng>>{firstEntry.key: firstEntry.value};
      setState(() => _isLoading = false);
      for (int i = 1; i < streams.length; i++) {
        final entry = await streams[i].first;
        result[entry.key] = entry.value;
      }
      return result;
    });
  }

  void _fitToGeofence(List<LatLng> points) {
    if (points.length < 2) return;
    _mapController.fitCamera(
      CameraFit.bounds(
        bounds: LatLngBounds.fromPoints(points),
        padding: const EdgeInsets.all(50),
      ),
    );
  }

  void _centerOnDevice(LatLng location) {
    _mapController.move(location, 17);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.deepPurple.shade50, Colors.white],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              _buildHeader(),
              Expanded(
                child: StreamBuilder<Map<String, List<LatLng>>>(
                  stream: allGeofencesStream,
                  builder: (context, snapshot) {
                    if (_isLoading) return _buildLoadingState();
                    if (_errorMessage != null) return _buildErrorState();
                    final geofenceMap = snapshot.data ?? {};
                    return _buildMapView(geofenceMap);
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.fromLTRB(8, 12, 16, 12),
      child: Row(
        children: [
          Container(
            margin: const EdgeInsets.only(left: 8),
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.deepPurple,
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.my_location, color: Colors.white, size: 26),
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Tracking System',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.deepPurple)),
                Text('Live Monitoring', style: TextStyle(fontSize: 12, color: Colors.grey)),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: Colors.green.shade100,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              children: [
                Container(width: 7, height: 7, decoration: const BoxDecoration(color: Colors.green, shape: BoxShape.circle)),
                const SizedBox(width: 5),
                const Text('Live', style: TextStyle(color: Colors.green, fontWeight: FontWeight.w600, fontSize: 12)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLoadingState() => const Center(
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          CircularProgressIndicator(color: Colors.deepPurple),
          SizedBox(height: 16),
          Text('Loading safe zones...'),
        ]),
      );

  Widget _buildErrorState() => Center(
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(Icons.error_outline, size: 64, color: Colors.orange.shade400),
          const SizedBox(height: 16),
          Text(_errorMessage ?? 'Something went wrong'),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: () => setState(() { _isLoading = true; _errorMessage = null; }),
            icon: const Icon(Icons.refresh),
            label: const Text('Retry'),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.deepPurple, foregroundColor: Colors.white),
          ),
        ]),
      );

  Widget _buildMapView(Map<String, List<LatLng>> geofenceMap) {
    final allPoints = geofenceMap.values.expand((pts) => pts).toList();
    final center = allPoints.isNotEmpty ? allPoints.first : const LatLng(3.1579, 101.7123);
    final visibleDevices = widget.appState.devices.where((d) => d.isVisible).toList();

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          _buildStatsCard(geofenceMap, visibleDevices),
          const SizedBox(height: 12),
          // Geofence filter chips
          SizedBox(
            height: 38,
            child: ListView(
              scrollDirection: Axis.horizontal,
              children: widget.appState.geofences.map((g) {
                final isOn = _visibleGeofenceIds.contains(g.docId);
                return Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: FilterChip(
                    avatar: Icon(g.icon, size: 16, color: isOn ? Colors.white : g.color),
                    label: Text(g.name, style: TextStyle(fontSize: 12, color: isOn ? Colors.white : g.color, fontWeight: FontWeight.w600)),
                    selected: isOn,
                    onSelected: (_) {
                      setState(() {
                        if (_visibleGeofenceIds.contains(g.docId)) {
                          _visibleGeofenceIds.remove(g.docId);
                        } else {
                          _visibleGeofenceIds.add(g.docId);
                        }
                      });
                    },
                    selectedColor: g.color,
                    backgroundColor: g.color.withValues(alpha: 0.1),
                    side: BorderSide(color: g.color),
                    showCheckmark: false,
                  ),
                );
              }).toList(),
            ),
          ),
          const SizedBox(height: 12),
          Expanded(
            child: Card(
              clipBehavior: Clip.antiAlias,
              child: Stack(
                children: [
                  StreamBuilder<List<DeviceLocation>>(
                    stream: LocationHelpers.combinedDeviceStream(visibleDevices),
                    builder: (context, deviceSnapshot) {
                      final deviceLocations = deviceSnapshot.data ?? [];

                      // Determine worst status across all visible devices for fill colour
                      final polygons = <Polygon>[];
                      for (final g in widget.appState.geofences) {
                        if (!_visibleGeofenceIds.contains(g.docId)) continue;
                        final pts = geofenceMap[g.docId] ?? [];
                        if (pts.length >= 3) {
                          // Pick worst device status to colour the polygon fill
                          GeofenceStatus worstStatus = GeofenceStatus.inside;
                          if (deviceLocations.isNotEmpty) {
                            for (final dl in deviceLocations) {
                              // Use RF-cached status from fusion_data if available,
                              // else fall back to geometric check
                              final s = _rfGeofenceStatus[dl.deviceId] ??
                                  LocationHelpers.geofenceStatus(dl.location, pts);
                              if (s == GeofenceStatus.outside) {
                                worstStatus = GeofenceStatus.outside;
                                break;
                              } else if (s == GeofenceStatus.nearBoundary) {
                                worstStatus = GeofenceStatus.nearBoundary;
                              }
                            }
                          }
                          final fillColor = LocationHelpers.statusColor(worstStatus);
                          polygons.add(Polygon(
                            points: pts,
                            color: fillColor.withValues(alpha: 0.18),
                            borderColor: fillColor,
                            borderStrokeWidth: 3,
                          ));
                        }
                      }

                      return FlutterMap(
                        mapController: _mapController,
                        options: MapOptions(
                          initialCenter: center,
                          initialZoom: 16,
                          minZoom: 10,
                          maxZoom: 22,
                        ),
                        children: [
                          TileLayer(
                            urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                            userAgentPackageName: 'com.example.tracker',
                            maxZoom: 22,
                            maxNativeZoom: 19,
                          ),
                          if (polygons.isNotEmpty) PolygonLayer(polygons: polygons),
                          MarkerLayer(
                            markers: deviceLocations
                                .where((dl) => visibleDevices.any((d) => d.id == dl.deviceId))
                                .map<Marker>((dl) {
                              final device = visibleDevices.firstWhere((d) => d.id == dl.deviceId);
                              // Use RF-cached status (from fusion_data + geofence_service.py).
                              // Falls back to geometric check if RF result not yet cached.
                              GeofenceStatus status =
                                  _rfGeofenceStatus[dl.deviceId] ?? (() {
                                GeofenceStatus geo = GeofenceStatus.outside;
                                for (final g in widget.appState.geofences) {
                                  final pts = geofenceMap[g.docId] ?? [];
                                  final s = LocationHelpers.geofenceStatus(dl.location, pts);
                                  if (s == GeofenceStatus.inside) { geo = GeofenceStatus.inside; break; }
                                  if (s == GeofenceStatus.nearBoundary) geo = GeofenceStatus.nearBoundary;
                                }
                                return geo;
                              }());
                              return _buildDeviceMarker(device, dl, status);
                            }).toList(),
                          ),
                        ],
                      );
                    },
                  ),
                  // Map controls
                  Positioned(
                    top: 16,
                    right: 16,
                    child: Column(
                      children: [
                        _buildMapButton(
                          icon: Icons.center_focus_strong,
                          onPressed: () {
                            if (allPoints.isNotEmpty) _fitToGeofence(allPoints);
                          },
                          tooltip: 'Fit All Zones',
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Marker _buildDeviceMarker(DeviceInfo device, DeviceLocation deviceLocation, GeofenceStatus status) {
    final markerColor = LocationHelpers.statusColor(status);
    final isInside = status != GeofenceStatus.outside;
    return Marker(
      point: deviceLocation.location,
      width: 80,
      height: 80,
      child: GestureDetector(
        onTap: () {
          _centerOnDevice(deviceLocation.location);
          _showDeviceInfo(device, deviceLocation, isInside);
        },
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(color: device.color, borderRadius: BorderRadius.circular(12)),
              child: Text(device.name, style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 2),
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: markerColor,
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white, width: 3),
                boxShadow: [BoxShadow(color: markerColor.withValues(alpha: 0.5), blurRadius: 10, spreadRadius: 2)],
              ),
              child: const Icon(Icons.person, color: Colors.white, size: 22),
            ),
          ],
        ),
      ),
    );
  }

  void _showDeviceInfo(DeviceInfo device, DeviceLocation location, bool isInside) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        padding: const EdgeInsets.all(20),
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.only(topLeft: Radius.circular(20), topRight: Radius.circular(20)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(color: device.color.withValues(alpha: 0.2), shape: BoxShape.circle),
                  child: Icon(Icons.person, color: device.color, size: 28),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(device.name, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                      Text('ID: ${device.id}', style: TextStyle(fontSize: 12, color: Colors.grey.shade600)),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: isInside ? Colors.green.shade100 : Colors.red.shade100,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Row(
                    children: [
                      Icon(isInside ? Icons.check_circle : Icons.warning, size: 16, color: isInside ? Colors.green : Colors.red),
                      const SizedBox(width: 4),
                      Text(isInside ? 'SAFE' : 'ALERT',
                          style: TextStyle(color: isInside ? Colors.green : Colors.red, fontWeight: FontWeight.bold, fontSize: 12)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            _buildInfoRow(Icons.location_on, 'Location',
                '${location.location.latitude.toStringAsFixed(6)}, ${location.location.longitude.toStringAsFixed(6)}'),
            const SizedBox(height: 12),
            _buildInfoRow(Icons.access_time, 'Last Update',
                '${location.timestamp.hour.toString().padLeft(2, '0')}:${location.timestamp.minute.toString().padLeft(2, '0')}:${location.timestamp.second.toString().padLeft(2, '0')}'),
            const SizedBox(height: 20),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: () { Navigator.pop(context); _centerOnDevice(location.location); },
                icon: const Icon(Icons.center_focus_strong),
                label: const Text('Center on Map'),
                style: ElevatedButton.styleFrom(backgroundColor: device.color, foregroundColor: Colors.white, padding: const EdgeInsets.symmetric(vertical: 12)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(icon, size: 20, color: Colors.grey.shade600),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label, style: TextStyle(fontSize: 12, color: Colors.grey.shade600)),
              const SizedBox(height: 2),
              Text(value, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildStatsCard(Map<String, List<LatLng>> geofenceMap, List<DeviceInfo> visibleDevices) {
    return StreamBuilder<List<DeviceLocation>>(
      stream: LocationHelpers.combinedDeviceStream(visibleDevices),
      builder: (context, deviceSnapshot) {
        final deviceLocations = deviceSnapshot.data ?? [];
        final alertCount = deviceLocations.where((loc) {
          // Use RF-cached status to match what the map markers show.
          final status = _rfGeofenceStatus[loc.deviceId] ?? (() {
            for (final g in widget.appState.geofences) {
              final pts = geofenceMap[g.docId] ?? [];
              if (LocationHelpers.isInsideGeofence(loc.location, pts)) return GeofenceStatus.inside;
            }
            return GeofenceStatus.outside;
          }());
          return status == GeofenceStatus.outside;
        }).length;

        return Card(
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStatItem(Icons.warning_amber_rounded, '$alertCount', 'Alerts', alertCount > 0 ? Colors.red : Colors.green),
                Container(width: 1, height: 36, color: Colors.grey.shade300),
                _buildStatItem(Icons.devices, '${visibleDevices.length}', 'Visible', Colors.deepPurple),
                Container(width: 1, height: 36, color: Colors.grey.shade300),
                _buildStatItem(Icons.shield, '${widget.appState.geofences.length}', 'Zones', Colors.teal),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildStatItem(IconData icon, String value, String label, Color color) {
    return Column(
      children: [
        Icon(icon, color: color, size: 22),
        const SizedBox(height: 6),
        Text(value, style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color)),
        Text(label, style: TextStyle(fontSize: 11, color: Colors.grey.shade600)),
      ],
    );
  }

  Widget _buildMapButton({required IconData icon, required VoidCallback onPressed, required String tooltip}) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.1), blurRadius: 8)],
      ),
      child: IconButton(icon: Icon(icon), onPressed: onPressed, tooltip: tooltip, color: Colors.deepPurple, iconSize: 22),
    );
  }
}

// ─────────────────────────────────────────────
// TAB 2: SAFE ZONES PAGE
// ─────────────────────────────────────────────

class SafeZonesPage extends StatefulWidget {
  final AppState appState;
  final void Function(String docId) onViewOnMap;

  const SafeZonesPage({super.key, required this.appState, required this.onViewOnMap});

  @override
  State<SafeZonesPage> createState() => _SafeZonesPageState();
}

class _SafeZonesPageState extends State<SafeZonesPage> {
  bool _showDeviceStatus = false;

  Stream<List<LatLng>> _geofenceStream(String docId) {
    // Use hardcoded points if available — no Firestore needed
    final geofence = widget.appState.geofences.firstWhere(
      (g) => g.docId == docId,
      orElse: () => const GeofenceInfo(docId: '', name: '', color: Colors.transparent, icon: Icons.place),
    );
    if (geofence.hardcodedPoints != null && geofence.hardcodedPoints!.isNotEmpty) {
      return Stream.value(geofence.hardcodedPoints!);
    }
    // Fallback to Firestore
    return FirebaseFirestore.instance
        .collection('geofence')
        .doc(docId)
        .snapshots()
        .map((snap) {
      if (!snap.exists || snap.data() == null) return <LatLng>[];
      return LocationHelpers.parseCoordinates(snap.data()!);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.teal.shade50, Colors.white],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Header
              Container(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(color: Colors.teal, borderRadius: BorderRadius.circular(12)),
                      child: const Icon(Icons.shield, color: Colors.white, size: 26),
                    ),
                    const SizedBox(width: 12),
                    const Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Safe Zones', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.teal)),
                        Text('Landmark Geofences', style: TextStyle(fontSize: 12, color: Colors.grey)),
                      ],
                    ),
                    const Spacer(),
                    // Toggle device status panel
                    IconButton(
                      tooltip: _showDeviceStatus ? 'Hide device status' : 'Show device status',
                      icon: Icon(
                        _showDeviceStatus ? Icons.people : Icons.people_outline,
                        color: Colors.teal,
                      ),
                      onPressed: () => setState(() => _showDeviceStatus = !_showDeviceStatus),
                    ),
                  ],
                ),
              ),
              const Padding(
                padding: EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                child: Text(
                  'Tap "View on Map" to jump directly to any zone.',
                  style: TextStyle(fontSize: 12, color: Colors.grey),
                ),
              ),
              // ── Per-device geofence breach status panel ───────────────
              if (_showDeviceStatus) _buildDeviceBreachPanel(),
              const SizedBox(height: 8),
              Expanded(
                child: ListView.builder(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  itemCount: widget.appState.geofences.length,
                  itemBuilder: (context, index) {
                    final g = widget.appState.geofences[index];
                    return StreamBuilder<List<LatLng>>(
                      stream: _geofenceStream(g.docId),
                      builder: (context, snapshot) {
                        final pts = snapshot.data ?? [];
                        final configured = pts.length >= 3;

                        return Card(
                          margin: const EdgeInsets.only(bottom: 12),
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.all(12),
                                      decoration: BoxDecoration(
                                        color: g.color.withValues(alpha: 0.15),
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: Icon(g.icon, color: g.color, size: 28),
                                    ),
                                    const SizedBox(width: 16),
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          Text(g.name,
                                              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                                          Text('Doc: ${g.docId}',
                                              style: TextStyle(fontSize: 11, color: Colors.grey.shade500)),
                                        ],
                                      ),
                                    ),
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                                      decoration: BoxDecoration(
                                        color: configured ? Colors.green.shade50 : Colors.orange.shade50,
                                        borderRadius: BorderRadius.circular(12),
                                        border: Border.all(
                                          color: configured ? Colors.green.shade300 : Colors.orange.shade300,
                                        ),
                                      ),
                                      child: Text(
                                        configured ? 'Active' : 'Not set',
                                        style: TextStyle(
                                          fontSize: 12,
                                          fontWeight: FontWeight.w600,
                                          color: configured ? Colors.green.shade700 : Colors.orange.shade700,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                                if (configured) ...[
                                  const SizedBox(height: 12),
                                  Row(
                                    children: [
                                      Icon(Icons.place, size: 14, color: Colors.grey.shade500),
                                      const SizedBox(width: 4),
                                      Text('${pts.length} boundary points',
                                          style: TextStyle(fontSize: 12, color: Colors.grey.shade600)),
                                    ],
                                  ),
                                  const SizedBox(height: 8),
                                  // Per-device inline breach status
                                  _DeviceBreachRow(appState: widget.appState, geofencePoints: pts, geofenceColor: g.color),
                                ],
                                const SizedBox(height: 12),
                                Row(
                                  children: [
                                    Expanded(
                                      child: OutlinedButton.icon(
                                        onPressed: configured ? () => widget.onViewOnMap(g.docId) : null,
                                        icon: const Icon(Icons.map, size: 16),
                                        label: const Text('View on Map'),
                                        style: OutlinedButton.styleFrom(
                                          foregroundColor: g.color,
                                          side: BorderSide(color: g.color),
                                          padding: const EdgeInsets.symmetric(vertical: 10),
                                        ),
                                      ),
                                    ),
                                    const SizedBox(width: 8),
                                    Expanded(
                                      child: ElevatedButton.icon(
                                        onPressed: configured ? () {
                                          showModalBottomSheet(
                                            context: context,
                                            backgroundColor: Colors.transparent,
                                            builder: (_) => _GeofenceDetailSheet(geofence: g, points: pts),
                                          );
                                        } : null,
                                        icon: const Icon(Icons.info_outline, size: 16),
                                        label: const Text('Details'),
                                        style: ElevatedButton.styleFrom(
                                          backgroundColor: g.color,
                                          foregroundColor: Colors.white,
                                          padding: const EdgeInsets.symmetric(vertical: 10),
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Full-page device breach status panel ─────────────────────────────────
  Widget _buildDeviceBreachPanel() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 4),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.teal.shade50,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.teal.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.people, size: 16, color: Colors.teal.shade700),
              const SizedBox(width: 6),
              Text('Live Device Status',
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Colors.teal.shade800)),
            ],
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: widget.appState.devices.map((device) {
              return StreamBuilder<DeviceLocation?>(
                stream: LocationHelpers.deviceLocationStream(device.id),
                builder: (context, snap) {
                  final loc = snap.data;
                  if (loc == null) {
                    return Chip(
                      avatar: Icon(Icons.person, size: 14, color: Colors.grey.shade400),
                      label: Text(device.name,
                          style: TextStyle(fontSize: 11, color: Colors.grey.shade500)),
                      backgroundColor: Colors.grey.shade100,
                      padding: EdgeInsets.zero,
                      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    );
                  }
                  // Check against all configured geofences (stream-based — use
                  // the latest cached data we already have from the card above)
                  return FutureBuilder<bool>(
                    future: _isDeviceInAnyGeofence(loc.location),
                    builder: (context, inSnap) {
                      final inside = inSnap.data ?? false;
                      return Chip(
                        avatar: Icon(Icons.person, size: 14,
                            color: inside ? Colors.green.shade700 : Colors.red.shade600),
                        label: Text(device.name,
                            style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w600,
                              color: inside ? Colors.green.shade800 : Colors.red.shade700,
                            )),
                        backgroundColor: inside ? Colors.green.shade50 : Colors.red.shade50,
                        side: BorderSide(color: inside ? Colors.green.shade300 : Colors.red.shade300),
                        padding: EdgeInsets.zero,
                        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                      );
                    },
                  );
                },
              );
            }).toList(),
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              _legendDot(Colors.green, 'Inside zone'),
              const SizedBox(width: 10),
              _legendDot(Colors.amber, 'Near boundary'),
              const SizedBox(width: 10),
              _legendDot(Colors.red, 'Outside zone'),
              const SizedBox(width: 10),
              _legendDot(Colors.grey, 'No data'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _legendDot(Color color, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(width: 8, height: 8, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(fontSize: 10, color: Colors.grey.shade600)),
      ],
    );
  }

  Future<bool> _isDeviceInAnyGeofence(LatLng point) async {
    for (final g in widget.appState.geofences) {
      // Use hardcoded points if available — no Firestore needed
      if (g.hardcodedPoints != null && g.hardcodedPoints!.isNotEmpty) {
        if (LocationHelpers.isInsideGeofence(point, g.hardcodedPoints!)) return true;
        continue;
      }
      // Fallback to Firestore
      final snap = await FirebaseFirestore.instance
          .collection('geofence')
          .doc(g.docId)
          .get();
      if (!snap.exists || snap.data() == null) continue;
      final pts = LocationHelpers.parseCoordinates(snap.data()!);
      if (LocationHelpers.isInsideGeofence(point, pts)) return true;
    }
    return false;
  }
}

// ── Per-device breach row inside each geofence card ─────────────────────────
class _DeviceBreachRow extends StatelessWidget {
  final AppState appState;
  final List<LatLng> geofencePoints;
  final Color geofenceColor;

  const _DeviceBreachRow({
    required this.appState,
    required this.geofencePoints,
    required this.geofenceColor,
  });

  @override
  Widget build(BuildContext context) {
    final devices = appState.devices.where((d) => d.isVisible).toList();
    if (devices.isEmpty) return const SizedBox.shrink();

    return Wrap(
      spacing: 6,
      runSpacing: 4,
      children: devices.map((device) {
        return StreamBuilder<DeviceLocation?>(
          stream: LocationHelpers.deviceLocationStream(device.id),
          builder: (context, snap) {
            final loc = snap.data;
            if (loc == null) return const SizedBox.shrink();
            final status = LocationHelpers.geofenceStatus(loc.location, geofencePoints);
            final statusColor = LocationHelpers.statusColor(status);
            final label = status == GeofenceStatus.inside
                ? '${device.name}: Inside'
                : status == GeofenceStatus.nearBoundary
                    ? '${device.name}: Near boundary'
                    : '${device.name}: Outside';
            final icon = status == GeofenceStatus.inside
                ? Icons.check_circle
                : status == GeofenceStatus.nearBoundary
                    ? Icons.warning_amber_rounded
                    : Icons.warning_rounded;
            return Tooltip(
              message: label,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: statusColor.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: statusColor.withValues(alpha: 0.6), width: 1),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(icon, size: 12, color: statusColor),
                    const SizedBox(width: 4),
                    Text(
                      device.name,
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: statusColor,
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      }).toList(),
    );
  }
}

class _GeofenceDetailSheet extends StatelessWidget {
  final GeofenceInfo geofence;
  final List<LatLng> points;

  const _GeofenceDetailSheet({required this.geofence, required this.points});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.only(topLeft: Radius.circular(20), topRight: Radius.circular(20)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(geofence.icon, color: geofence.color, size: 28),
              const SizedBox(width: 12),
              Text(geofence.name, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 16),
          Text('Boundary Coordinates (${points.length} points)',
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Colors.grey.shade700)),
          const SizedBox(height: 8),
          SizedBox(
            height: 200,
            child: ListView.separated(
              itemCount: points.length,
              separatorBuilder: (context, i) => const Divider(height: 1),
              itemBuilder: (context, i) => ListTile(
                dense: true,
                leading: CircleAvatar(
                  radius: 12,
                  backgroundColor: geofence.color.withValues(alpha: 0.2),
                  child: Text('${i + 1}', style: TextStyle(fontSize: 10, color: geofence.color, fontWeight: FontWeight.bold)),
                ),
                title: Text(
                  '${points[i].latitude.toStringAsFixed(6)}, ${points[i].longitude.toStringAsFixed(6)}',
                  style: const TextStyle(fontSize: 12, fontFamily: 'monospace'),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
// TAB 3: HISTORY PAGE
// ─────────────────────────────────────────────

class HistoryPage extends StatefulWidget {
  final AppState appState;

  const HistoryPage({super.key, required this.appState});

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  DeviceInfo? _selectedDevice;
  final MapController _historyMapController = MapController();
  bool _showMap = false;

  @override
  void initState() {
    super.initState();
    _selectedDevice = widget.appState.devices.first;
  }

  /// Fetch last 24 hours of location events for a device
  Stream<List<DeviceLocation>> _historyStream(String deviceId) {
    final since = Timestamp.fromDate(DateTime.now().subtract(const Duration(hours: 24)));

    return FirebaseFirestore.instance
        .collection('ttn_uplinks')
        .doc(deviceId)
        .collection('events')
        .where('serverTime', isGreaterThanOrEqualTo: since)
        .orderBy('serverTime', descending: true)
        .snapshots()
        .map((snap) {
      return snap.docs.map((doc) {
        final data = doc.data();
        final decoded = data['decoded'];
        if (decoded == null || decoded is! Map) return null;
        final lat = decoded['lat'] ?? decoded['latitude'];
        final lon = decoded['lon'] ?? decoded['longitude'];
        if (lat == null || lon == null) return null;
        final location = LatLng(
          lat is num ? lat.toDouble() : double.tryParse(lat.toString()) ?? 0.0,
          lon is num ? lon.toDouble() : double.tryParse(lon.toString()) ?? 0.0,
        );
        DateTime? timestamp;
        final serverTime = data['serverTime'];
        if (serverTime is Timestamp) {
          timestamp = serverTime.toDate().toLocal();
        }
        return DeviceLocation(deviceId: deviceId, location: location, timestamp: timestamp ?? DateTime.now());
      }).whereType<DeviceLocation>().toList();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.indigo.shade50, Colors.white],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Header
              Container(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(color: Colors.indigo, borderRadius: BorderRadius.circular(12)),
                      child: const Icon(Icons.history, color: Colors.white, size: 26),
                    ),
                    const SizedBox(width: 12),
                    const Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Location History', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.indigo)),
                        Text('Last 24 hours', style: TextStyle(fontSize: 12, color: Colors.grey)),
                      ],
                    ),
                    const Spacer(),
                    // Toggle list/map
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.indigo.shade50,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: Colors.indigo.shade200),
                      ),
                      child: Row(
                        children: [
                          _viewToggleBtn(Icons.list, !_showMap, () => setState(() => _showMap = false)),
                          _viewToggleBtn(Icons.map, _showMap, () => setState(() => _showMap = true)),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // Device selector
              Container(
                height: 48,
                margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: widget.appState.devices.length,
                  itemBuilder: (context, index) {
                    final d = widget.appState.devices[index];
                    final selected = _selectedDevice?.id == d.id;
                    return GestureDetector(
                      onTap: () => setState(() => _selectedDevice = d),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        margin: const EdgeInsets.only(right: 8),
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                        decoration: BoxDecoration(
                          color: selected ? d.color : d.color.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(24),
                          border: Border.all(color: d.color, width: 1.5),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.person, size: 16, color: selected ? Colors.white : d.color),
                            const SizedBox(width: 6),
                            Text(d.name,
                                style: TextStyle(
                                  fontSize: 13,
                                  fontWeight: FontWeight.w600,
                                  color: selected ? Colors.white : d.color,
                                )),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),

              const SizedBox(height: 8),

              // Content
              Expanded(
                child: _selectedDevice == null
                    ? const Center(child: Text('Select a device above'))
                    : StreamBuilder<List<DeviceLocation>>(
                        stream: _historyStream(_selectedDevice!.id),
                        builder: (context, snapshot) {
                          if (snapshot.connectionState == ConnectionState.waiting) {
                            return const Center(child: CircularProgressIndicator(color: Colors.indigo));
                          }
                          final history = snapshot.data ?? [];
                          if (history.isEmpty) {
                            return Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(Icons.location_off, size: 56, color: Colors.grey.shade300),
                                  const SizedBox(height: 12),
                                  Text('No data in last 24 hours',
                                      style: TextStyle(fontSize: 15, color: Colors.grey.shade600)),
                                ],
                              ),
                            );
                          }
                          return _showMap
                              ? _buildHistoryMap(history)
                              : _buildHistoryList(history);
                        },
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _viewToggleBtn(IconData icon, bool active, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: active ? Colors.indigo : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Icon(icon, size: 18, color: active ? Colors.white : Colors.indigo),
      ),
    );
  }

  Widget _buildHistoryList(List<DeviceLocation> history) {
    final device = _selectedDevice!;
    return ListView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      itemCount: history.length,
      itemBuilder: (context, index) {
        final loc = history[index];
        final isFirst = index == 0;
        return Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Timeline line
            Column(
              children: [
                Container(
                  width: 14,
                  height: 14,
                  margin: const EdgeInsets.only(top: 18),
                  decoration: BoxDecoration(
                    color: isFirst ? device.color : device.color.withValues(alpha: 0.4),
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 2),
                    boxShadow: isFirst ? [BoxShadow(color: device.color.withValues(alpha: 0.4), blurRadius: 6)] : [],
                  ),
                ),
                if (index < history.length - 1)
                  Container(width: 2, height: 50, color: device.color.withValues(alpha: 0.2)),
              ],
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Card(
                margin: const EdgeInsets.only(bottom: 8),
                elevation: isFirst ? 2 : 1,
                child: ListTile(
                  dense: true,
                  leading: isFirst
                      ? Container(
                          padding: const EdgeInsets.all(6),
                          decoration: BoxDecoration(color: device.color, shape: BoxShape.circle),
                          child: const Icon(Icons.navigation, color: Colors.white, size: 14),
                        )
                      : null,
                  title: Text(
                    '${loc.location.latitude.toStringAsFixed(6)}, ${loc.location.longitude.toStringAsFixed(6)}',
                    style: const TextStyle(fontSize: 12, fontFamily: 'monospace'),
                  ),
                  subtitle: Text(
                    _formatTimestamp(loc.timestamp),
                    style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
                  ),
                  trailing: isFirst
                      ? Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                          decoration: BoxDecoration(
                            color: Colors.green.shade100,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Text('Latest', style: TextStyle(fontSize: 10, color: Colors.green, fontWeight: FontWeight.bold)),
                        )
                      : null,
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildHistoryMap(List<DeviceLocation> history) {
    final device = _selectedDevice!;
    final trailPoints = history.reversed.map((h) => h.location).toList();
    final center = history.first.location;

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Card(
        clipBehavior: Clip.antiAlias,
        child: Stack(
          children: [
            FlutterMap(
              mapController: _historyMapController,
              options: MapOptions(
                initialCenter: center,
                initialZoom: 16,
                minZoom: 10,
                maxZoom: 22,
              ),
              children: [
                TileLayer(
                  urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                  userAgentPackageName: 'com.example.tracker',
                  maxZoom: 22,
                  maxNativeZoom: 19,
                ),
                // Trail polyline
                if (trailPoints.length >= 2)
                  PolylineLayer(
                    polylines: [
                      Polyline(
                        points: trailPoints,
                        strokeWidth: 3,
                        color: device.color.withValues(alpha: 0.7),
                      ),
                    ],
                  ),
                // Markers: only first (latest) and last (oldest)
                MarkerLayer(
                  markers: [
                    // Oldest point
                    Marker(
                      point: trailPoints.first,
                      width: 28,
                      height: 28,
                      child: Container(
                        decoration: BoxDecoration(
                          color: Colors.grey.shade400,
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 2),
                        ),
                        child: const Icon(Icons.circle, color: Colors.white, size: 10),
                      ),
                    ),
                    // Latest point
                    Marker(
                      point: trailPoints.last,
                      width: 44,
                      height: 44,
                      child: Container(
                        decoration: BoxDecoration(
                          color: device.color,
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 3),
                          boxShadow: [BoxShadow(color: device.color.withValues(alpha: 0.4), blurRadius: 8)],
                        ),
                        child: const Icon(Icons.person, color: Colors.white, size: 22),
                      ),
                    ),
                  ],
                ),
              ],
            ),
            // Info overlay
            Positioned(
              top: 12,
              left: 12,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  color: device.color,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.2), blurRadius: 4)],
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.timeline, color: Colors.white, size: 16),
                    const SizedBox(width: 6),
                    Text('${history.length} points • 24hr trail',
                        style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 12)),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _formatTimestamp(DateTime dt) {
    final now = DateTime.now();
    final diff = now.difference(dt);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago · ${_hm(dt)}';
    if (diff.inHours < 24) return '${diff.inHours}h ago · ${_hm(dt)}';
    return '${dt.day}/${dt.month} ${_hm(dt)}';
  }

  String _hm(DateTime dt) =>
      '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
}

// ─────────────────────────────────────────────
// TAB 4: DEVICES PAGE
// ─────────────────────────────────────────────

class DevicesPage extends StatefulWidget {
  final AppState appState;

  const DevicesPage({super.key, required this.appState});

  @override
  State<DevicesPage> createState() => _DevicesPageState();
}

class _DevicesPageState extends State<DevicesPage> {
  void _toggleDevice(int index) {
    setState(() {
      widget.appState.toggleDevice(index);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.purple.shade50, Colors.white],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Header
              Container(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(color: Colors.deepPurple, borderRadius: BorderRadius.circular(12)),
                      child: const Icon(Icons.devices, color: Colors.white, size: 26),
                    ),
                    const SizedBox(width: 12),
                    const Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Devices', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.deepPurple)),
                        Text('Manage Tracked Devices', style: TextStyle(fontSize: 12, color: Colors.grey)),
                      ],
                    ),
                    const Spacer(),
                    // Show all / Hide all
                    TextButton(
                      onPressed: () {
                        final allVisible = widget.appState.devices.every((d) => d.isVisible);
                        setState(() {
                          for (int i = 0; i < widget.appState.devices.length; i++) {
                            widget.appState.devices[i].isVisible = !allVisible;
                          }
                        });
                      },
                      child: Text(
                        widget.appState.devices.every((d) => d.isVisible) ? 'Hide All' : 'Show All',
                        style: const TextStyle(color: Colors.deepPurple, fontWeight: FontWeight.w600),
                      ),
                    ),
                  ],
                ),
              ),

              // Summary banner
              Container(
                margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.deepPurple.withValues(alpha: 0.08),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.deepPurple.withValues(alpha: 0.2)),
                ),
                child: Row(
                  children: [
                    Icon(Icons.info_outline, size: 16, color: Colors.deepPurple.shade300),
                    const SizedBox(width: 8),
                    Text(
                      '${widget.appState.devices.where((d) => d.isVisible).length} of ${widget.appState.devices.length} devices visible on map',
                      style: TextStyle(fontSize: 13, color: Colors.deepPurple.shade700),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 8),

              // Device list
              Expanded(
                child: ListView.builder(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  itemCount: widget.appState.devices.length,
                  itemBuilder: (context, index) {
                    final device = widget.appState.devices[index];
                    return StreamBuilder<DeviceLocation?>(
                      stream: LocationHelpers.deviceLocationStream(device.id),
                      builder: (context, snapshot) {
                        final loc = snapshot.data;
                        final hasData = loc != null;

                        return Card(
                          margin: const EdgeInsets.only(bottom: 10),
                          elevation: device.isVisible ? 2 : 0.5,
                          color: device.isVisible ? Colors.white : Colors.grey.shade50,
                          child: Padding(
                            padding: const EdgeInsets.all(14),
                            child: Row(
                              children: [
                                // Avatar
                                Stack(
                                  children: [
                                    Container(
                                      width: 52,
                                      height: 52,
                                      decoration: BoxDecoration(
                                        color: device.isVisible
                                            ? device.color.withValues(alpha: 0.15)
                                            : Colors.grey.shade200,
                                        shape: BoxShape.circle,
                                        border: Border.all(
                                          color: device.isVisible ? device.color : Colors.grey.shade400,
                                          width: 2,
                                        ),
                                      ),
                                      child: Icon(
                                        Icons.person,
                                        color: device.isVisible ? device.color : Colors.grey.shade400,
                                        size: 28,
                                      ),
                                    ),
                                    if (hasData)
                                      Positioned(
                                        right: 0,
                                        bottom: 0,
                                        child: Container(
                                          width: 14,
                                          height: 14,
                                          decoration: BoxDecoration(
                                            color: Colors.green,
                                            shape: BoxShape.circle,
                                            border: Border.all(color: Colors.white, width: 2),
                                          ),
                                        ),
                                      ),
                                  ],
                                ),
                                const SizedBox(width: 14),
                                // Info
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(device.name,
                                          style: TextStyle(
                                            fontSize: 16,
                                            fontWeight: FontWeight.bold,
                                            color: device.isVisible ? Colors.black87 : Colors.grey.shade500,
                                          )),
                                      Text('ID: ${device.id}',
                                          style: TextStyle(fontSize: 11, color: Colors.grey.shade500)),
                                      if (hasData) ...[
                                        const SizedBox(height: 4),
                                        Text(
                                          '${loc.location.latitude.toStringAsFixed(5)}, ${loc.location.longitude.toStringAsFixed(5)}',
                                          style: TextStyle(fontSize: 11, color: Colors.grey.shade600, fontFamily: 'monospace'),
                                        ),
                                        Text(
                                          _timeAgo(loc.timestamp),
                                          style: const TextStyle(fontSize: 11, color: Colors.green),
                                        ),
                                      ] else
                                        Text('No recent data',
                                            style: TextStyle(fontSize: 11, color: Colors.grey.shade400)),
                                    ],
                                  ),
                                ),
                                // Toggle
                                Switch(
                                  value: device.isVisible,
                                  onChanged: (_) => _toggleDevice(index),
                                  activeThumbColor: device.color,
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  String _timeAgo(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    return '${diff.inHours}h ago';
  }
}

// ─────────────────────────────────────────────
// TAB 5: ANOMALIES PAGE
// ─────────────────────────────────────────────

class AnomaliesPage extends StatefulWidget {
  final AppState appState;

  const AnomaliesPage({super.key, required this.appState});

  @override
  State<AnomaliesPage> createState() => _AnomaliesPageState();
}

class _AnomaliesPageState extends State<AnomaliesPage> {
  DeviceInfo? _selectedDevice;
  bool _useIsolationForest = true; // true = Isolation Forest (best.pkl), false = Dart Z-score

  @override
  void initState() {
    super.initState();
    _selectedDevice = widget.appState.devices.first;
  }

  // ── Fetch last 24 h of events and run anomaly detection ──────────────────

  Stream<List<AnomalyResult>> _anomalyStream(String deviceId) {
    final since =
        Timestamp.fromDate(DateTime.now().subtract(const Duration(hours: 24)));

    // Stream directly from ttn_uplinks so the Z-score tab updates in real-time
    // as simulation or real devices write new events — no anomaly_service.py needed.
    return FirebaseFirestore.instance
        .collection('ttn_uplinks')
        .doc(deviceId)
        .collection('events')
        .where('serverTime', isGreaterThanOrEqualTo: since)
        .orderBy('serverTime', descending: false)
        .snapshots()
        .map((snap) {
      final points = <DeviceLocationPoint>[];
      for (final doc in snap.docs) {
        final data = doc.data();
        final decoded = data['decoded'];
        if (decoded == null || decoded is! Map) continue;
        final lat = decoded['lat'] ?? decoded['latitude'];
        final lon = decoded['lon'] ?? decoded['longitude'];
        if (lat == null || lon == null) continue;
        final location = LatLng(
          lat is num ? lat.toDouble() : double.tryParse(lat.toString()) ?? 0.0,
          lon is num ? lon.toDouble() : double.tryParse(lon.toString()) ?? 0.0,
        );
        final serverTime = data['serverTime'];
        final ts = serverTime is Timestamp
            ? serverTime.toDate().toLocal()
            : DateTime.now();
        points.add(DeviceLocationPoint(
            deviceId: deviceId, location: location, timestamp: ts));
      }
      return AnomalyDetector.analyze(points);
    });
  }

  // ── Isolation Forest results stream (written by anomaly_service.py) ────────

  /// Reads pre-computed results from anomaly_results/{deviceId}/live_alerts
  /// which are written by the Python anomaly_service.py in real-time.
  Stream<List<Map<String, dynamic>>> _isolationForestStream(String deviceId) {
    return FirebaseFirestore.instance
        .collection('anomaly_results')
        .doc(deviceId)
        .collection('live_alerts')
        .orderBy('detectedAt', descending: true)
        .limit(50)
        .snapshots()
        .map((snap) => snap.docs.map((d) => d.data()).toList());
  }

  /// Summary document written by the /process/deviceId endpoint
  Stream<Map<String, dynamic>?> _isolationForestSummaryStream(String deviceId) {
    return FirebaseFirestore.instance
        .collection('anomaly_results')
        .doc(deviceId)
        .snapshots()
        .map((snap) => snap.exists ? snap.data() : null);
  }

  Stream<Map<String, double>> _statsStream(String deviceId) {
    final since =
        Timestamp.fromDate(DateTime.now().subtract(const Duration(hours: 24)));

    return FirebaseFirestore.instance
        .collection('ttn_uplinks')
        .doc(deviceId)
        .collection('events')
        .where('serverTime', isGreaterThanOrEqualTo: since)
        .orderBy('serverTime', descending: false)
        .snapshots()
        .map((snap) {
      final points = <DeviceLocationPoint>[];
      for (final doc in snap.docs) {
        final data = doc.data();
        final decoded = data['decoded'];
        if (decoded == null || decoded is! Map) continue;
        final lat = decoded['lat'] ?? decoded['latitude'];
        final lon = decoded['lon'] ?? decoded['longitude'];
        if (lat == null || lon == null) continue;
        final location = LatLng(
          lat is num ? lat.toDouble() : double.tryParse(lat.toString()) ?? 0.0,
          lon is num ? lon.toDouble() : double.tryParse(lon.toString()) ?? 0.0,
        );
        final serverTime = data['serverTime'];
        final ts = serverTime is Timestamp
            ? serverTime.toDate().toLocal()
            : DateTime.now();
        points.add(DeviceLocationPoint(
            deviceId: deviceId, location: location, timestamp: ts));
      }
      return AnomalyDetector.stats(points);
    });
  }

  // ── UI ────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.orange.shade50, Colors.white],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              _buildHeader(),
              _buildModelToggle(),
              _buildDeviceSelector(),
              const SizedBox(height: 8),
              if (_selectedDevice != null) ...[
                if (!_useIsolationForest) _buildStatsRow(_selectedDevice!.id),
                if (!_useIsolationForest) const SizedBox(height: 8),
                Expanded(
                  child: _useIsolationForest
                      ? _buildIsolationForestPanel(_selectedDevice!)
                      : _buildAnomalyList(_selectedDevice!),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.deepOrange,
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.analytics, color: Colors.white, size: 26),
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Anomaly Detection',
                    style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.deepOrange)),
                Text('Movement analysis · Last 24 h',
                    style: TextStyle(fontSize: 12, color: Colors.grey)),
              ],
            ),
          ),
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: _useIsolationForest
                  ? Colors.deepPurple.shade100
                  : Colors.orange.shade100,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              children: [
                Icon(
                  _useIsolationForest ? Icons.forest : Icons.psychology_outlined,
                  size: 14,
                  color: _useIsolationForest
                      ? Colors.deepPurple.shade800
                      : Colors.orange.shade800,
                ),
                const SizedBox(width: 4),
                Text(
                  _useIsolationForest ? 'best.pkl' : 'Z-Score + Rules',
                  style: TextStyle(
                      fontSize: 11,
                      color: _useIsolationForest
                          ? Colors.deepPurple.shade800
                          : Colors.orange.shade800,
                      fontWeight: FontWeight.w600)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDeviceSelector() {
    return Container(
      height: 48,
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: widget.appState.devices.length,
        itemBuilder: (context, index) {
          final d = widget.appState.devices[index];
          final selected = _selectedDevice?.id == d.id;
          return GestureDetector(
            onTap: () => setState(() => _selectedDevice = d),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              margin: const EdgeInsets.only(right: 8),
              padding:
                  const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: selected ? d.color : d.color.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: d.color, width: 1.5),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.person,
                      size: 16,
                      color: selected ? Colors.white : d.color),
                  const SizedBox(width: 6),
                  Text(d.name,
                      style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: selected ? Colors.white : d.color)),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildModelToggle() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 6),
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Row(
        children: [
          Expanded(
            child: GestureDetector(
              onTap: () => setState(() => _useIsolationForest = false),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                padding: const EdgeInsets.symmetric(vertical: 8),
                decoration: BoxDecoration(
                  color: !_useIsolationForest ? Colors.deepOrange : Colors.transparent,
                  borderRadius: BorderRadius.circular(9),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.show_chart, size: 14,
                        color: !_useIsolationForest ? Colors.white : Colors.grey.shade600),
                    const SizedBox(width: 6),
                    Text('Z-Score (Dart)',
                        style: TextStyle(
                            fontSize: 12, fontWeight: FontWeight.w600,
                            color: !_useIsolationForest ? Colors.white : Colors.grey.shade600)),
                  ],
                ),
              ),
            ),
          ),
          Expanded(
            child: GestureDetector(
              onTap: () => setState(() => _useIsolationForest = true),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                padding: const EdgeInsets.symmetric(vertical: 8),
                decoration: BoxDecoration(
                  color: _useIsolationForest ? Colors.deepPurple : Colors.transparent,
                  borderRadius: BorderRadius.circular(9),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.forest, size: 14,
                        color: _useIsolationForest ? Colors.white : Colors.grey.shade600),
                    const SizedBox(width: 6),
                    Text('Isolation Forest',
                        style: TextStyle(
                            fontSize: 12, fontWeight: FontWeight.w600,
                            color: _useIsolationForest ? Colors.white : Colors.grey.shade600)),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Isolation Forest results panel (reads anomaly_results from Firestore) ──

  Widget _buildIsolationForestPanel(DeviceInfo device) {
    return Column(
      children: [
        // Summary card — only shown when batch processing (/process/<id>) has run.
        // When absent, the service hint in the Expanded list below handles messaging.
        StreamBuilder<Map<String, dynamic>?>(
          stream: _isolationForestSummaryStream(device.id),
          builder: (context, snap) {
            final summary = snap.data;
            if (summary == null) {
              return const SizedBox.shrink();
            }
            final total = summary['totalAnalysed'] ?? 0;
            final anomalies = summary['totalAnomalies'] ?? 0;
            return Container(
              margin: const EdgeInsets.fromLTRB(16, 0, 16, 8),
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.deepPurple.shade700, Colors.deepPurple.shade400],
                ),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.forest, color: Colors.white, size: 18),
                      const SizedBox(width: 8),
                      const Text('Isolation Forest · best.pkl',
                          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14)),
                      const Spacer(),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(device.name,
                            style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w600)),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _ifStat('$total', 'Analysed', Colors.white70),
                      _ifStat('$anomalies', 'Anomalies', Colors.yellow.shade200),
                      _ifStat('${total - anomalies}', 'Normal', Colors.green.shade200),
                    ],
                  ),
                ],
              ),
            );
          },
        ),
        // Live alert events
        Expanded(
          child: StreamBuilder<List<Map<String, dynamic>>>(
            stream: _isolationForestStream(device.id),
            builder: (context, snap) {
              if (snap.connectionState == ConnectionState.waiting) {
                return const Center(
                  child: CircularProgressIndicator(color: Colors.deepPurple),
                );
              }
              final alerts = snap.data ?? [];
              if (alerts.isEmpty) {
                return _buildServiceHint(device);
              }
              return ListView.builder(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                itemCount: alerts.length,
                itemBuilder: (context, index) {
                  final a = alerts[index];
                  const color = Colors.deepOrange;
                  final score = (a['anomalyScore'] as num?)?.toStringAsFixed(4) ?? '—';
                  final lat = (a['lat'] as num?)?.toStringAsFixed(5) ?? '—';
                  final lon = (a['lon'] as num?)?.toStringAsFixed(5) ?? '—';
                  return Card(
                    margin: const EdgeInsets.only(bottom: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                      side: BorderSide(color: color.withValues(alpha: 0.4)),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: color.withValues(alpha: 0.15),
                              shape: BoxShape.circle,
                            ),
                            child: Icon(Icons.forest, color: color, size: 20),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: color,
                                        borderRadius: BorderRadius.circular(6),
                                      ),
                                      child: Text('ANOMALY',
                                          style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)),
                                    ),
                                    const SizedBox(width: 8),
                                    Text('Score: $score',
                                        style: TextStyle(fontSize: 11, color: Colors.grey.shade500)),
                                  ],
                                ),
                                const SizedBox(height: 4),
                                Text('acc_mag: ${(a['accMag'] as num?)?.toStringAsFixed(2) ?? '—'} '
                                    '· acc_delta: ${(a['accDelta'] as num?)?.toStringAsFixed(3) ?? '—'} '
                                    '· speed: ${(a['speed'] as num?)?.toStringAsFixed(2) ?? '—'} m/s',
                                    style: const TextStyle(fontSize: 11, fontFamily: 'monospace')),
                                const SizedBox(height: 2),
                                Text('$lat, $lon',
                                    style: TextStyle(fontSize: 11, color: Colors.grey.shade500)),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                },
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _ifStat(String value, String label, Color color) {
    return Column(
      children: [
        Text(value, style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color)),
        Text(label, style: TextStyle(fontSize: 10, color: Colors.white.withValues(alpha: 0.7))),
      ],
    );
  }

  Widget _buildServiceHint(DeviceInfo device) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.deepPurple.shade50,
                shape: BoxShape.circle,
              ),
              child: Icon(Icons.forest, size: 48, color: Colors.deepPurple.shade300),
            ),
            const SizedBox(height: 16),
            Text('Isolation Forest (best.pkl)',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.deepPurple.shade700)),
            const SizedBox(height: 8),
            Text('No results yet for ${device.name}.\nStart the Python service to process data.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 13, color: Colors.grey.shade600)),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.grey.shade300),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Start anomaly_service.py:', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                  SizedBox(height: 4),
                  Text('python anomaly_service.py',
                      style: TextStyle(fontSize: 11, fontFamily: 'monospace', color: Colors.deepOrange)),
                  SizedBox(height: 8),
                  Text('Or trigger batch processing:', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                  SizedBox(height: 4),
                  Text('POST http://localhost:5000/process/all',
                      style: TextStyle(fontSize: 11, fontFamily: 'monospace', color: Colors.deepOrange)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatsRow(String deviceId) {
    return StreamBuilder<Map<String, double>>(
      stream: _statsStream(deviceId),
      builder: (context, snapshot) {
        final stats = snapshot.data ?? {};
        final mean = stats['mean'] ?? 0;
        final max = stats['max'] ?? 0;
        final totalKm = stats['totalKm'] ?? 0;

        return Container(
          margin: const EdgeInsets.symmetric(horizontal: 16),
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: Colors.orange.shade200),
            boxShadow: [
              BoxShadow(
                  color: Colors.orange.withValues(alpha: 0.08),
                  blurRadius: 10,
                  offset: const Offset(0, 4)),
            ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildStatChip(
                  Icons.speed, '${mean.toStringAsFixed(1)} km/h', 'Avg Speed',
                  Colors.blue),
              Container(
                  width: 1,
                  height: 36,
                  color: Colors.grey.shade200),
              _buildStatChip(
                  Icons.bolt, '${max.toStringAsFixed(1)} km/h', 'Max Speed',
                  Colors.orange),
              Container(
                  width: 1,
                  height: 36,
                  color: Colors.grey.shade200),
              _buildStatChip(
                  Icons.route, '${totalKm.toStringAsFixed(2)} km',
                  'Distance', Colors.teal),
            ],
          ),
        );
      },
    );
  }

  Widget _buildStatChip(
      IconData icon, String value, String label, Color color) {
    return Column(
      children: [
        Icon(icon, color: color, size: 18),
        const SizedBox(height: 4),
        Text(value,
            style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
                color: color)),
        Text(label,
            style:
                TextStyle(fontSize: 10, color: Colors.grey.shade500)),
      ],
    );
  }

  Widget _buildAnomalyList(DeviceInfo device) {
    return StreamBuilder<List<AnomalyResult>>(
      stream: _anomalyStream(device.id),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                CircularProgressIndicator(color: Colors.deepOrange),
                SizedBox(height: 12),
                Text('Analysing movement data…',
                    style: TextStyle(color: Colors.grey)),
              ],
            ),
          );
        }

        final anomalies = snapshot.data ?? [];

        if (anomalies.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: Colors.green.shade50,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(Icons.verified_user_rounded,
                      size: 56, color: Colors.green.shade400),
                ),
                const SizedBox(height: 16),
                Text('All Clear!',
                    style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: Colors.green.shade700)),
                const SizedBox(height: 6),
                Text('No anomalies detected for ${device.name}',
                    style: TextStyle(
                        fontSize: 14, color: Colors.grey.shade600)),
                const SizedBox(height: 4),
                Text('in the last 24 hours',
                    style: TextStyle(
                        fontSize: 12, color: Colors.grey.shade400)),
              ],
            ),
          );
        }

        // Summary banner
        return Column(
          children: [
            // Alert count banner
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 16),
              padding: const EdgeInsets.symmetric(
                  horizontal: 16, vertical: 10),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.deepOrange.shade600,
                    Colors.orange.shade500
                  ],
                ),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Row(
                children: [
                  const Icon(Icons.warning_amber_rounded,
                      color: Colors.white, size: 22),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '${anomalies.length} anomal${anomalies.length == 1 ? 'y' : 'ies'} detected',
                          style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                              fontSize: 15),
                        ),
                        Text(
                          'Tap any event for details',
                          style: TextStyle(
                              color:
                                  Colors.white.withValues(alpha: 0.8),
                              fontSize: 11),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color:
                          Colors.white.withValues(alpha: 0.25),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      '${anomalies.where((a) => a.severity == AnomalySeverity.high).length} HIGH',
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.bold),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 10),
            // Anomaly list
            Expanded(
              child: ListView.builder(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                itemCount: anomalies.length,
                itemBuilder: (context, index) {
                  final a = anomalies[index];
                  return _AnomalyCard(
                      anomaly: a, device: device, index: index);
                },
              ),
            ),
          ],
        );
      },
    );
  }
}

// ── Single anomaly card ───────────────────────────────────────────────────────

class _AnomalyCard extends StatelessWidget {
  final AnomalyResult anomaly;
  final DeviceInfo device;
  final int index;

  const _AnomalyCard({
    required this.anomaly,
    required this.device,
    required this.index,
  });

  Color get _severityColor {
    switch (anomaly.severity) {
      case AnomalySeverity.high:
        return Colors.red;
      case AnomalySeverity.medium:
        return Colors.orange;
      case AnomalySeverity.low:
        return Colors.amber;
    }
  }

  IconData get _severityIcon {
    switch (anomaly.severity) {
      case AnomalySeverity.high:
        return Icons.emergency;
      case AnomalySeverity.medium:
        return Icons.warning_amber_rounded;
      case AnomalySeverity.low:
        return Icons.info_outline;
    }
  }

  String get _severityLabel {
    switch (anomaly.severity) {
      case AnomalySeverity.high:
        return 'HIGH';
      case AnomalySeverity.medium:
        return 'MEDIUM';
      case AnomalySeverity.low:
        return 'LOW';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: BorderSide(
            color: _severityColor.withValues(alpha: 0.4), width: 1.2),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: () => _showDetail(context),
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Severity icon
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: _severityColor.withValues(alpha: 0.12),
                  shape: BoxShape.circle,
                ),
                child: Icon(_severityIcon, color: _severityColor, size: 22),
              ),
              const SizedBox(width: 12),
              // Details
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 2),
                          decoration: BoxDecoration(
                            color: _severityColor,
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: Text(
                            _severityLabel,
                            style: const TextStyle(
                                color: Colors.white,
                                fontSize: 10,
                                fontWeight: FontWeight.bold),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          _formatTime(anomaly.timestamp),
                          style: TextStyle(
                              fontSize: 11, color: Colors.grey.shade500),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      anomaly.reason,
                      style: const TextStyle(
                          fontSize: 13, fontWeight: FontWeight.w500),
                    ),
                    const SizedBox(height: 6),
                    // Speed / distance pills
                    Wrap(
                      spacing: 6,
                      children: [
                        if (anomaly.speedKmh > 0)
                          _pill(
                              Icons.speed,
                              '${anomaly.speedKmh.toStringAsFixed(1)} km/h',
                              Colors.blue),
                        if (anomaly.distanceM > 0)
                          _pill(
                              Icons.straighten,
                              '${anomaly.distanceM.toStringAsFixed(0)} m',
                              Colors.teal),
                        _pill(
                            Icons.location_on,
                            '${anomaly.location.latitude.toStringAsFixed(5)}, '
                                '${anomaly.location.longitude.toStringAsFixed(5)}',
                            Colors.grey),
                      ],
                    ),
                  ],
                ),
              ),
              Icon(Icons.chevron_right,
                  color: Colors.grey.shade400, size: 20),
            ],
          ),
        ),
      ),
    );
  }

  Widget _pill(IconData icon, String text, Color color) {
    return Container(
      padding:
          const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 10, color: color),
          const SizedBox(width: 4),
          Text(text,
              style: TextStyle(
                  fontSize: 10,
                  color: color,
                  fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }

  void _showDetail(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) => Container(
        padding: const EdgeInsets.all(24),
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.only(
              topLeft: Radius.circular(24),
              topRight: Radius.circular(24)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Handle
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey.shade300,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: _severityColor.withValues(alpha: 0.15),
                    shape: BoxShape.circle,
                  ),
                  child:
                      Icon(_severityIcon, color: _severityColor, size: 30),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 3),
                            decoration: BoxDecoration(
                              color: _severityColor,
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(_severityLabel,
                                style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold)),
                          ),
                          const SizedBox(width: 8),
                          Text('Anomaly',
                              style: TextStyle(
                                  color: Colors.grey.shade500,
                                  fontSize: 13)),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text(device.name,
                          style: const TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            _detailRow(Icons.warning_amber, 'Detection Reason',
                anomaly.reason),
            const Divider(height: 20),
            _detailRow(Icons.access_time, 'Time',
                anomaly.timestamp.toString()),
            const Divider(height: 20),
            if (anomaly.speedKmh > 0) ...[
              _detailRow(Icons.speed, 'Speed Detected',
                  '${anomaly.speedKmh.toStringAsFixed(2)} km/h'),
              const Divider(height: 20),
            ],
            if (anomaly.distanceM > 0) ...[
              _detailRow(Icons.straighten, 'Distance',
                  '${anomaly.distanceM.toStringAsFixed(2)} m'),
              const Divider(height: 20),
            ],
            _detailRow(
                Icons.location_on,
                'Coordinates',
                '${anomaly.location.latitude.toStringAsFixed(6)}, '
                    '${anomaly.location.longitude.toStringAsFixed(6)}'),
            const SizedBox(height: 24),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () => Navigator.pop(context),
                style: ElevatedButton.styleFrom(
                  backgroundColor: _severityColor,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text('Dismiss',
                    style: TextStyle(fontWeight: FontWeight.bold)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _detailRow(IconData icon, String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 20, color: Colors.grey.shade400),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label,
                  style: TextStyle(
                      fontSize: 11, color: Colors.grey.shade500)),
              const SizedBox(height: 2),
              Text(value,
                  style: const TextStyle(
                      fontSize: 14, fontWeight: FontWeight.w600)),
            ],
          ),
        ),
      ],
    );
  }

  String _formatTime(DateTime dt) {
    final now = DateTime.now();
    final diff = now.difference(dt);
    final hm =
        '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago · $hm';
    if (diff.inHours < 24) return '${diff.inHours}h ago · $hm';
    return '${dt.day}/${dt.month} $hm';
  }
}