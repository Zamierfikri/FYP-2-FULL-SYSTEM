import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

/// Default [FirebaseOptions] for use with your Firebase apps.
class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return web;
    }
    // Fallback to web config if run on other platforms without specific setup
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
      case TargetPlatform.iOS:
      case TargetPlatform.macOS:
      case TargetPlatform.windows:
      case TargetPlatform.linux:
      case TargetPlatform.fuchsia:
        return web;
    }
  }

  static const FirebaseOptions web = FirebaseOptions(
    apiKey: 'AIzaSyDjIXwPAfeYZCLCyC8A2BmBOM9HbHhf_a0',
    appId: '1:697575902423:android:590c5edaf3e57ce77492',
    messagingSenderId: '697575902423',
    projectId: 'fulcrum-86a7b',
    authDomain: 'fulcrum-86a7b.firebaseapp.com',
    storageBucket: 'fulcrum-86a7b.appspot.com',
    measurementId: 'G-R51WFTP7T3',
  );
}
