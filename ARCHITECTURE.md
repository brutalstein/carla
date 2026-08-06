# Architecture

```text
CARLA Server
   │
   ├── Ego Vehicle Adapter
   ├── ODD Environment
   └── Sensor Actors
          │ callbacks
          ▼
   Exact-Frame Synchronizer
          │
          ├── Calibration Registry
          ├── Ground-Truth Localization Adapter + GNSS/IMU Health
          └── Semantic LiDAR Instance Detector
                         │
                         ▼
                  PerceptionFrame
```

## Fail-fast davranışları

- Yanlış veya eksik YAML: `ConfigurationError`
- CARLA API import/connection sorunu: `CarlaConnectionError`
- Gerekli sensor frame timeout: `SensorTimeoutError`
- Gerekli blueprint/attribute yokluğu: çalışma başlamadan hata
- Opsiyonel blueprint/attribute yokluğu: warning ve kontrollü atlama
