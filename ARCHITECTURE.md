# Architecture

```text
CARLA Server
   │
   ├── Ego Vehicle Adapter
   ├── ODD Environment
   └── Raw Sensor Actors
          │ callbacks
          ▼
   Exact-Frame Synchronizer
          │
          ├── Calibration Registry
          └── GNSS + IMU Planar Error-State EKF
                         │
                         ▼
                 LocalizationEstimate
                         │
                         ▼
                     ODD Monitor
```

Runtime yalnızca raw kamera, normal ray-cast LiDAR, radar, GNSS ve IMU sensörlerini
kullanır. Semantik/instance sensör blueprint'leri konfigürasyon doğrulamasında
reddedilir. CARLA actor transformu veya actor velocity lokalizasyon girdisi değildir.

## Fail-fast davranışları

- Yanlış veya eksik YAML: `ConfigurationError`
- Ground-truth sensör blueprint'i: `ConfigurationError`
- CARLA API import/connection sorunu: `CarlaConnectionError`
- Gerekli sensor frame timeout: `SensorTimeoutError`
- Gerekli blueprint/attribute yokluğu: çalışma başlamadan hata
- Opsiyonel blueprint/attribute yokluğu: warning ve kontrollü atlama
