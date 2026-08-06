# CARLA L4 Runtime ve Lokalizasyon Temeli

CARLA için modüler, deterministik ve yapılandırma odaklı araç, ODD, sensör, ortak
runtime ve GNSS/IMU lokalizasyon temeli. Bu sürümde perception, fusion, prediction,
planning ve control algoritmaları bulunmaz; bu katmanların kullanacağı ortak yürütme
altyapısı hazırdır.

## Güncel kapsam

1. **Araç / simülasyon platformu** — Lincoln MKZ 2020 ve kontrollü blueprint fallback.
2. **ODD sınırı** — harita, hız, hava, görüş, lokalizasyon ve sensör sağlığı.
3. **Raw sensör sistemi** — tek normal ray-cast LiDAR, 6 RGB kamera, 5 radar,
   GNSS ve IMU.
4. **Senkronizasyon** — synchronous CARLA, sabit 50 ms adım ve exact-frame GNSS/IMU.
5. **Ortak runtime** — lifecycle, priority executor, bounded channel, immutable message,
   atomik snapshot, deadline/freshness, health, lineage ve supervisor.
6. **Lokalizasyon runtime component'i** — GNSS + IMU + pusula planar ESKF.

## Runtime veri akışı

```text
Exact-frame GNSS/IMU
        ↓
MessageEnvelope[SensorFrame]
        ↓
PriorityExecutor(localization)
        ↓
LocalizationRuntimeComponent
        ↓
MessageEnvelope[LocalizationEstimate]
        ├── AtomicSnapshotStore
        ├── BoundedChannel
        ├── DeadlineMonitor
        ├── HealthRegistry
        └── LineageStore
```

Katmanlar ortak mutable değişken düzenlemez. Her çıktı tamamlandıktan sonra immutable
mesaj olarak yayınlanır; source time, publish time, validity ve parent message id taşır.

Detaylı belgeler:

- [`docs/RUNTIME_ARCHITECTURE.md`](docs/RUNTIME_ARCHITECTURE.md)
- [`docs/RUNTIME_COMPONENT_GUIDE.md`](docs/RUNTIME_COMPONENT_GUIDE.md)
- [`ARCHITECTURE.md`](ARCHITECTURE.md)

## Ground-truth politikası

Çalışma zamanı semantic LiDAR, semantic/instance kamera, actor transformu veya actor
velocity kullanmaz. Bu blueprint'ler config doğrulamasında reddedilir. Ground truth
yalnızca `tests/test_localization_benchmark.py` içinde algoritmik hata metriği üretmek
için kullanılır; runtime girdisi değildir.

## Lokalizasyon algoritması

Durum:

```text
[p_e, p_n, v_e, v_n, yaw, b_ax, b_ay, b_gz]
```

- IMU propagation,
- WGS-84 ECEF → local ENU,
- GNSS anten lever-arm,
- GNSS position update,
- compass yaw update,
- chi-square/NIS outlier gating,
- Joseph covariance update,
- accelerometer/gyro bias random walk,
- covariance tabanlı health.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate          # Linux
# .venv\Scripts\activate           # Windows PowerShell
pip install -e ".[dev]"
```

CARLA Python istemcisi CARLA sunucu sürümüyle eşleşmelidir.

## Yapılandırma doğrulama

```bash
l4stack --config-dir config validate
l4stack --config-dir config coverage
```

`validate`, 14 sensörü ve `localization` runtime contract'ını doğrular.

## CARLA çalıştırma

```bash
l4stack --config-dir config run --frames 400
```

Araç şu anda tam frenli bekler; sürüş kontrol katmanı yoktur.

Çıktılar:

- `output/calibration.json`: sensör extrinsic ve attribute kayıtları,
- `output/frames.jsonl`: ODD, lokalizasyon, message metadata, deadline ve health kayıtları.

## Yapılandırma dosyaları

- `config/simulator.yaml`: CARLA bağlantısı, map, timestep, seed ve hava.
- `config/vehicle.yaml`: ego araç ve başlangıç kontrolü.
- `config/sensors.yaml`: raw sensör placement ve sensör gürültüleri.
- `config/localization.yaml`: ESKF covariance, noise ve gate değerleri.
- `config/runtime.yaml`: executor profilleri, component contract, queue ve deadline.
- `config/odd.yaml`: izin verilen operasyon koşulları.
- `config/logging.yaml`: log seviyesi ve çıktı adı.

## Test ve kalite kapıları

```bash
python -m compileall -q src tests
pytest -q
ruff check .
```

Test paketi şunları kapsar:

- runtime message kimliği, validity ve immutability,
- bounded channel taşma ve blocking davranışı,
- atomik snapshot sürümleme,
- lifecycle ve supervisor dependency sırası,
- priority executor sırası,
- deadline/freshness ihlalleri,
- health ve lineage,
- lokalizasyon runtime entegrasyonu,
- ESKF benchmark ve GNSS outlier reddi,
- geodesy, ODD, config, kamera kapsaması ve sensör frame bariyeri.

## Gerçek zaman sınırı

Python executor hard real-time garanti vermez. Mevcut altyapı queue büyümesini sınırlar,
busy-spin'i önler, priority sınıfları uygular ve deadline ihlallerini kaydeder. Gerçek
araç üretim ortamında RTOS/PREEMPT_RT, CPU affinity, process isolation ve safety
sertifikasyon gereksinimleri ayrıca ele alınmalıdır.
