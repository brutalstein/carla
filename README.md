# CARLA L4 Runtime, Lokalizasyon ve Perception Temeli

CARLA için modüler, deterministik ve yapılandırma odaklı L4 araştırma stack'i. Mevcut
sürüm ortak runtime, GNSS/IMU lokalizasyonu ve hazır checkpoint'lerle çalışacak süreç-
izole perception mimarisini içerir. Fusion/world model, prediction, planning ve control
algoritmaları henüz bulunmaz.

## Güncel kapsam

1. **Araç / simülasyon platformu** — Lincoln MKZ 2020 ve kontrollü blueprint fallback.
2. **ODD sınırı** — harita, hız, hava, görüş, lokalizasyon ve sensör sağlığı.
3. **Raw sensör sistemi** — tek ray-cast LiDAR, 6 RGB kamera, 5 radar, GNSS ve IMU.
4. **Senkronizasyon** — synchronous CARLA, sabit 50 ms adım ve exact-frame GNSS/IMU.
5. **Ortak runtime** — lifecycle, priority executor, bounded channel, immutable message,
   atomik snapshot, deadline/freshness, health, lineage ve supervisor.
6. **Lokalizasyon** — GNSS + IMU + pusula planar error-state EKF.
7. **Perception altyapısı** — BEVFusion Detection, BEVFusion Segmentation, MapTRv2,
   TLD-READY ve CitySemSegFormer için izole backend, adapter, config, artifact transport,
   model başına lifecycle/deadline/health ve partial snapshot.

## Perception veri akışı

```text
CARLA cameras + LiDAR + calibration
                ↓
        ArtifactRef transport
                ↓
 MessageEnvelope[PerceptionInput]
                ↓
      independent model routes
       ├── BEVFusion Detection
       ├── BEVFusion Segmentation
       ├── MapTRv2
       ├── TLD-READY
       └── CitySemSegFormer
                ↓
       normalized ModelOutput
                ↓
        PerceptionSnapshot
```

Model bağımlılıkları ana Python ortamına kurulmaz. Her model ayrı conda/container
sürecinde JSONL protokolüyle çalışır. Bir modelin açılmaması veya timeout olması diğer
model component'lerini durdurmaz.

Detaylı belgeler:

- [`docs/RUNTIME_ARCHITECTURE.md`](docs/RUNTIME_ARCHITECTURE.md)
- [`docs/RUNTIME_COMPONENT_GUIDE.md`](docs/RUNTIME_COMPONENT_GUIDE.md)
- [`docs/PERCEPTION_ARCHITECTURE.md`](docs/PERCEPTION_ARCHITECTURE.md)
- [`docs/PERCEPTION_MODEL_SOURCES.md`](docs/PERCEPTION_MODEL_SOURCES.md)
- [`models/perception/README.md`](models/perception/README.md)
- [`ARCHITECTURE.md`](ARCHITECTURE.md)

## Ground-truth politikası

Runtime semantic LiDAR, semantic/instance kamera, actor transformu veya actor velocity
kullanmaz. Bu blueprint'ler config doğrulamasında reddedilir. Ground truth yalnızca
offline benchmark/regression testlerinde referans olabilir; model runtime girdisine
aktarılmaz.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate          # Linux
# .venv\Scripts\activate           # Windows PowerShell
python -m pip install -e '.[dev]'
```

CARLA Python istemcisi CARLA sunucu sürümüyle eşleşmelidir.

## Yapılandırma doğrulama

```bash
l4stack --config-dir config validate
l4stack --config-dir config coverage
l4stack --config-dir config perception-doctor
```

`perception-doctor`, model dosyalarını, minimum boyutları, opsiyonel SHA-256 değerlerini,
backend komutunu ve zorunlu environment değişkenlerini kontrol eder. Model dosyaları
kurulana kadar perception varsayılan olarak kapalıdır.

## Model kurulumu

Model ağırlıkları Git repository'sine eklenmez. İndirme ve yerleştirme adımları:

```text
models/perception/bevfusion_detection/README.md
models/perception/bevfusion_segmentation/README.md
models/perception/maptrv2/README.md
models/perception/tld_ready/README.md
models/perception/citysemsegformer/README.md
```

Önce tek model kurup etkinleştir; health/deadline ölçümleri doğrulandıktan sonra diğer
modelleri sırayla aç.

## CARLA çalıştırma

```bash
l4stack --config-dir config run --frames 400
```

Araç şu anda tam frenli bekler; sürüş kontrol katmanı yoktur. Perception etkinse ana
CARLA tick'i model sonucunu beklemez; modeller kendi executor hızlarında asenkron
çalışır.

Çıktılar:

- `output/calibration.json`: sensör extrinsic ve attribute kayıtları,
- `output/perception_artifacts/`: bounded kamera/LiDAR artifact'ları,
- `output/frames.jsonl`: ODD, lokalizasyon, perception snapshot, health ve deadline.

## Test ve kalite kapıları

```bash
python -m compileall -q src tests scripts
pytest -q
ruff check .
```

Perception testleri:

- JSONL subprocess readiness ve inference round-trip,
- beş repository wrapper'ının mock handshake'i,
- model başına lifecycle ve hata izolasyonu,
- stale input, timeout, protocol ve schema failure yolları,
- rate gating, queue backpressure ve partial snapshot,
- artifact manifest/minimum size/SHA-256 denetimi,
- CARLA BGRA/LiDAR artifact yazımı ve frame cache,
- kamera/LiDAR source-time skew kontrolü,
- normalize BEV detection/segmentation/vector-map/light/image-mask çıktıları.

Gerçek model inference testleri, checkpoint ve GPU ortamı kurulunca model README'lerindeki
resmî smoke testlerden sonra çalıştırılmalıdır.

## Gerçek zaman ve doğruluk sınırı

Python executor hard real-time garanti vermez. Disk tabanlı artifact transportu ilk
entegrasyon ve replay içindir; düşük gecikmeli üretim akışında shared-memory veya CUDA
IPC gerekir. Hazır modeller gerçek dünya dataset'leriyle eğitilmiştir; resmî benchmark
sonuçları CARLA doğruluğu değildir. CARLA zero-shot metriği ayrıca ölçülmelidir.
