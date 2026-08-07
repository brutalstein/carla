# CARLA L4 Runtime, Lokalizasyon ve CUDA Perception Temeli

CARLA için modüler, deterministik ve yapılandırma odaklı L4 araştırma stack'i. Güncel
sürüm ortak runtime, GNSS/IMU lokalizasyonu ve RTX 5090 üzerinde CUDA-only çalışacak
perception yürütme altyapısını içerir. Fusion/world model, prediction, planning ve
control algoritmaları henüz bulunmaz.

## Güncel kapsam

1. Lincoln MKZ 2020 tabanlı CARLA platformu ve ODD sınırı.
2. Tek 32-kanal LiDAR, 6 RGB kamera, 5 radar, GNSS ve IMU.
3. 20 Hz deterministic CARLA tick ve exact-frame GNSS/IMU lokalizasyonu.
4. Lifecycle, bounded channel, atomic snapshot, deadline, health ve lineage runtime'ı.
5. GNSS + IMU + pusula planar error-state EKF.
6. BEVFusion Detection/Segmentation, MapTRv2, TLD-READY ve CitySemSegFormer için
   süreç-izole model component'leri.
7. Disk yerine lease korumalı POSIX shared-memory sensör ve calibration transportu.
8. Worker rasterları için protokol v2 release onaylı bounded output ring.
9. Tek RTX 5090 için global deadline-aware GPU admission controller ve CUDA MPS.
10. CUDA readiness handshake; CPU inference fallback veya sentetik backend yoktur.

## Online perception veri akışı

```text
CARLA cameras + LiDAR
        ↓ tek host-copy
POSIX shared-memory bounded rings
        ↓ shm:// ArtifactRef
MessageEnvelope[PerceptionInput]
        ↓ global GPU admission
persistent CUDA workers under MPS
 ├── BEVFusion Detection
 ├── TLD-READY
 ├── MapTRv2
 ├── BEVFusion Segmentation
 └── CitySemSegFormer
        ↓
latest-valid partial PerceptionSnapshot
```

Model sonucu ana CARLA tick'ini bloklamaz. Queue kapasitesi birdir; GPU bütçesine
alınmayan eski frame birikmeden atlanır. Shared-memory slotu, kabul edilen bütün model
future'ları tamamlanana kadar yeniden kullanılamaz.

## RTX 5090 referans platformu

```text
Ubuntu 24.04
NVIDIA driver >= 610.43.02
CUDA 13.3 + TensorRT 11.1.0
PyTorch 2.12.1 + CUDA 13.0
NVIDIA Container Toolkit 1.19.1
Python 3.12
CUDA MPS
```

Kurulum rehberi:

- [`docs/RTX5090_CUDA_SETUP.md`](docs/RTX5090_CUDA_SETUP.md)
- [`docs/PERCEPTION_PERFORMANCE.md`](docs/PERCEPTION_PERFORMANCE.md)
- [`docs/PERCEPTION_ARCHITECTURE.md`](docs/PERCEPTION_ARCHITECTURE.md)
- [`docs/PERCEPTION_MODEL_SOURCES.md`](docs/PERCEPTION_MODEL_SOURCES.md)
- [`models/perception/README.md`](models/perception/README.md)

## Host kurulumu

Önce audit:

```bash
./scripts/perception/setup_rtx5090_host.sh
```

Sonra kontrollü uygulama:

```bash
./scripts/perception/setup_rtx5090_host.sh --apply
```

Image'lar:

```bash
docker compose -f infra/perception/docker-compose.cuda.yml build
docker compose -f infra/perception/docker-compose.cuda.yml up -d
```

## Ana Python ortamı

Ana `l4stack` ortamına model framework'leri kurulmaz:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

BEVFusion/MapTR/TLD/TensorRT bağımlılıkları kendi container image'larında tutulur.

## Doğrulama

```bash
l4stack --config-dir config validate
l4stack --config-dir config coverage
l4stack --config-dir config cuda-doctor
l4stack --config-dir config perception-doctor
```

`cuda-doctor` gerçek RTX 5090, driver, compute capability 12.0, VRAM, Docker,
NVIDIA Container Toolkit, MPS ve `/dev/shm` kapasitesini kontrol eder.
`perception-doctor` checkpoint, target-SM120 engine, backend komutu ve gerekli environment
alanlarını doğrular.

## Model kurulumu

Model ağırlıkları ve engine dosyaları repository'ye eklenmez. Her modelin dizininde
indirme, port/export ve hedef dosya yolu açıklanır:

```text
models/perception/bevfusion_detection/README.md
models/perception/bevfusion_segmentation/README.md
models/perception/maptrv2/README.md
models/perception/tld_ready/README.md
models/perception/citysemsegformer/README.md
```

TensorRT engine'leri hedef RTX 5090 ve TensorRT 11.1 üzerinde yeniden üretilir. Başka
GPU'dan kopyalanan engine kabul edilmez. MapTRv2 eski resmî PyTorch 1.9 ortamıyla
çalıştırılmaz; PyTorch 2.12/CUDA 13 port smoke testi ve doğrulama kaydı zorunludur.

## CARLA çalıştırma

Modeller doğrulanıp `config/perception.yaml` içinde açıldıktan sonra:

```bash
l4stack --config-dir config run --frames 400
```

Araç halen tam frenli bekler; sürüş kontrol katmanı yoktur. Çıktılar:

- `output/calibration.json`
- `output/frames.jsonl`

Online kamera/LiDAR frame dosyası üretilmez.

## Gerçek performans kapısı

```bash
python scripts/perception/benchmark_guard.py output/frames.jsonl \
  --minimum-frames 400 \
  --model-p95-ms bevfusion_detection=80 \
  --model-p95-ms tld_ready=50
```

Bu araç boş veya sentetik logu kabul etmez; gerçek model çıktısı, failure rate, GPU skip
rate ve p95 ölçümlerini kontrol eder.

## Test ve kalite kapıları

```bash
python -m compileall -q src tests scripts
pytest -q
ruff check .
```

CI, GPU bulunmayan runner'da protokol, shared-memory ring, lease, config ve admission
controller testlerini çalıştırır. Gerçek CUDA/engine/CARLA testleri yalnız RTX 5090 host
deployment kapısında çalıştırılır; CI sonucu gerçek model doğruluğu olarak yorumlanmaz.

## Ground-truth politikası

Runtime semantic LiDAR, semantic/instance kamera, actor transformu veya actor velocity
kullanmaz. CARLA ground truth yalnız offline benchmark/regression referansı olabilir.

## Sınırlar

Bu sistem araştırma altyapısıdır; ISO 26262 veya ISO 21448 sertifikası değildir. Python
runtime hard real-time garanti vermez. Hazır modeller gerçek dünya veri setleriyle
eğitilmiştir; CARLA zero-shot doğruluğu gerçek benchmark ile ayrıca ölçülmelidir.
