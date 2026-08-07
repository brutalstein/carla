# Sistem Mimarisi

```text
CARLA Server — 20 Hz synchronous world
   │
   ├── GNSS + IMU ──► exact-frame SensorFrame
   │                      ↓
   │               Localization executor
   │                      ↓
   │                 GNSS/IMU ESKF
   │                      ↓
   │              LocalizationEstimate
   │
   └── 6 RGB + LiDAR ─► latest-at-or-before barrier
                          ↓ one host-copy
                    POSIX shared-memory rings
                          ↓ ArtifactRef + timestamp
                    PerceptionInput envelope
                          ↓ rate gate
                    Global GPU admission
                          ↓ CUDA MPS
            ┌─────────────┼─────────────┬─────────────┐
            ▼             ▼             ▼             ▼
        BEVFusion       TLD-READY      MapTRv2    BEV/City Seg.
        Detection       TensorRT       PyTorch       TensorRT
            └─────────────┴─────────────┴─────────────┘
                          ↓
                partial PerceptionSnapshot
                          ↓
                 Future Fusion/World Model
```

## Yatay runtime

```text
clock / message / channel / snapshot / lifecycle / executor
contracts / deadline / health / lineage / supervisor
```

Mesajlar immutable ve sürümlüdür. Katmanlar ortak mutable state paylaşmaz. Queue'lar
bounded, output'lar atomik latest-valid snapshot'tır.

## Perception çekirdeği

```text
src/l4stack/perception/
├── types.py
├── protocol.py
├── backend_contracts.py
├── backend_process.py
├── server.py
├── shared_memory.py
├── gpu_admission.py
├── adapters*.py
├── config.py
├── manifest.py
├── input.py
├── component.py
├── orchestrator.py
└── factory.py
```

## Bütünlük ve performans kuralları

1. Runtime'da sentetik/fake backend yoktur.
2. Online sensör frame'i diske yazılmaz.
3. Kamera/LiDAR JSON'a veya base64'e gömülmez.
4. Sensör, calibration ve raster shared-memory slotları generation/lease ile korunur.
5. Worker modeli startup'ta bir kez GPU'ya yükler.
6. CUDA readiness doğrulanmadan model ACTIVE olmaz.
7. CPU inference fallback yasaktır.
8. Her model executor queue kapasitesi birdir.
9. Global GPU admission bütçeye alınmayan işi queue'ya göndermez.
10. Critical/required/opportunistic sınıfları tek GPU baskısını yönetir.
11. CUDA MPS process context scheduling maliyetini azaltır.
12. Bir model hatası diğer model lifecycle'ını kapatmaz.
13. Snapshot bütün modelleri beklemez; son geçerli sonuçları birleştirir.
14. Ground truth yalnız offline benchmark referansıdır.

## Bağımlılık izolasyonu

Host ana stack:

```text
Python 3.12 + NumPy + PyYAML
```

TensorRT worker:

```text
Ubuntu 24.04 + CUDA 13.3 + TensorRT 11.1
```

PyTorch/MapTR worker:

```text
Ubuntu 24.04 + CUDA 13.0 + PyTorch 2.12.1
```

TLD worker:

```text
PyTorch 2.12.1 + Ultralytics 8.4.104 + TensorRT export/runtime
```

MapTRv2 eski PyTorch 1.9/MMCV 1.4 environment'ıyla RTX 5090 üzerinde çalıştırılmaz.
Modern port gerçek checkpoint ve gerçek altı kamera smoke testinden geçmelidir.

## Fail-fast ve degradation

- CUDA/RTX 5090/driver/MPS eksikliği: `cuda-doctor` başarısız.
- Checkpoint veya SM120 engine eksikliği: `perception-doctor` başarısız.
- Worker CPU readiness bildirirse configure başarısız.
- Shared-memory slot yoksa frame bounded timeout ile atlanır.
- GPU budget yetersizse düşük öncelikli model frame'i atlanır.
- Backend timeout/protokol/inference hatası ilgili modeli `ERROR/FAILED` yapar.
- Süresi dolmuş output snapshot'tan çıkarılır.

Detaylar:

- `docs/RTX5090_CUDA_SETUP.md`
- `docs/PERCEPTION_ARCHITECTURE.md`
- `docs/PERCEPTION_PERFORMANCE.md`
