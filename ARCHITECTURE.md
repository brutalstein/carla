# Sistem Mimarisi

```text
CARLA Server
   │
   ├── Lincoln MKZ Ego Vehicle
   ├── ODD Environment
   └── Raw Sensor Actors
          │ callbacks
          ▼
   Sensor Synchronizer
          │
          ├───────────────────────────────────────────────────┐
          │                                                   │
          ▼                                                   ▼
MessageEnvelope[SensorFrame]                    PerceptionArtifactStore
          │                                      camera/LiDAR file:// refs
          ▼                                                   │
Priority Executor: localization                              ▼
          │                               MessageEnvelope[PerceptionInput]
          ▼                                                   │
Managed Localization Component                  independent model executors
          │                                      ├── BEVFusion Detection
          ├── GNSS/IMU ESKF                     ├── BEVFusion Segmentation
          ├── Deadline/Health                   ├── MapTRv2
          ├── Snapshot/Lineage                  ├── TLD-READY
          └── Bounded Channel                   └── CitySemSegFormer
          │                                                   │
          ▼                                                   ▼
LocalizationEstimate                              normalized ModelOutput
          │                                                   │
          ├── ODD Monitor                                    ▼
          │                                       partial PerceptionSnapshot
          └──────────────────────┬────────────────────────────┘
                                 ▼
                         JSONL Diagnostics
```

## Yatay runtime katmanı

```text
src/l4stack/runtime/
├── clock.py
├── message.py
├── sensor_frame.py
├── channel.py
├── snapshot.py
├── contracts.py
├── deadline.py
├── health.py
├── lineage.py
├── lifecycle.py
├── executor.py
├── supervisor.py
└── context.py
```

Fonksiyonel katmanlar ortak mutable state paylaşmaz. Mesajlar immutable, sürümlü,
zaman damgalı ve validity sürelidir. Queue'lar bounded, output'lar atomik snapshot'tır.

## Perception modülleri

```text
src/l4stack/perception/
├── types.py          # Input/output ve ArtifactRef sözleşmeleri
├── protocol.py       # JSONL request/response sürümü
├── backend.py        # İzole subprocess client ve test backend'i
├── server.py         # Model-side JSONL server iskeleti
├── adapters.py       # Model çıktısını ortak şemaya normalize eder
├── config.py         # Model, process ve artifact manifest config'i
├── manifest.py       # Dosya/hash/backend readiness kontrolü
├── input.py          # CARLA BGRA/LiDAR artifact writer ve input publisher
├── component.py      # Model başına lifecycle/deadline/health/lineage
├── orchestrator.py   # Async fan-out, rate gate, backpressure, partial snapshot
└── factory.py        # YAML'dan izole component/supervisor üretimi
```

## Bütünlük kuralları

1. Model framework bağımlılıkları ana stack ortamına kurulmaz.
2. Her model ayrı OS süreci, executor ve lifecycle sahibidir.
3. Bir modelin startup/inference hatası diğer modeli durdurmaz.
4. Görüntü/nokta bulutu JSON içine gömülmez; artifact URI taşınır.
5. Her artifact kaynak frame ve timestamp taşır.
6. Farklı sensör hızları latest-at-or-before bariyeriyle korunur; sahte exact-frame yoktur.
7. Sensör skew sınırını aşan girdi model çağrısından önce reddedilir.
8. Model queue kapasitesi birdir; eski backlog oluşturulmaz.
9. Artifact yalnız due model olduğunda yazılır; aynı frame cache üzerinden paylaşılır.
10. Pipeline output'ların tamamını beklemez; latest-valid partial snapshot üretir.
11. Kalıcı model hatası yalnız ilgili route'u devre dışı bırakır.
12. Tüm çıktılar kaynak input message ID'sine lineage parent olarak bağlıdır.
13. Output frame'i açıkça `EGO_LOCAL`, `EGO_BEV_RASTER` veya `CAMERA_PIXEL` olur.

## Fail-fast ve degradation

- Yanlış perception YAML: `ConfigurationError`.
- Eksik model dosyası/backend env: ilgili model configure hatası.
- Geçersiz veya stale input: ilgili frame reddedilir; model süreç hatasına alınmaz.
- Backend timeout/protokol/inference hatası: ilgili model `ERROR/FAILED`.
- Executor queue dolu: yalnızca ilgili submit reddedilir, diğer modeller devam eder.
- Süresi dolmuş output: perception snapshot'tan çıkarılır.
- Semantic/instance CARLA runtime blueprint'i: config yüklenirken reddedilir.

Detaylar `docs/PERCEPTION_ARCHITECTURE.md` içindedir.
