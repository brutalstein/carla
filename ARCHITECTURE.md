# Sistem Mimarisi

```text
CARLA Server
   │
   ├── Lincoln MKZ Ego Vehicle
   ├── ODD Environment
   └── Raw Sensor Actors
          │ callback
          ▼
   Exact-Frame Sensor Synchronizer
          │
          ▼
   MessageEnvelope[SensorFrame]
          │
          ├── source timestamp / lifespan
          ├── sequence / message id
          └── lineage parent bilgisi
          │
          ▼
   Priority Executor: localization
          │
          ▼
   Managed Localization Component
          │
          ├── GNSS + IMU Planar ESKF
          ├── Deadline/Freshness Monitor
          ├── Health Registry
          ├── Lineage Store
          ├── Bounded Output Channel
          └── Atomic Snapshot Store
          │
          ▼
   MessageEnvelope[LocalizationEstimate]
          │
          ├── ODD Monitor
          └── JSONL Runtime Diagnostics
```

Runtime katmanı fonksiyonel algoritmalardan ayrıdır. Lokalizasyon, gelecekte eklenecek
perception, world model, prediction, planning ve control bileşenleriyle aynı mesaj,
lifecycle, zamanlama ve sağlık sözleşmelerini kullanır.

## Runtime modülleri

```text
src/l4stack/runtime/
├── clock.py          # Simulation/steady/manual zaman kaynakları
├── message.py        # Immutable, sürümlü ve zaman damgalı mesaj zarfı
├── sensor_frame.py   # Exact-frame sensör bundle sözleşmesi
├── channel.py        # Bounded queue ve taşma politikaları
├── snapshot.py       # Atomik latest-valid snapshot deposu
├── contracts.py      # Deadline, priority ve channel sözleşmeleri
├── deadline.py       # Freshness, budget ve output-period denetimi
├── health.py         # Merkezi component health registry
├── lineage.py        # Çıktıdan girdiye veri soy ağacı
├── lifecycle.py      # Managed component state machine
├── executor.py       # Priority worker ve periodic scheduler
├── supervisor.py     # Bağımlılık sıralı başlatma/kapatma
└── context.py        # Ortak servis dependency injection nesnesi
```

## Temel bütünlük kuralları

1. Katmanlar ortak mutable state paylaşmaz.
2. Bir çıktı tamamlanmadan yayınlanmaz.
3. Yayınlanan mesaj immutable kabul edilir.
4. Her mesaj source time, publish time ve validity süresi taşır.
5. Katmanlar tek bir snapshot sürümünü işlem boyunca sabit kullanır.
6. Queue'lar bounded'dır; sınırsız backlog oluşmaz.
7. Eski veya süresi dolmuş girdiler sözleşmeye göre reddedilir.
8. Lifecycle geçişleri supervisor tarafından bağımlılık sırasıyla uygulanır.
9. Normal veri zinciri lineage parent kimlikleriyle geriye izlenebilir.
10. Python runtime hard real-time iddiasında bulunmaz; deadline ölçer ve ihlalleri görünür kılar.

## Fail-fast davranışları

- Yanlış veya eksik YAML: `ConfigurationError`
- Ground-truth sensör blueprint'i: `ConfigurationError`
- Geçersiz runtime contract: `ConfigurationError`
- Geçersiz lifecycle geçişi: `LifecycleError`
- Süresi dolmuş lokalizasyon girdisi: `StaleLocalizationInput`
- Kapatılmış channel kullanımı: `ChannelClosed`
- CARLA API import/connection sorunu: `CarlaConnectionError`
- Gerekli sensor frame timeout: `SensorTimeoutError`
- Gerekli blueprint/attribute yokluğu: çalışma başlamadan hata
