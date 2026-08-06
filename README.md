# CARLA L4 Perception Foundation

CARLA için modüler, deterministik ve yapılandırma odaklı bir başlangıç mimarisi.
Kapsam **araç platformundan algılama çıktısına kadar** olan katmanlardır; tahmin,
davranış planlama, hareket planlama ve kontrol bu sürüme dahil değildir.

## Uygulanan katmanlar

1. **Araç / simülasyon platformu** — Lincoln MKZ 2020, güvenli blueprint fallback'i.
2. **ODD ve görev sınırı** — harita, hız, hava, görüş, lokalizasyon ve sensör sağlığı.
3. **Sensör sistemi** — tek semantik LiDAR, 6 RGB kamera, 5 radar, 12 yakın alan
   sensörü, GNSS ve IMU.
4. **Senkronizasyon ve kalibrasyon** — synchronous mode, sabit 50 ms zaman adımı,
   exact-frame bariyeri, YAML tabanlı rigid extrinsics.
5. **Lokalizasyon arayüzü** — deterministik CARLA ground-truth pozisyonu; GNSS/IMU
   frame sağlığı ile birlikte. Bu bileşen simülasyon özeldir.
6. **Algılama** — CARLA semantik LiDAR instance ID'lerinden deterministik 3B kutular,
   sınıf, menzil ve frame-to-frame göreli hız.

## Tasarım sınırı

Algılama modülü bir yapay zekâ modeli değildir. CARLA'nın
`sensor.lidar.ray_cast_semantic` çıktısındaki `object_idx` ve `object_tag`
alanlarını kullanır. Bu yaklaşım:

- aynı sahnede tekrarlanabilir sonuç verir,
- GPU/model dosyası gerektirmez,
- mimari arayüzleri ve veri akışını güvenilir biçimde kurar,
- gerçek araca doğrudan taşınamaz.

Gerçek sensör algılamasına geçerken `SemanticLidarInstanceDetector` yerine aynı
`PerceptionFrame` sözleşmesini üreten kamera/LiDAR/radar modelleri eklenmelidir.

## Desteklenen çalışma biçimi

Önerilen taban CARLA **0.9.16**'dır. Kod 0.9.x Python API sözleşmesini kullanır ve
0.10.0 ile ortak API yüzeylerinde çalışacak şekilde yazılmıştır; fakat paket CARLA
sunucusu ve Python istemcisinin aynı sürüm olmasını bekler.

Python: 3.10 veya 3.11.

## Kurulum

```bash
cd carla_l4_perception_stack
python -m venv .venv
source .venv/bin/activate          # Linux
# .venv\Scripts\activate           # Windows PowerShell
pip install -e ".[dev]"
```

CARLA Python API kurulumu kullanılan CARLA dağıtımıyla eşleşmelidir. Paket
kurulumundan sonra şu komut çalışmalıdır:

```bash
python -c "import carla; print(carla.__file__)"
```

## CARLA'yı çalıştırma

Önce CARLA sunucusunu açın. Ardından:

```bash
l4stack --config-dir config validate
l4stack --config-dir config coverage
l4stack --config-dir config run --frames 400
```

Alternatif:

```bash
python -m l4stack.cli --config-dir config run --frames 400
```

Çıktılar `output/` altında oluşur:

- `calibration.json`: tüm sensör extrinsic ve attribute değerleri,
- `frames.jsonl`: her simülasyon frame'i için ODD, lokalizasyon ve algılama sonucu.

## Yapılandırma

- `config/simulator.yaml`: CARLA bağlantısı, map, fixed delta, seed ve hava.
- `config/vehicle.yaml`: araç blueprint'i, spawn noktası ve ilk kontrol.
- `config/sensors.yaml`: sensör placement, yaw/pitch/roll ve sensör ayarları.
- `config/odd.yaml`: çalışılmasına izin verilen koşullar.
- `config/perception.yaml`: filtreleme, sınıflar ve tracker parametreleri.
- `config/logging.yaml`: log seviyesi ve çıktı sıklığı.

CARLA koordinatı:

- `+x`: ön,
- `+y`: sağ,
- `+z`: yukarı,
- açılar: derece.

## Determinizm kuralları

- World synchronous mode zorunludur.
- `fixed_delta_seconds=0.05` kullanılır.
- Hava koşulları ve seed sabittir.
- Gerekli sensörler aynı frame'i üretmeden işleme yapılmaz.
- Eksik gerekli sensör frame'i eski veriyle doldurulmaz; timeout hatası üretilir.
- Tespitler `object_idx` sırasına göre kararlı biçimde sıralanır.
- Ego araç varsayılan olarak frenli ve sabittir; bu paket kontrol katmanı içermez.

## Test

```bash
pytest
ruff check .
```

Testler CARLA sunucusu gerektirmez. CARLA ile entegrasyon testi için:

```bash
l4stack --config-dir config run --frames 20
```

## Dizin yapısı

```text
src/l4stack/
├── app/             # Uçtan uca çalışma döngüsü
├── config/          # YAML yükleme ve doğrulama
├── core/            # Veri sözleşmeleri ve yaşam döngüsü
├── localization/    # Lokalizasyon arayüzü
├── odd/             # ODD monitor
├── perception/      # Deterministik 3B algılama ve tracker
├── sensors/         # Factory, sync, calibration, decoder, coverage
└── simulation/      # CARLA bağlantısı, dünya ve araç adaptörü
```

## Sonraki teknik adım

Bu temel doğrulandıktan sonra gerçekçi algılama için sırayla şunlar eklenmelidir:

1. normal ray-cast LiDAR + 3B detector,
2. kamera tabanlı trafik ışığı/levha/şerit algılama,
3. radar association,
4. kalibrasyon ve zaman gecikmesi hata enjeksiyonu,
5. perception uncertainty ve safety monitor.
