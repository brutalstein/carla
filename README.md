# CARLA L4 Localization Foundation

CARLA için modüler, deterministik ve yapılandırma odaklı araç, ODD, sensör ve
lokalizasyon temeli. Bu sürümde algılama, tahmin, planlama ve kontrol bulunmaz.

## Uygulanan katmanlar

1. **Araç / simülasyon platformu** — Lincoln MKZ 2020 ve kontrollü blueprint fallback.
2. **ODD sınırı** — harita, hız, hava, görüş, lokalizasyon ve sensör sağlığı.
3. **Raw sensör sistemi** — tek normal ray-cast LiDAR, 6 RGB kamera, 5 radar,
   GNSS ve IMU.
4. **Senkronizasyon ve kalibrasyon kaydı** — synchronous mode, sabit 50 ms adım,
   exact-frame bariyeri ve YAML tabanlı rigid extrinsics.
5. **Lokalizasyon** — GNSS + IMU + pusula ölçümlerini birleştiren planar
   error-state extended Kalman filter.

## Ground-truth politikası

Çalışma zamanı semantik LiDAR, semantik/instance kamera, actor transformu veya actor
velocity kullanmaz. Bu sensör blueprint'leri konfigürasyon yüklenirken reddedilir.
Ground truth yalnızca `tests/test_localization_benchmark.py` içinde tahmin hatasını
ölçen referans yörünge olarak kullanılır ve runtime paketine veri sağlamaz.

## Lokalizasyon algoritması

Filtre durumu:

```text
[p_e, p_n, v_e, v_n, yaw, b_ax, b_ay, b_gz]
```

- IMU ivmesi ve yaw-rate ile nominal durum propagasyonu,
- WGS-84 ECEF → local ENU dönüşümü,
- GNSS anten lever-arm modeli,
- GNSS yatay konum güncellemesi,
- pusula yaw güncellemesi,
- chi-square/NIS outlier gating,
- Joseph-form covariance update,
- accelerometer ve gyro bias random walk,
- covariance tabanlı `NOMINAL/DEGRADED` sağlık durumu.

Koordinat sistemi `LOCAL_ENU`'dur: `+x east`, `+y north`, `+z up`. Yaw, east
ekseninden saat yönünün tersine pozitiftir. İlk kabul edilen GNSS ölçümü local origin'i
oluşturur.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate          # Linux
# .venv\Scripts\activate           # Windows PowerShell
pip install -e ".[dev]"
```

CARLA Python istemcisi sunucu sürümüyle eşleşmelidir.

## Çalıştırma

```bash
l4stack --config-dir config validate
l4stack --config-dir config coverage
l4stack --config-dir config run --frames 400
```

Çıktılar:

- `output/calibration.json`: sensör extrinsic ve attribute kayıtları,
- `output/frames.jsonl`: ODD ve sensör tabanlı lokalizasyon sonuçları.

## Yapılandırma

- `config/simulator.yaml`: CARLA bağlantısı, map, fixed delta, seed ve hava.
- `config/vehicle.yaml`: ego araç ve başlangıç kontrolü.
- `config/sensors.yaml`: raw sensörler, placement ve sensör gürültüleri.
- `config/localization.yaml`: ESKF covariance, noise, gate ve sağlık eşikleri.
- `config/odd.yaml`: izin verilen operasyon koşulları.
- `config/logging.yaml`: log seviyesi ve çıktı adı.

## Determinizm

- World synchronous mode zorunludur.
- `fixed_delta_seconds=0.05` kullanılır.
- Sensör noise seed değerleri sabittir.
- GNSS ve IMU aynı frame'i üretmeden filtre çalışmaz.
- Eksik gerekli frame eski veriyle doldurulmaz; timeout oluşur.
- Ego araç varsayılan olarak frenlidir; kontrol katmanı yoktur.

## Test

```bash
pytest
ruff check .
```

Test paketi geodesy round-trip, exact-frame bariyeri, ODD, konfigürasyon ve
sentetik sabit hızlı yörüngede ground-truth benchmark RMSE kontrolünü kapsar.
