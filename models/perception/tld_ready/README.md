# TLD-READY Kurulumu

## Görev

Ön, ön-sol ve ön-sağ kameralarda trafik ışığı bbox, durum, pictogram ve mümkünse ego
şeridi açısından relevance üretir.

- Resmî repo: https://github.com/KASTEL-MobilityLab/traffic-light-detection
- Yayın: IEEE ITSC 2024.
- Lisans: AGPL-3.0.
- Seçilen detector: `traffic_lights_yolov8x.pt`.

Repository tablosu YOLOv8 XL için 0.87 precision, 0.74 recall ve 0.82 mAP50; road
marking relevance için 0.96 precision/recall raporlar.

## Kurulum ve ağırlıklar

```bash
cd models/perception/tld_ready/external
git clone https://github.com/KASTEL-MobilityLab/traffic-light-detection.git
cd traffic-light-detection/model_weights
chmod +x download_weights.sh
./download_weights.sh
```

Kopyala:

```text
traffic_lights_yolov8x.pt
  → models/perception/tld_ready/model/traffic_lights_yolov8x.pt

road_markingsyolov8m.pt (opsiyonel relevance)
  → models/perception/tld_ready/model/road_markingsyolov8m.pt
```

## Runtime komutu

Resmî Docker veya ayrı Ultralytics ortamındaki runner'ı tanımla:

```bash
export L4STACK_TLD_READY_COMMAND='python /absolute/path/to/tld_ready_backend.py'
```

Çıktı:

```json
{
  "traffic_lights": [{
    "camera_name": "camera_front",
    "bbox_xyxy": [100, 40, 120, 85],
    "state": "RED",
    "pictogram": "circle",
    "confidence": 0.93,
    "relevant_to_ego": true
  }]
}
```

Relevance modeli hazır değilse `relevant_to_ego: null` kullan. Yan yol ışığını ego
ışığı gibi işaretlemekten kaçınmak için `false` uydurulmaz.

## Lisans notu

AGPL-3.0 ve Ultralytics lisans koşulları ürün dağıtımı öncesi hukuk incelemesi ister.
Bu repository'yi teknik olarak kullanabilmek, kapalı kaynak üründe otomatik lisans
uygunluğu sağlamaz.
