# Perception Model Kaynakları ve Seçim Gerekçesi

Bu belge yalnızca resmî model repository/model kartlarını kaynak kabul eder. Buradaki
benchmark değerleri ilgili veri setine aittir; CARLA veya gerçek araç güvenlik garantisi
değildir.

## BEVFusion Detection

- Resmî kaynak: https://github.com/mit-han-lab/bevfusion
- Yayın: ICRA 2023.
- Lisans: Apache-2.0.
- Seçilen config:
  `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml`
- Checkpoint: `bevfusion-det.pth`.
- Resmî nuScenes validation: 68.52 mAP, 71.38 NDS.
- Girdi: altı kamera + LiDAR + kalibrasyon.

İndirme:

```bash
wget -O models/perception/bevfusion_detection/model/bevfusion-det.pth \
  'https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v'
```

NVIDIA TensorRT alternatifi:
https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion

NVIDIA implementasyonu CUDA >=11, cuDNN >=8.2, TensorRT >=8.5 ve sm_80+ GPU ister.
Bu paket runtime optimizasyon alternatifi olarak tutulur; MIT checkpoint ve NVIDIA
ResNet50/TensorRT paketinin aynı ağırlık/config olmadığı unutulmamalıdır.

## BEVFusion Segmentation

- Aynı resmî BEVFusion repository'si.
- Seçilen config: `configs/nuscenes/seg/fusion-bev256d2-lss.yaml`.
- Checkpoint: `bevfusion-seg.pth`.
- Resmî nuScenes validation: 62.95 mIoU.

İndirme:

```bash
wget -O models/perception/bevfusion_segmentation/model/bevfusion-seg.pth \
  'https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9'
```

Resmî eski bağımlılık seti:

- Python >=3.8,<3.9
- PyTorch >=1.9,<=1.10.2
- MMCV 1.4.0
- MMDetection 2.20.0
- OpenMPI 4.0.4 / mpi4py 3.0.3
- Pillow 8.4.0

## MapTRv2

- Resmî kaynak: https://github.com/hustvl/MapTR/tree/maptrv2
- Yayınlar: MapTR ICLR 2023 Spotlight; MapTRv2 IJCV 2024.
- Lisans: MIT.
- Seçilen model: MapTRv2 R50 + BEVPool, 24 epoch, nuScenes.
- Resmî repository sonucu: 61.4 mAP, 14.1 FPS.
- FPS ölçümü RTX 3090, batch 1 ve altı kamera içindir.

Resmî kurulum:
https://github.com/hustvl/MapTR/blob/maptrv2/docs/install.md

Bağımlılıklar:

- Python 3.8
- PyTorch 1.9.1 + CUDA 11.1
- torchvision 0.10.1
- mmcv-full 1.4.0
- mmdet 2.14.0
- mmsegmentation 0.14.1
- timm
- shapely 1.8.5.post1
- av2

Standart checkpoint lane divider, road boundary ve pedestrian crossing gibi online map
elemanları içindir. Centerline ekleyen `MapTRv2*` ayrı checkpoint'tir; resmî tabloda
54.3 mAP ve FPS `WIP` olarak verildiği için bu sürümde varsayılan seçilmemiştir.

## TLD-READY

- Resmî kaynak: https://github.com/KASTEL-MobilityLab/traffic-light-detection
- Yayın: IEEE ITSC 2024.
- Lisans: AGPL-3.0.
- Seçilen detector: `traffic_lights_yolov8x.pt`.
- Opsiyonel relevance modeli: `road_markingsyolov8m.pt`.

Resmî ağırlık script'i:

```bash
cd model_weights
chmod +x download_weights.sh
./download_weights.sh
```

Repository sonuç tablosu YOLOv8 XL trafik ışığı modeli için 0.87 precision, 0.74
recall ve 0.82 mAP50; road-marking relevance için 0.96 precision/recall bildirir.

AGPL-3.0 lisansı kapalı kaynak ürün dağıtımında hukuki değerlendirme gerektirir. Kodun
teknik olarak çalışması lisans uygunluğu anlamına gelmez.

## NVIDIA CitySemSegFormer

- Resmî model kartı:
  https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/citysemsegformer
- Dağıtım: imzalı deployable ONNX.
- Girdi: RGB, sabit 3×1024×1024.
- Çıktı: 19 sınıflı semantic segmentation maskesi.
- Örnek sınıflar: road, sidewalk, wall, traffic light, traffic sign, person, car,
  truck, bus, motorcycle ve bicycle.

İndirilecek dosyalar:

```text
citysemsegformer.onnx
labels.txt
nvinfer_config.txt
```

Resmî preprocessing:

```text
offsets: 123.675;116.28;103.53
net scale factor: 0.01735207357279195
network mode: FP16 önerilir
```

TAO 5.2 / TensorRT / DeepStream 6.1+ dağıtımı model kartında belirtilir. Model kartı
"commercial-ready" ifadesini kullanır; bu safety certification değildir.

## Ortak riskler

1. **Domain gap:** Modeller CARLA görüntüsüyle eğitilmemiştir.
2. **Kalibrasyon:** Yanlış intrinsic/extrinsic 3B ve vector-map çıktısını doğrudan bozar.
3. **Koordinat sistemi:** CARLA ve nuScenes eksenleri aynı kabul edilmemelidir.
4. **Sınıf eşleme:** nuScenes/şehir dataset sınıfları ODD ihtiyaçlarımızla birebir değildir.
5. **Latency:** Resmî FPS farklı GPU, çözünürlük ve yazılım sürümlerinde ölçülmüştür.
6. **Lisans:** Özellikle TLD-READY/Ultralytics akışı ürün lisans incelemesi ister.
7. **Safety:** Hiçbir checkpoint tek başına ISO 26262 veya ISO 21448 sertifikalı değildir.
