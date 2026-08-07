# BEVFusion Detection Kurulumu

## Görev ve seçilen sürüm

Altı çevresel RGB kamera ile `lidar_top` nokta bulutundan 3B nesne kutuları üretir.
Bu sürümde manifest'in beklediği model, MIT BEVFusion repository'sindeki:

```text
config: configs/nuscenes/det/transfusion/secfpn/camera+lidar/
        swint_v0p075/convfuser.yaml
weight: bevfusion-det.pth
```

Çıktı `OBJECT_DETECTION_3D`, koordinat sistemi normalize edildikten sonra `EGO_LOCAL`.

## Resmî kaynak ve sonuç

- https://github.com/mit-han-lab/bevfusion
- ICRA 2023, Apache-2.0.
- nuScenes validation: 68.52 mAP, 71.38 NDS.

## Ağırlığı indir

Repo kökünde:

```bash
mkdir -p models/perception/bevfusion_detection/model
wget -O models/perception/bevfusion_detection/model/bevfusion-det.pth \
  'https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v'
```

Dosyanın burada olması gerekir:

```text
models/perception/bevfusion_detection/model/bevfusion-det.pth
```

## Kod ortamı

Ana `l4stack` ortamına kurma. Ayrı container/conda ortamı kullan. MIT kodu için resmî
bağımlılık aralığı Python 3.8, PyTorch 1.9–1.10.2, MMCV 1.4.0 ve MMDetection 2.20.0'dır.

```bash
cd models/perception/bevfusion_detection/external
git clone https://github.com/mit-han-lab/bevfusion.git
```

Resmî Dockerfile da kullanılabilir. NVIDIA CUDA-BEVFusion TensorRT alternatifi ayrı bir
model/config dağıtımıdır; MIT checkpoint ile aynı dosya gibi kullanılmamalıdır:

https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion

## Runtime komutu

İzole ortam içindeki gerçek model runner'ı JSONL protokolünü uygulamalı ve çıktıyı
`detections_3d` şemasına dönüştürmelidir. Komutu tanımla:

```bash
export L4STACK_BEVFUSION_DETECTION_COMMAND='conda run -n bevfusion python /absolute/path/to/bevfusion_detection_backend.py'
```

`run_backend.sh` bu komutu başlatır. Model olmadan protokol testi:

```bash
L4STACK_PERCEPTION_MOCK=1 \
  bash models/perception/bevfusion_detection/run_backend.sh
```

## CARLA adapter sorumlulukları

- BGRA8 → RGB.
- Altı kamerayı config sırasına koyma.
- Kamera intrinsic ve ego extrinsic matrislerini `calibration.json` içinden üretme.
- Görüntüleri 256×704 pipeline'ına uygun resize/pad etme.
- CARLA LiDAR `[x,y,z,intensity]` eksenlerini model eksenine çevirme.
- Model kutularını tekrar CARLA `EGO_LOCAL` frame'ine döndürme.
- Sınıf, confidence, merkez, W/L/H, yaw ve varsa hızı JSONL çıktısına yazma.

## Etkinleştirme

Önce:

```bash
l4stack --config-dir config perception-doctor
```

sonra `config/perception.yaml` içinde hem global `perception.enabled` hem de
`bevfusion_detection.enabled` değerini `true` yap.
