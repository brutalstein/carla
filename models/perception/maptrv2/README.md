# MapTRv2 Kurulumu

## Görev ve seçilen sürüm

Altı çevresel kameradan online vektörel yol elemanları üretir:

- lane divider
- road boundary
- pedestrian crossing

Seçilen model standart **MapTRv2 R50 + BEVPool, 24 epoch, nuScenes** checkpoint'idir.
Resmî repository sonucu 61.4 mAP ve RTX 3090'da batch 1/altı kamera için 14.1 FPS'dir.
Bu checkpoint centerline ekleyen `MapTRv2*` değildir.

Kaynak: https://github.com/hustvl/MapTR/tree/maptrv2

## İzole ortam

Resmî kurulum:

```bash
conda create -n maptr python=3.8 -y
conda activate maptr
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install timm shapely==1.8.5.post1 av2
```

```bash
cd models/perception/maptrv2/external
git clone https://github.com/hustvl/MapTR.git
cd MapTR
git checkout maptrv2
cd mmdetection3d && python setup.py develop
cd ../projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
```

Resmî ayrıntı:
https://github.com/hustvl/MapTR/blob/maptrv2/docs/install.md

## Checkpoint

Resmî `MapTRv2 | R50 | bevpool | 24ep` checkpoint bağlantısı:

https://drive.google.com/file/d/1AmQ3fT-J-MM4B8kh_9Gm2G5guM92Agww/view?usp=sharing

İndirilen dosyayı şu adla yerleştir:

```text
models/perception/maptrv2/model/maptrv2_r50_bevpool_24ep.pth
```

Google Drive bazen model yerine HTML indirir. Kontrol et:

```bash
file models/perception/maptrv2/model/maptrv2_r50_bevpool_24ep.pth
ls -lh models/perception/maptrv2/model/maptrv2_r50_bevpool_24ep.pth
```

## Runtime komutu

```bash
export L4STACK_MAPTRV2_COMMAND='conda run -n maptr python /absolute/path/to/maptrv2_backend.py'
```

Backend çıktısı `EGO_LOCAL` metre cinsinden olmalıdır:

```json
{
  "vector_map": [{
    "category": "lane_divider",
    "confidence": 0.92,
    "points_xyz_m": [[0.0, -1.8, 0.0], [10.0, -1.7, 0.0]]
  }],
  "diagnostics": {"inference_ms": 70.0}
}
```

nuScenes model koordinatlarını CARLA koordinatı gibi döndürmek yasaktır; dönüşüm backend
içinde açıkça yapılmalıdır.
