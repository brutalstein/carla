# BEVFusion Segmentation Kurulumu

## Görev ve seçilen sürüm

Altı kamera ve LiDAR girdisinden ego merkezli BEV semantik raster üretir.
Detection checkpoint'inden farklıdır.

```text
config: configs/nuscenes/seg/fusion-bev256d2-lss.yaml
weight: bevfusion-seg.pth
```

Resmî nuScenes validation sonucu 62.95 mIoU'dur.

Kaynak: https://github.com/mit-han-lab/bevfusion

## İndirme

```bash
mkdir -p models/perception/bevfusion_segmentation/model
wget -O models/perception/bevfusion_segmentation/model/bevfusion-seg.pth \
  'https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9'
```

Beklenen yol:

```text
models/perception/bevfusion_segmentation/model/bevfusion-seg.pth
```

## İzole bağımlılıklar

- Python >=3.8,<3.9
- PyTorch >=1.9,<=1.10.2
- MMCV 1.4.0
- MMDetection 2.20.0
- OpenMPI 4.0.4 / mpi4py 3.0.3
- Pillow 8.4.0

```bash
cd models/perception/bevfusion_segmentation/external
git clone https://github.com/mit-han-lab/bevfusion.git
```

Ana stack ortamına bu paketleri kurma.

## Runtime komutu

```bash
export L4STACK_BEVFUSION_SEGMENTATION_COMMAND='conda run -n bevfusion python /absolute/path/to/bevfusion_segmentation_backend.py'
```

Backend bir adet `EGO_BEV_RASTER` artifact döndürmelidir:

```json
{
  "rasters": [{
    "name": "bev_semantic",
    "uri": "file:///.../bev_semantic.npy",
    "media_type": "application/x-npy",
    "shape": [H, W],
    "dtype": "uint8",
    "byte_size": 12345
  }],
  "diagnostics": {"inference_ms": 120.0, "class_order": ["drivable", "divider"]}
}
```

Class order ve BEV metre/hücre çözünürlüğü diagnostics içinde sürümlenmelidir.
