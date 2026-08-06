# NVIDIA CitySemSegFormer Kurulumu

## Görev

Ön, ön-sol ve ön-sağ kameralarda 19 şehir sınıfı için piksel seviyesinde semantic
segmentation üretir: road, sidewalk, building, wall, fence, pole, traffic light,
traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train,
motorcycle ve bicycle.

Resmî model kartı:
https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/citysemsegformer

## İndirilecek dosyalar

NGC file browser'dan:

```text
citysemsegformer.onnx
labels.txt
nvinfer_config.txt
```

Yerleşim:

```text
models/perception/citysemsegformer/model/
├── citysemsegformer.onnx
├── labels.txt
└── nvinfer_config.txt
```

## Resmî preprocessing

```text
input: RGB 3×1024×1024
offsets: 123.675;116.28;103.53
net scale factor: 0.01735207357279195
önerilen network mode: FP16
```

Maskeyi orijinal çözünürlüğe döndürürken nearest-neighbor kullan; class index üzerinde
bilinear interpolasyon yapılmaz.

## İzole runtime

NVIDIA model kartındaki TAO 5.2/TensorRT/DeepStream 6.1+ ortamını veya doğrulanmış bir
ONNX Runtime GPU ortamını ayrı süreçte kullan. Ana stack'e TensorRT/DeepStream kurma.

```bash
export L4STACK_CITYSEMSEGFORMER_COMMAND='python /absolute/path/to/citysemsegformer_backend.py'
```

Her kamera maskesi ayrı artifact olmalıdır:

```json
{
  "rasters": [
    {"name": "camera_front_semantic", "uri": "file:///.../front.npy",
     "media_type": "application/x-npy", "shape": [540, 960],
     "dtype": "uint8", "byte_size": 518400},
    {"name": "camera_front_left_semantic", "uri": "file:///.../left.npy",
     "media_type": "application/x-npy", "shape": [540, 960],
     "dtype": "uint8", "byte_size": 518400}
  ],
  "diagnostics": {"labels_version": "ngc-citysemsegformer-v1"}
}
```

Model kartındaki "commercial-ready" ifadesi safety certification değildir.
