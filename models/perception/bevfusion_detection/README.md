# BEVFusion Detection — RTX 5090

## Kaynak

- MIT BEVFusion: https://github.com/mit-han-lab/bevfusion
- NVIDIA CUDA-BEVFusion: https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion

Ana checkpoint:

```text
model/bevfusion-det.pth
```

Runtime engine:

```text
model/bevfusion-det-sm120.engine
```

## Neden NVIDIA CUDA-BEVFusion runtime?

Orijinal eğitim deposu eski PyTorch/MMCV ortamındadır. Online inference için custom CUDA
BEV pooling, sparse convolution ve TensorRT hattı bulunan NVIDIA implementasyonu hedeflenir.
Checkpoint yalnız kaynak ağırlıktır; engine RTX 5090 compute capability 12.0 ve TensorRT
11.1 üzerinde yeniden üretilir.

## Kurulum

```bash
git clone https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution.git \
  models/perception/bevfusion_detection/external/Lidar_AI_Solution
```

Resmî checkpoint'i indirin ve `model/bevfusion-det.pth` konumuna yerleştirin. NVIDIA
CUDA-BEVFusion build/export talimatını TensorRT worker image içinde çalıştırın. Build
komutuna SM120 mimarisi eklenmeli ve eski `.engine` kopyalanmamalıdır.

Engine smoke test gerçek altı CARLA kamera artifact'ı, LiDAR artifact'ı ve
`output/calibration.json` ile yapılır. Sadece nuScenes örneğinin çalışması CARLA adapter
başarısı sayılmaz.

## Worker

Persistent worker komutu:

```bash
export L4STACK_BEVFUSION_DETECTION_COMMAND='docker exec -i l4stack-perception-trt <bevfusion-jsonl-worker>'
```

Worker gereksinimleri:

- modeli startup'ta bir kez yüklemek,
- TensorRT execution context ve binding buffer'larını tekrar kullanmak,
- FP16 çalışmak,
- sabit shape mümkünse CUDA Graph capture yapmak,
- `shm://` kamera/LiDAR girdilerini async H2D kopyalamak,
- 3B kutuları `EGO_LOCAL` koordinatına normalize etmek,
- diagnostics içinde CUDA event `preprocess_ms`, `inference_ms`, `postprocess_ms` vermek.

## Etkinleştirme

`perception-doctor` engine ve command'i READY gösterdikten sonra:

```yaml
perception:
  enabled: true
  models:
    bevfusion_detection:
      enabled: true
```
