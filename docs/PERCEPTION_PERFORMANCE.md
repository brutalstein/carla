# Perception Performans Tasarımı

## Hedef

20 Hz CARLA ana tick'ini model inference sürelerinden ayırmak, tek RTX 5090'u bounded ve
ölçülebilir kullanmak, eski frame backlog'unu sıfırlamak.

## Ana prensipler

1. Model yükleme startup'ta bir kez.
2. Online frame diske yazılmaz.
3. Sabit-slotlu shared memory.
4. Queue depth 1.
5. Global GPU admission.
6. Critical iş önce.
7. Ölçülen p95 tahmine geri beslenir.
8. Snapshot latest-valid sonuçlardan oluşur.

## Copy yolu

```text
CARLA bytes/calibration → önceden ayrılmış /dev/shm slotu → worker read-only mmap → async H2D
```

CARLA callback memory'si doğrudan CUDA memory değildir; en az bir host copy vardır.
Ama disk write/read, per-frame file open, per-frame SHA-256 ve per-frame POSIX segment
oluşturma kaldırılmıştır. Calibration JSON da host dosya yolunu container'a taşımak yerine
immutable shared-memory artifact'ı olarak yayınlanır.

## Neden model başına process kalıyor?

Bağımlılıklar ve hata alanları farklıdır. Process izolasyonu:

- CUDA/TensorRT ve PyTorch stack'lerini ayırır,
- bir model crash'inin host stack'i öldürmesini önler,
- bağımsız restart/health sağlar.

MPS bu izolasyonu korurken process context scheduling maliyetini düşürür. Tek process'e
zorla birleştirmek MapTR/BEVFusion/TLD dependency conflict'lerini ana runtime'a taşır.

## GPU admission örneği

Aynı frame'de due:

```text
BEV Det. 24 ms critical
TLD       7 ms critical
MapTR    30 ms required
CitySem  26 ms opportunistic
```

42 ms budget, 2 concurrency slot ve 1.15 margin altında BEV + TLD seçilir. MapTR sonraki
uygun release'e kalır; CitySem atlanır. Hiçbiri executor backlog'una eklenmez.

## İlk frekanslar

```text
BEVFusion Detection     10 Hz
TLD-READY               10 Hz
MapTRv2                  5 Hz
BEVFusion Segmentation   2 Hz
CitySemSegFormer         2 Hz
```

TLD ve BEV detection critical; MapTR required; raster modeller opportunistic'tir.

## Memory hesabı

Altı 960×540 BGRA kamera:

```text
2,073,600 bytes/frame/camera
6 × 8 slot ≈ 94.9 MiB
```

LiDAR:

```text
4 MiB × 8 slot = 32 MiB
```

Input ring toplamı yaklaşık 127 MiB'dir. Calibration ring ve worker output raster ring'leri
ayrıca eklenir. `cuda-doctor` en az 512 MiB boş `/dev/shm` ister.

## CUDA worker optimizasyonları

### TensorRT

- FP16 başlangıç.
- Engine hedef SM120 üzerinde build.
- Bir execution context/model worker.
- Binding buffers startup'ta allocate.
- Dynamic shape yalnız gerekiyorsa.
- CUDA Graph sabit shape hattında.
- `enqueueV3`/async stream.
- Host timing yerine CUDA event timing.

### PyTorch/MapTR

- PyTorch 2.12.1 cu130.
- `torch.inference_mode()`.
- `autocast(device_type="cuda", dtype=torch.float16/bfloat16)`.
- `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:True`.
- `TORCH_CUDA_ARCH_LIST=12.0`.
- Custom ops source build.
- Warm-up sonrası compile/CUDA graph sadece doğruluk regression'ı geçerse.

## Ölçüm ayrımı

Ayrı ayrı kaydedilmelidir:

```text
sensor_source_age
shared_memory_copy_ms
executor_queue_wait_ms
worker_request_round_trip_ms
preprocess_cuda_ms
inference_cuda_ms
postprocess_cuda_ms
end_to_end_ms
output_age_at_consumption
```

Current admission controller güvenli tarafta kalmak için request end-to-end p95 kullanır.
Worker diagnostics içine gerçek CUDA event süreleri eklenince GPU-only süreler ayrıca
raporlanır.

## Gerçek performans kapısı

Bir model şu koşullar sağlanmadan nominal kabul edilmez:

- En az 400 gerçek CARLA frame'i.
- Hiçbir sentetik backend veya üretilmiş fake output yok.
- Failure rate ≤ %1.
- Critical model GPU budget skip rate kabul edilmez.
- Deadline p95 model sözleşmesinin altında.
- Output source age lifespan altında.
- VRAM OOM, Xid veya MPS fault yok.
- CARLA ground truth yalnız offline metrik hesabında kullanılır.
