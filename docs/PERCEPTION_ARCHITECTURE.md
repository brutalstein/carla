# Perception Katmanı — RTX 5090 CUDA Runtime

## Amaç

Tek RTX 5090 üzerinde beş hazır modeli CARLA ana tick'ini bloklamadan, disk I/O
oluşturmadan ve kontrolsüz CUDA process yarışına girmeden çalıştırmak.

```text
CARLA sensors
      ↓ latest-at-or-before
shared-memory producer
      ↓ shm:// ArtifactRef
rate gate
      ↓
global GPU admission
      ↓
CUDA MPS persistent workers
      ↓
normalized ModelOutput
      ↓
partial latest-valid snapshot
```

## Model rolleri

- BEVFusion Detection: ana 3B nesne algılama, critical.
- TLD-READY: trafik ışığı, critical.
- MapTRv2: vektörel yol geometrisi, required.
- BEVFusion Segmentation: yardımcı BEV raster, opportunistic.
- CitySemSegFormer: kamera semantiği/doğrulama, opportunistic.

## Shared-memory transport

Frame başına disk dosyası, SHA-256 ve allocation yoktur. Her sensör lazy-first-use
sırasında sabit boyutlu bir POSIX shared-memory ring alır. Kamera için BGRA8, LiDAR için
CARLA `[x,y,z,intensity]` float32 byte düzeni korunur.

```text
camera slot: 2,500,000 bytes × 8
lidar slot:  4,194,304 bytes × 8
```

Slot state:

```text
FREE
 → producer reserved
 → committed(reader_count = accepted model count)
 → model future completions release
 → FREE
```

Slot URI generation, slot ve capacity taşır. Eski token yeni generation'ı release edemez.
Ring doluysa en fazla 5 ms beklenir; ana tick sınırsız bloklanmaz.

Worker `open_shared_artifact()` ile `/dev/shm/<segment>` dosyasını read-only mmap eder.
Consumer process `multiprocessing.resource_tracker` kullanmaz; producer'a ait segmenti
worker shutdown sırasında yanlışlıkla unlink edemez.

CARLA callback belleği dış kütüphaneye ait olduğu için minimum bir host-copy zorunludur.
Bu copy doğrudan önceden ayrılmış slotadır. Worker tarafında pinned buffer ve non-blocking
CUDA stream H2D kullanılması gerekir.

## Global GPU admission

Her model ayrı process olsa da tek fiziksel GPU kullanılır. Bu nedenle model executor
queue'larından önce tek global admission uygulanır.

Karar girdileri:

- execution class,
- priority,
- rate gate,
- global max concurrent,
- model max inflight,
- frame GPU budget,
- başlangıç estimated GPU ms,
- ölçülen rolling p95,
- safety margin.

Sınıflar:

```text
CRITICAL       BEVFusion Detection, TLD-READY
REQUIRED       MapTRv2
OPPORTUNISTIC  BEV Segmentation, CitySemSegFormer
```

Critical işler budget tahminini aşsa bile boş concurrency slotu varsa korunur.
Opportunistic modeller baskı altında frame atlar. Hiçbir skipped iş queue'ya girmez.

Varsayılan 20 Hz release bütçesi:

```text
frame budget = 42 ms
max concurrent = 2
safety margin = 1.15
```

Bu değerler tahmindir; gerçek RTX 5090 p95 ölçümünden sonra kalibre edilir.

## CUDA MPS

Model process izolasyonu korunur, fakat hepsi aynı GPU'ya bağlanır. CUDA MPS:

- process CUDA context scheduling maliyetini azaltır,
- küçük TLD kernel'lerinin ağır modele kontrollü overlap etmesine yardım eder,
- process'lerin aynı MPS server ve aynı UID altında çalışmasını gerektirir.

Compose `ipc: host`, aynı UID/GID ve MPS pipe mount kullanır. MPS fatal fault başka
client'ları etkileyebileceği için worker health ve NVIDIA Xid logları ayrıca izlenmelidir.

## Lifecycle ve CUDA readiness

Model component configure aşamasında:

1. checkpoint/engine preflight,
2. worker process start,
3. protocol ping,
4. gerçek CUDA readiness,
5. model_loaded doğrulaması

yapar.

CUDA zorunlu worker cevabı:

```text
device=cuda
cuda_available=true
cpu_fallback=false
model_loaded=true
compute_capability>=12.0
precision=fp16/bf16/fp8
```

Bir model startup veya inference hatasında yalnız kendi route'unu kapatır. Diğer modeller
çalışmaya devam eder. MPS fatal GPU fault davranışı nedeniyle GPU Xid/MPS fault ayrıca
host supervisor tarafından izlenmelidir.

## Worker process sözleşmesi

JSONL yalnız kontrol ve metadata içindir. Kamera, LiDAR veya raster base64 olarak JSON'a
gömülmez.

### Ping

```json
{"protocol_version":2,"type":"ping"}
```

### Ready

```json
{
  "protocol_version":2,
  "type":"ready",
  "device":"cuda",
  "cuda_available":true,
  "cpu_fallback":false,
  "model_loaded":true,
  "compute_capability":12.0,
  "precision":"fp16"
}
```

### Inference

Request, `PerceptionInput` ve `shm://` artifact referanslarını taşır. Worker response'u
normalize adapter'ın beklediği küçük detection/polyline metadata'sını veya büyük output
için worker-owned shared-memory artifact referansını döndürür. Yeni output eskisinin
yerini aldığında, snapshot süresi dolduğunda veya stack kapandığında host protokol v2
`release` mesajı gönderir. Worker ring slotunu `released` onayından sonra yeniden kullanır;
maskenin okunurken overwrite edilmesine izin verilmez.

## CUDA runtime kuralları

- Model ve engine startup'ta bir kez yüklenir.
- TensorRT execution context/binding buffer tekrar kullanılır.
- PyTorch `torch.inference_mode()` ve autocast kullanır.
- CPU fallback yasaktır.
- CUDA memory allocation request başına yapılmaz.
- PyTorch allocator `cudaMallocAsync` kullanır.
- Sabit shape engine'lerde CUDA Graph capture değerlendirilir.
- TensorRT engine hedef RTX 5090 / SM120 üzerinde üretilir.
- Worker diagnostics CUDA event sürelerini döndürür.

## Zaman ve senkronizasyon

Dünya ve LiDAR 20 Hz, kameralar 10 Hz'dir. Kamera frame'i yapay olarak LiDAR frame
numarasına çevrilmez. `latest-at-or-before` barrier hedef frame'den ileri olmayan son
ölçümü verir. Artifact gerçek source frame ve timestamp taşır; adapter model bazındaki
max sensor skew sınırını kontrol eder.

## Backpressure

- Model executor queue kapasitesi 1.
- GPU admission reddi queue öncesi.
- Executor full ise submit reddedilir.
- Shared-memory ring full ise bounded timeout.
- Output channel `LATEST_ONLY`.
- Model input süresi dolarsa task execution öncesi düşer.
- Permanent component error route'u devre dışı bırakır.
- Süresi dolmuş output snapshot'tan otomatik çıkarılır.

## Ölçülebilirlik

`frames.jsonl` perception diagnostics:

- submitted,
- skipped_by_rate,
- skipped_by_gpu_budget,
- rejected_by_backpressure,
- completed,
- failed,
- p50/p95/max observed request latency,
- inflight,
- shared-memory allocated/busy slot,
- model health/deadline,
- output source time ve validity

alanlarını taşır.

`benchmark_guard.py`, gerçek output bulunmayan logu reddeder.

## Ground-truth politikası

Runtime semantic/instance CARLA sensörü kullanmaz. Ground truth yalnız offline
precision/recall/mAP/mIoU ve regression metriğinde referanstır.

## Sınırlar

Python host hard real-time değildir. CUDA worker ve shared-memory yolu düşük gecikmeli
olsa da otomotiv safety certification sağlamaz. Hazır gerçek-dünya modellerinin CARLA
zero-shot doğruluğu gerçek benchmark ile ölçülmelidir.

## Worker output ring

Segmentation worker'ları büyük maskeleri JSON içine koymaz. `WorkerOutputStore` her
output adı için sabit-slotlu POSIX shared-memory ring oluşturur. Worker server şu şekilde
bağlanır:

```python
store = WorkerOutputStore(WorkerOutputConfig(namespace="citysem"))
server = JsonlBackendServer(
    handler=run_inference,
    ready_metadata=real_cuda_metadata,
    release_handler=store.release,
)
```

Model handler `store.publish(...)` ile `ArtifactRef` döndürür. Host snapshot output'unu
değiştirince, expiry olduğunda veya shutdown'da protokol v2 `release` gönderir. Worker
`released` cevabından önce slotu yeniden kullanmaz.
