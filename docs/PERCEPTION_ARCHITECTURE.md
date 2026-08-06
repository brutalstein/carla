# Perception Katmanı Mimarisi

## Amaç

Bu katman, CARLA'daki altı RGB kamera ve tek 32 kanallı LiDAR girdisini dört bağımsız
hazır model ailesine dağıtır:

1. **BEVFusion Detection** — kamera + LiDAR 3B nesne algılama.
2. **BEVFusion Segmentation** — kamera + LiDAR BEV semantik raster.
3. **MapTRv2** — altı kameradan vektörel yol elemanları.
4. **TLD-READY** — ön kameralarda trafik ışığı tespiti, durum ve relevance.
5. **CitySemSegFormer** — ön kamera görüntülerinde piksel seviyesinde şehir semantiği.

BEVFusion'ın detection ve segmentation checkpoint'leri ayrı model bileşenleridir; bu
nedenle runtime açısından toplam beş model component'i vardır.

## Neden tek Python ortamı kullanılmıyor?

Model bağımlılıkları doğrudan çakışır:

- MIT BEVFusion Python 3.8, PyTorch 1.9–1.10.2, MMCV 1.4.0 ve MMDetection 2.20.0 ister.
- MapTRv2 Python 3.8, PyTorch 1.9.1, MMCV 1.4.0, MMDetection 2.14.0 ve
  MMSegmentation 0.14.1 ister.
- TLD-READY Ultralytics/YOLO ve kendi Docker akışını kullanır.
- CitySemSegFormer NVIDIA TAO/TensorRT/DeepStream ONNX dağıtımıdır.

Bu bağımlılıkları ana `l4stack` ortamına kurmak, lokalizasyon ve runtime paketinin
tekrarlanabilirliğini bozar. Her model bu nedenle ayrı conda/container sürecinde
çalışır. Ana süreç model framework'ünü import etmez.

## Veri akışı

```text
CARLA callbacks
      │ latest-at-or-before barrier
      ├── 6 × RGB camera BGRA8
      ├── raw ray-cast LiDAR float32
      └── calibration.json
      │
      ▼
PerceptionArtifactStore
      │ yalnız due model periyodunda, frame-cache'li file:// ArtifactRef
      ▼
MessageEnvelope[PerceptionInput]
      │ source time / validity / lineage
      ├──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼              ▼              ▼
BEV Det.         BEV Seg.        MapTRv2       TLD-READY      CitySem
process          process         process        process         process
      │              │              │              │              │
      ▼              ▼              ▼              ▼              ▼
3D boxes        BEV raster      polylines      lights         image masks
      └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
                                      │
                                      ▼
                           PerceptionSnapshot
                                      │
                                      ▼
                         Fusion / World Model (sonraki katman)
```

## Ağır veri taşıma

JSONL protokolüne görüntü veya nokta bulutu base64 olarak gömülmez. Ana süreç ağır
veriyi artifact dosyasına yazar ve yalnızca aşağıdaki metadata'yı yollar:

```json
{
  "name": "camera_front",
  "uri": "file:///.../camera_front.bgra8",
  "media_type": "application/x-carla-bgra8",
  "shape": [540, 960, 4],
  "dtype": "uint8",
  "byte_size": 2073600,
  "source_frame": 1250,
  "source_timestamp": 62.5
}
```

İlk transport disk/replay odaklıdır. `ArtifactRef` URI sözleşmesi gelecekte
`shm://` veya `cuda-ipc://` transportuna geçilebilmesi için model kodundan ayrıdır.

## Zaman ve senkronizasyon

Her sensör artifact'ı kendi CARLA frame ve timestamp bilgisini taşır. Model girdisinin
referans zamanı ana CARLA tick zamanıdır. `SensorSynchronizer`, 10 Hz kamera ile 20 Hz
LiDAR/dünya arasında exact-frame zorlaması yapmak yerine hedef frame'den ileri olmayan
son ölçümü busy-spin yapmadan bekler. Adapter, model config'indeki `max_sensor_skew_s`
sınırını aşan kamera veya LiDAR verisini reddeder.

Bu ayrım zorunludur; kameralar 10 Hz, LiDAR 20 Hz ve dünya 20 Hz çalıştığı için bütün
artifact'ların aynı frame numarasında olması beklenmez. Eski veriyi güncelmiş gibi
etiketlemek yerine gerçek yaşı korunur.

## Model component yaşam döngüsü

Her model ayrı `ManagedComponent` olarak çalışır:

```text
UNCONFIGURED
    │ configure: artifact preflight + backend process start + ping
    ▼
INACTIVE
    │ activate
    ▼
ACTIVE
    │ inference / deadline / health
    ├──────────────► ERROR
    │ deactivate
    ▼
INACTIVE
    │ cleanup: backend process stop
    ▼
UNCONFIGURED
```

Her model için ayrı `RuntimeSupervisor` vardır. Bir modelin checkpoint'i eksik veya
backend'i açılamıyorsa yalnızca o model finalized olur. Başarılı modeller çalışmaya
devam eder. Inference hatası component'i `ERROR` durumuna taşırsa route otomatik olarak
devre dışı kalır; her CARLA tick'inde aynı kalıcı hata yeniden üretilmez. Bu davranış
başlangıç ve inference hatalarında test edilir.

## Executor ve backpressure

Her model ayrı, tek worker'lı ve kapasitesi bir olan priority executor kullanır.

- `due_models()` her modelin hedef hızını source-time üzerinde belirler; sensör artifact'ı
  yalnızca en az bir modelin periyodu geldiğinde oluşturulur.
- Aynı kamera frame'i tekrar kullanılıyorsa artifact cache disk yazma ve SHA-256 hesabını
  tekrar etmez.
- Queue doluysa pipeline diğer modelleri submit etmeye devam eder.
- Eski frame biriktirilmez.
- Task çalışmaya başlamadan validity süresi dolarsa executor görevi düşürür.
- Output channel `LATEST_ONLY` politikasındadır.
- Pipeline her model için `submitted`, `skipped_by_rate`,
  `rejected_by_backpressure`, `completed` ve `failed` sayaçlarını tutar.

## Deadline sözleşmeleri

Başlangıç sözleşmeleri `config/runtime.yaml` içindedir. Bunlar güvenlik garantisi veya
nihai tuning değildir. Gerçek checkpoint ve GPU kurulduktan sonra CARLA replay ile
p50/p95/p99 ölçülerek güncellenecektir.

Her model için:

- `max_input_age_s`: kabul edilen en eski girdi.
- `execution_budget_s`: inference için hedef üst süre.
- `expected_output_period_s`: iki başarılı çıktı arasında izin verilen süre.
- `output_lifespan_s`: world model'in çıktıyı kullanabileceği süre.
- `request_timeout_s`: model sürecinin cevap vermesi için sert IPC timeout'u.

Config doğrulaması `request_timeout_s <= execution_budget_s` ve
`expected_output_period_s >= 1 / target_rate_hz` koşullarını uygular.

## JSONL backend protokolü

Backend stdin/stdout üzerinden satır başına tek JSON mesajı kullanır.

### Readiness

Ana süreç:

```json
{"protocol_version":1,"type":"ping"}
```

Backend:

```json
{"protocol_version":1,"type":"ready"}
```

### Inference isteği

```json
{
  "protocol_version": 1,
  "type": "infer",
  "request_id": "carla_l4/perception_input/42:maptrv2",
  "model_name": "maptrv2",
  "source_timestamp": 10.5,
  "payload": {"frame": 210, "timestamp": 10.5, "cameras": []}
}
```

### Başarılı yanıt

```json
{
  "protocol_version": 1,
  "type": "result",
  "request_id": "carla_l4/perception_input/42:maptrv2",
  "ok": true,
  "payload": {"vector_map": [], "diagnostics": {"inference_ms": 70.0}}
}
```

Request ID eşleşmezse, protokol sürümü farklıysa veya payload şeması bozuksa çıktı
reddedilir.

## Normalize edilmiş çıktılar

### BEVFusion Detection

`EGO_LOCAL` koordinat sisteminde:

- sınıf
- güven
- 3B merkez
- width/length/height
- yaw
- varsa x/y hız

### BEVFusion Segmentation

`EGO_BEV_RASTER` frame'inde tek raster artifact. Class index/label sürümü diagnostics
alanında tutulmalıdır.

### MapTRv2

`EGO_LOCAL` frame'inde category, confidence ve xyz polyline noktaları. Seçilen standart
R50 BEVPool 24ep checkpoint'i centerline checkpoint'i değildir. `MapTRv2*` daha sonra
ayrı model sürümü olarak eklenebilir.

### TLD-READY

`CAMERA_PIXEL` frame'inde kamera adı, bbox, RED/YELLOW/GREEN/UNKNOWN durumu,
pictogram, confidence ve `relevant_to_ego`. Relevance üretilmiyorsa `null` kullanılır;
uydurma `false` yazılmaz.

### CitySemSegFormer

`CAMERA_PIXEL` frame'inde kamera başına semantic mask artifact'ı. Çoklu kamera maskesi
tek raster gibi gösterilmez; `rasters[]` içinde ayrı adlarla taşınır.

## CARLA koordinat adaptasyonu

Model süreçleri şu dönüşümlerden sorumludur:

1. CARLA BGRA8 → RGB.
2. Kamera intrinsic matrisinin CARLA FOV ve çözünürlükten oluşturulması.
3. `x forward, y right, z up` CARLA ego frame'inin model frame'ine dönüştürülmesi.
4. LiDAR float32 `[x,y,z,intensity]` noktalarının model eksen/range sözleşmesine taşınması.
5. Model çıktısının tekrar `EGO_LOCAL` sözleşmesine dönüştürülmesi.
6. Resize/pad sonrası trafik ışığı bbox'larının orijinal kamera pikseline geri ölçeklenmesi.

Bu dönüşümler model backend'indedir; ana runtime model-spesifik koordinat varsayımı
yapmaz.

## Ground-truth politikası

Runtime semantic/instance CARLA sensörü kullanmaz. CARLA ground truth yalnızca ileride:

- offline benchmark,
- precision/recall/mAP/mIoU ölçümü,
- regression testi,
- zero-shot domain-gap raporu

için test aracında kullanılabilir. Runtime model girdisine aktarılmaz.

## Sınırlar

- Hazır gerçek-dünya checkpoint'leri CARLA'da zero-shot çalışacaktır; CARLA doğruluğu
  resmî nuScenes/DriveU/şehir benchmark sonuçlarıyla aynı kabul edilemez.
- Bu PR model ağırlıklarını veya üçüncü taraf repository'leri dağıtmaz.
- Model-side gerçek inference runner'ı seçilen resmî ortam içinde kurulmalıdır.
- Disk artifact transportu nihai düşük-gecikmeli transport değildir.
- Perception çıktıları henüz world model ile fuse edilmez; bu sonraki katmandır.
