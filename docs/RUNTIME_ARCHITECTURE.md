# Ortak Runtime Altyapısı — Teknik Tasarım

## 1. Amaç ve kapsam

Bu paket, otonom sürüş katmanlarının algoritmik içeriğini değil, katmanların güvenli ve
tutarlı biçimde birlikte çalışmasını sağlayan ortak yürütme altyapısını tanımlar.

Hedeflenen gelecek veri zinciri:

```text
Localization
Perception
Sensor Fusion / World Model
Prediction
Behavior Planning
Motion Planning
Control
Safety / Command Gate
```

Bu katmanların tamamı farklı frekanslarda çalışabilir. Runtime'ın görevi bütün
katmanları tek bir seri döngüye zorlamak değildir. Runtime aşağıdaki sistem
özelliklerini standartlaştırır:

- lifecycle ve bağımlılık sıralaması,
- mesaj kimliği, zaman damgası ve geçerlilik süresi,
- bounded channel ve backpressure,
- immutable/atomik snapshot,
- executor ve priority sınıfları,
- freshness ve deadline gözlemi,
- component health durumu,
- data lineage,
- deterministik simülasyon/replay saati.

## 2. Literatür ve standart dayanakları

Tasarım aşağıdaki yerleşik yaklaşımlarla uyumludur:

- ROS 2 Managed Nodes: `Unconfigured → Inactive → Active → Finalized` lifecycle ve
  harici supervisor yönetimi.
  <https://design.ros2.org/articles/node_lifecycle.html>
- ROS 2 QoS: deadline, lifespan ve liveliness kavramlarıyla stale veri ve yayıncı
  sağlığının ayrı değerlendirilmesi.
  <https://design.ros2.org/articles/qos_deadline_liveliness_lifespan.html>
- ROS 2 Executor/WaitSet: callback'lerin veri veya timer hazır olduğunda çalışması,
  veri yokken busy-spin yapılmaması ve kritik zincirler için yürütme sırasının
  açıkça kontrol edilmesi.
  <https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Executors.html>
- AUTOSAR Adaptive Execution Management: application start/stop, health monitoring,
  mode management ve zaman senkronizasyonunun platform sorumluluğu olarak ayrılması.
  <https://www.autosar.org/working-groups/adaptive-platform>

Python thread scheduler hard real-time garanti vermez. Bu implementasyon gerçek zaman
özelliklerini **ölçülebilir ve sınırlandırılabilir** hâle getirir; üretim ECU'sunda
PREEMPT_RT, RTOS, process isolation, CPU affinity ve uygun middleware ayrıca gerekir.

## 3. Mesaj sözleşmesi

### 3.1 `MessageEnvelope[T]`

Her katman çıktısı ortak zarf içinde taşınır.

| Alan | Tip | Açıklama |
|---|---|---|
| `message_id` | `str` | `namespace/source/sequence` biçiminde deterministik kimlik |
| `source` | `str` | Mesajı üreten component |
| `sequence_id` | `int` | Yayıncı bazında monoton sıra |
| `source_timestamp` | `float` | Verinin temsil ettiği simülasyon zamanı |
| `publish_timestamp` | `float` | Mesajın runtime'a yayınlandığı zaman |
| `valid_until` | `float` | Mesajın kullanılabileceği son source-time |
| `coordinate_frame` | `str` | Payload koordinat sistemi |
| `parents` | `tuple[str, ...]` | Çıktının dayandığı input message id'leri |
| `payload` | `T` | Fonksiyonel katman çıktısı |

### 3.2 Zaman semantiği

```text
sensor_time = source_timestamp
processing_latency = publish_timestamp - source_timestamp
message_age(now) = now - source_timestamp
expired = now > valid_until
```

`source_timestamp` ile `publish_timestamp` birbirine karıştırılmaz. Ağır bir perception
modeli eski bir frame üzerinde çalışmışsa dünya modeli bu yaş bilgisini görebilir.

### 3.3 Immutability

`MessageEnvelope` frozen dataclass'tır. Dict/list tabanlı payload'lar
`freeze_payload=True` ile recursive olarak immutable yapılabilir. CARLA'nın dış
measurement nesneleri kopyalanmaz; sahiplik sözleşmesi gereği yayın sonrası değiştirilmez.

## 4. Saat sistemi

### `SimulationClock`

- CARLA snapshot zamanıyla ilerletilir.
- Geriye gidemez.
- Bütün source-time freshness hesaplarının tek kaynağıdır.
- Replay sırasında aynı timestamp dizisi tekrar verilebilir.

### `SteadyClock`

- İşlem süresi ve timeout ölçümünde kullanılır.
- İşletim sistemi monoton saatidir.
- Duvar saati değişikliklerinden etkilenmez.

### `ManualClock`

- Unit testlerde deadline ve stale senaryolarını beklemeden üretir.

Simülasyon zamanı ile işlem süresi aynı saatten ölçülmez. CARLA durmuş olsa bile CPU
üzerinde geçen inference süresi `SteadyClock` ile ölçülür.

## 5. Bounded channel

`BoundedChannel[T]` thread-safe ve blocking bir producer/consumer kanalıdır. Veri yokken
consumer `Condition.wait` üzerinde uyur; boş tur dönmez.

Taşma politikaları:

| Politika | Davranış | Uygun kullanım |
|---|---|---|
| `LATEST_ONLY` | Eski bekleyen mesajları siler, en yeniyi tutar | Kamera/perception girdisi |
| `DROP_OLDEST` | Queue dolunca en eskiyi düşürür | Localization/world model çıktısı |
| `DROP_NEWEST` | Yeni mesajı reddeder | Eski sıranın korunması gereken audit kanalı |
| `BLOCK` | Yer açılana kadar producer'ı uyutur | Kayıpsız ama kritik olmayan offline iş |

Normal sürüş zincirinde sınırsız FIFO kullanılmaz. Perception 300 ms sürerken 10 Hz kamera
framelerinin tamamını biriktirmek gecikmeyi sürekli büyütür.

### Channel output

`publish` kabul durumunu `bool` olarak verir. `stats()` aşağıdakileri sağlar:

- published,
- received,
- dropped,
- rejected,
- current depth,
- capacity,
- closed.

## 6. Atomik snapshot

`AtomicSnapshotStore[T]`, tamamlanmış son çıktının lock altında tek işlemle aktif hâle
gelmesini sağlar.

```text
Aktif: WorldModel v104
Hazırlanıyor: WorldModel v105
v105 tamamlandı
publish(v105)
Aktif: WorldModel v105
```

Okuyucu v104 veya v105 görür; yarım v105 göremez. Her yayın monoton `version` üretir.
`wait_for_newer(version)` busy-spin yapmadan yeni sürümü bekler.

Planlama bir iterasyona başladığında aldığı snapshot'ı iterasyon sonuna kadar değişmeden
kullanmalıdır.

## 7. Runtime contract

`ComponentContract` her fonksiyonel katman için aşağıdaki sınırları tek noktada tanımlar:

| Alan | Açıklama |
|---|---|
| `criticality` | Safety-critical, firm real-time veya soft real-time |
| `priority` | Executor priority sınıfı |
| `max_input_age_s` | Kabul edilen maksimum input yaşı |
| `execution_budget_s` | Fonksiyonun işlem bütçesi |
| `expected_output_period_s` | İki çıktı arasındaki beklenen üst süre |
| `output_lifespan_s` | Çıktının kullanım süresi |
| `channel_capacity` | Output queue sınırı |
| `overflow_policy` | Queue dolma davranışı |
| `drop_expired_inputs` | Stale input'un işlenip işlenmeyeceği |

Kontratlar `config/runtime.yaml` içinden yüklenir. Kod içine dağılmış gizli timeout ve
queue sabitleri kullanılmaz.

## 8. Deadline ve freshness denetimi

`DeadlineMonitor` dört ihlal türü üretir:

```text
INPUT_STALE
INPUT_EXPIRED
EXECUTION_BUDGET
OUTPUT_PERIOD
```

### Input

```text
age > max_input_age_s  → INPUT_STALE
now > valid_until      → INPUT_EXPIRED
```

### Execution

```text
processing_end - processing_start > execution_budget_s
→ EXECUTION_BUDGET
```

### Output period

```text
current_publish_time - previous_publish_time > expected_output_period_s
→ OUTPUT_PERIOD
```

Her component için toplam execution, toplam ihlal, ardışık ihlal, son ve maksimum işlem
süresi tutulur. Deadline monitor otomatik fren kararı vermez; safety supervisor'ın
kullanacağı objektif kanıtı üretir.

## 9. Lifecycle

`ManagedComponent` durumları:

```text
UNCONFIGURED
     │ configure
     ▼
INACTIVE
     │ activate
     ▼
ACTIVE
     │ deactivate
     ▼
INACTIVE
     │ cleanup
     ▼
UNCONFIGURED
     │ shutdown
     ▼
FINALIZED
```

Her hook hatasında component `ERROR` durumuna geçer. Geçersiz state transition doğrudan
`LifecycleError` üretir.

Fonksiyonel component yalnızca şu hook'ları override eder:

- `on_configure`: config/model/queue kaynaklarını hazırla,
- `on_activate`: input kabul etmeye başla,
- `on_deactivate`: yeni iş kabulünü durdur,
- `on_cleanup`: yeniden configure edilebilir duruma dön,
- `on_shutdown`: thread/channel/dosya kaynaklarını kapat,
- `on_error`: bileşene özel hata işlemi.

## 10. Supervisor

`RuntimeSupervisor` component bağımlılıklarını topolojik sıralar.

Örnek:

```text
sensors → localization → world_model → planning → control
```

Başlatma:

```text
configure: sensors, localization, world_model, planning, control
activate:  sensors, localization, world_model, planning, control
```

Kapatma ters sıradadır. Başlatmanın ortasında hata oluşursa aktive/configure edilmiş
component'ler ters sırada rollback edilir.

## 11. Executor

### `PriorityExecutor`

- Bounded priority queue kullanır.
- Worker'lar `queue.get()` üzerinde uyur.
- Küçük priority sayısı önce çalışır.
- Aynı priority için submission sequence deterministik FIFO sağlar.
- Task daha çalışmadan deadline'ı dolmuşsa opsiyonel olarak düşürülebilir.
- `Future` üzerinden sonuç veya exception çağırana aktarılır.

Priority sınıfları:

```text
0   SAFETY
10  CONTROL
20  LOCALIZATION
30  WORLD_MODEL
40  PLANNING
50  PERCEPTION
100 BACKGROUND
```

Bu sınıflar OS thread priority değildir; uygulama içi ready-queue sırasıdır.

### `PeriodicScheduler`

Fixed-rate görevleri executor'a bırakır. Bir sonraki release, önceki release zamanına
period eklenerek hesaplanır; callback çalışma süresi kadar drift biriktirilmez.

### `ExecutorRegistry`

`runtime.yaml` içindeki profillerden executor'ları lazy oluşturur ve topluca kapatır.

## 12. Health registry

`HealthReport`:

| Alan | Açıklama |
|---|---|
| component | Rapor sahibi |
| state | NOMINAL/DEGRADED/STALE/FAILED/UNAVAILABLE |
| timestamp | Rapor zamanı |
| reason | İnsan okunabilir neden |
| metrics | Katmana özel ölçümler |

Registry yalnızca son tam raporu gösterir. `mark_stale` ile belirli süre rapor üretmeyen
component `STALE` yapılabilir. Aggregate health en ağır durumu döndürür.

## 13. Data lineage

Her output mesajı parent input kimliklerini taşır:

```text
ControlCommand 901
└── Trajectory 52
    └── WorldModel 240
        ├── Localization 2500
        ├── Perception 120
        └── RadarFrame 800
```

`LineageStore` bounded kayıt tutar ve `trace(output_id)` ile çıktıdan sensöre doğru soy
ağacı üretir. Bu yapı regression analizi, incident replay ve safety evidence için
kullanılır.

## 14. Lokalizasyon entegrasyonu

### Input

```python
MessageEnvelope[SensorFrame]
```

`SensorFrame` alanları:

| Alan | Açıklama |
|---|---|
| frame | CARLA frame numarası |
| timestamp | CARLA elapsed simulation time |
| measurements | GNSS ve IMU measurement tablosu |

Runner exact-frame synchronizer'dan aldığı bundle'ı `SensorFrame` yapar. Input message:

```text
source = sensor_synchronizer
coordinate_frame = CARLA_SENSOR_BUNDLE
valid_until = source_timestamp + sensor_bundle_lifespan_s
```

### İşlem

`LocalizationRuntimeComponent`:

1. `ACTIVE` state kontrolü yapar.
2. Input age ve expiry kontrolü yapar.
3. ESKF işlem süresini steady clock ile ölçer.
4. `LocalizationEstimate` üretir.
5. Parent olarak sensor message id ekler.
6. Output'u lineage store'a kaydeder.
7. Atomik snapshot ve bounded channel'a yayınlar.
8. Deadline ve health raporunu günceller.

### Output

```python
MessageEnvelope[LocalizationEstimate]
```

```text
source = localization
coordinate_frame = LOCAL_ENU
parents = (sensor_message_id,)
```

Payload:

- local ENU pose,
- local ENU velocity,
- body angular rate,
- speed,
- position/yaw uncertainty,
- ESKF health,
- GNSS/compass acceptance diagnostics.

## 15. Gelecekteki katmanların entegrasyon kuralı

Yeni katman doğrudan başka katmanın Python nesnesini düzenlemeyecek. Her katman:

1. `ManagedComponent` türetir.
2. `ComponentContract` alır.
3. Input olarak `MessageEnvelope[TInput]` kabul eder.
4. Input freshness kontrolü yapar.
5. Tek snapshot sürümünü hesap boyunca kullanır.
6. Output olarak `MessageEnvelope[TOutput]` üretir.
7. Parent id'leri lineage'a ekler.
8. Bounded output channel ve atomic snapshot yayınlar.
9. Deadline ve health raporu üretir.

Bu kurallar perception, fusion, prediction, behavior planning, motion planning ve
control için değişmez.
