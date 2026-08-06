# Yeni Runtime Component Ekleme Rehberi

Bu belge gelecekte eklenecek perception, world model, prediction, planning ve control
katmanlarının ortak runtime'a nasıl bağlanacağını tanımlar.

## 1. Payload veri tipini tanımla

Payload mümkünse frozen dataclass olmalıdır:

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PredictionFrame:
    reference_time: float
    world_model_version: int
    trajectories: tuple[PredictedTrajectory, ...]
```

Payload içinde paylaşılan mutable `list` veya `dict` kullanılmamalıdır.

## 2. Runtime contract ekle

`config/runtime.yaml`:

```yaml
runtime:
  components:
    prediction:
      criticality: "FIRM_REALTIME"
      priority: 40
      max_input_age_s: 0.15
      execution_budget_s: 0.050
      expected_output_period_s: 0.10
      output_lifespan_s: 0.20
      channel_capacity: 1
      overflow_policy: "LATEST_ONLY"
      drop_expired_inputs: true
```

Değerler ölçüm ve benchmark ile doğrulanmadan safety requirement kabul edilmemelidir.

## 3. Managed component oluştur

```python
class PredictionComponent(ManagedComponent):
    def __init__(self, runtime, contract):
        super().__init__("prediction", dependencies=("world_model",))
        self.runtime = runtime
        self.contract = contract
        self.output_channel = BoundedChannel(
            "prediction.output",
            contract.channel_capacity,
            contract.overflow_policy,
        )
        self.output_snapshot = AtomicSnapshotStore("prediction.latest")
```

## 4. Lifecycle kaynaklarını hook'larda yönet

```python
def on_configure(self):
    self.runtime.deadlines.register(self.contract)
    self.model = load_model()

def on_activate(self):
    warmup(self.model)

def on_deactivate(self):
    stop_accepting_new_work()

def on_shutdown(self):
    self.output_channel.close()
```

Model yükleme constructor içinde yapılmamalıdır; configure başarısızlığı supervisor
tarafından görülebilmelidir.

## 5. Input'u doğrula

```python
def process(self, message):
    self.require_active()
    violations = self.runtime.deadlines.validate_input(self.name, message)
    if violations and self.contract.drop_expired_inputs:
        raise StaleInput(...)
```

## 6. Tek snapshot kullan

Bir işlem başladığında gereken girdiler alınır ve işlem boyunca değiştirilmez:

```python
world_model = world_model_store.require()
route = route_store.require()
```

İşlem ortasında yeni world model gelirse mevcut iterasyon ona geçmez.

## 7. Output ve lineage üret

```python
output = factory.create(
    prediction,
    source_timestamp=world_model.value.source_timestamp,
    lifespan_s=contract.output_lifespan_s,
    parents=(world_model_message_id,),
)
runtime.lineage.record(output)
output_snapshot.publish(output, output.publish_timestamp)
output_channel.publish(output)
```

## 8. Deadline ve health raporu üret

```python
started = deadlines.start_execution()
# algoritma
violations = deadlines.finish_execution(...)
health.report(HealthReport(...))
```

## 9. Supervisor'a kaydet

```python
supervisor.register(world_model_component)
supervisor.register(prediction_component)
```

Bağımlılık döngüsü varsa supervisor başlatmayı reddeder.

## 10. Zorunlu testler

Her yeni component için en az:

- lifecycle configure/activate/deactivate/shutdown,
- stale input reddi,
- output message parent id,
- atomic snapshot version artışı,
- bounded channel taşma politikası,
- deadline budget ihlali,
- health state geçişi,
- deterministic aynı-input/aynı-output testi,
- supervisor dependency sırası,
- algoritmik benchmark.
