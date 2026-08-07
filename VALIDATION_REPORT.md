# Validation Report — RTX 5090 CUDA Perception Hardening

## Kapsam

Bu tur online perception veri yolundaki disk I/O, sentetik backend, kontrolsüz çoklu GPU
submit ve CUDA readiness eksiklerini kapatır.

## Yerel doğrulamalar

Çalışma ortamında RTX 5090, CARLA server ve gerçek checkpoint bulunmadığı için gerçek
GPU inference sonucu üretilmemiştir. Aşağıdaki doğrulamalar gerçek işletim sistemi
shared-memory kaynağı ve gerçek child-process protokolüyle yapılmıştır:

- POSIX shared-memory byte round-trip.
- Kamera/LiDAR ring slot boyutu ve shape kontrolü.
- Producer reservation, consumer lease ve release.
- Abort sonrası slotun tekrar kullanılabilmesi.
- Stale generation/token reddi.
- GPU admission class/priority/budget kararı.
- Global ve model in-flight sınırı.
- Out-of-order reservation completion.
- Rolling p50/p95 ölçüm geri beslemesi.
- CUDA readiness metadata kabul/red yolları.
- CPU fallback readiness reddi.
- Persistent JSONL protocol v2 start/infer/release/shutdown.
- Calibration JSON'un host-path bağımsız immutable shared-memory artifact'ına taşınması.
- WorkerOutputStore raster slot lease ve host release acknowledgement davranışı.
- Normal shutdown'da future drain → output release → backend stop → input ring unlink sırası.
- Wrapper dosyalarında mock yolunun bulunmaması.
- Config deadline/rate/lifespan tutarlılığı.

Özel yeni modül testi:

```text
4 passed
```

Tam repository testi GitHub Actions üzerinde Python 3.10/3.11/3.12 matrisinde
çalıştırılacaktır. PR CI sonucu görünmeden bu rapor tam test paketinin geçtiğini iddia
etmez.

## Üretim kabul kapıları

RTX 5090 host'ta sırasıyla:

```bash
l4stack --config-dir config cuda-doctor
l4stack --config-dir config perception-doctor
l4stack --config-dir config run --frames 400
python scripts/perception/benchmark_guard.py output/frames.jsonl ...
```

çalıştırılmalıdır.

Bir model nominal kabul edilmeden:

1. Gerçek checkpoint/engine yüklenmeli.
2. Readiness `device=cuda`, `model_loaded=true`, `cpu_fallback=false` vermeli.
3. Compute capability 12.0 doğrulanmalı.
4. En az 400 gerçek CARLA frame'i işlenmeli.
5. Failure rate ≤ %1 olmalı.
6. Critical model deadline p95 altında kalmalı.
7. VRAM OOM, Xid veya MPS fault oluşmamalı.
8. CARLA ground truth yalnız offline doğruluk metriğinde kullanılmalı.

## Açık model-port kapısı

MapTRv2 resmî environment'ı RTX 5090 ile doğrudan uyumlu değildir. PyTorch 2.12/CUDA
13 portu gerçek GPU'da doğrulanmadan ve `rtx5090_port_verified.json` üretilmeden model
READY sayılmaz. Bu durum gizlenmez veya sentetik başarı dosyasıyla geçilmez.
