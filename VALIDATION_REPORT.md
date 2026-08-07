# Validation Report

## Kapsam

Bu rapor ortak runtime/lokalizasyon tabanına eklenen süreç-izole perception mimarisinin
yerel doğrulamasını özetler. Gerçek checkpoint inference benchmark'ı model dosyaları ve
GPU ortamları kurulmadan çalıştırılmamıştır.

## Perception testleri

Çalıştırılan komut:

```bash
PYTHONPATH=src python -m pytest -q tests/test_perception_core.py
```

Sonuç:

```text
26 passed
```

Doğrulanan davranışlar:

- ArtifactRef URI, boyut, shape ve source-time doğrulaması.
- Duplicate kamera, eksik LiDAR ve sensor skew reddi.
- BEVFusion 3B detection normalize şeması.
- BEVFusion BEV raster normalize şeması.
- MapTRv2 vector-map polyline şeması.
- TLD-READY state/relevance şeması.
- CitySemSegFormer çoklu kamera raster şeması.
- Model component health, lineage ve atomik snapshot yayını.
- Stale input ve inference failure yolu.
- Priority executor üzerinde async rate gate ve yalnız-due-model artifact üretimi.
- Manifest file size, SHA-256 ve backend komut kontrolü.
- Gerçek subprocess stdin/stdout JSONL readiness ve inference round-trip.
- Factory'nin yalnız etkin modelleri kurması ve lifecycle başlatması.
- CARLA BGRA8/LiDAR float32 artifact yazımı, frame cache ve input parent lineage.
- 10 Hz kamera / 20 Hz LiDAR için latest-at-or-before sensör bariyeri.
- Kalıcı component hatasında route'un otomatik devre dışı bırakılması.
- Repository içindeki beş `run_backend.sh` wrapper'ının ping ve inference handshake'i.
- Hatalı readiness handshake sonrası child-process cleanup davranışı.
- JSONL response `ok` alanının strict boolean doğrulaması.

## Tam test paketi

```bash
PYTHONPATH=src python -m pytest -q
```

Sonuç:

```text
30 passed
```

## Derleme

```bash
PYTHONPATH=src python -m compileall -q src tests scripts
```

Sonuç: passed.

## Yapılamayan doğrulamalar

- BEVFusion, MapTRv2, TLD-READY ve CitySemSegFormer gerçek ağırlıkları bu çalışma
  ortamında bulunmadığından gerçek GPU inference çalıştırılmadı.
- CARLA sunucusu bulunmadığından actor/sensor callback entegrasyon koşusu yapılmadı.
- İnternet/paket index erişimi olmadığı için yerel Ruff kurulamadı; repository CI
  workflow'u Ruff ve tam pytest paketini çalıştıracaktır.
- Resmî benchmark değerleri CARLA sonucu değildir. CARLA zero-shot metriği sonraki
  validation aşamasında ground-truth yalnızca test referansı olarak ölçülmelidir.

## Kabul kriterleri

Model etkinleştirilmeden önce:

1. `l4stack perception-doctor` ilgili model için READY vermeli.
2. Resmî dataset/example smoke testi modelin kendi ortamında geçmeli.
3. JSONL readiness/inference protokol testi geçmeli.
4. CARLA replay testinde coordinate/calibration adapter doğrulanmalı.
5. p50/p95/p99 inference süreleri runtime contract değerleriyle karşılaştırılmalı.
6. Output class/frame/shape sözleşmeleri regression testine alınmalı.
