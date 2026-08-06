# Validation Report

Date: 2026-08-06

## Değişiklik kapsamı

- Ortak runtime paketi eklendi.
- Message envelope, source/publish timestamp, lifespan ve lineage sözleşmeleri eklendi.
- Bounded channel ve atomik snapshot mekanizmaları eklendi.
- Managed lifecycle ve dependency-aware supervisor eklendi.
- Priority executor, periodic scheduler ve executor registry eklendi.
- Deadline/freshness monitor ve merkezi health registry eklendi.
- `config/runtime.yaml` ile executor/component contract yönetimi eklendi.
- Planar GNSS/IMU ESKF, `LocalizationRuntimeComponent` içine taşındı.
- Runner lokalizasyonu priority executor üzerinden çalıştıracak şekilde güncellendi.
- Runtime metadata, health ve deadline sonuçları JSONL kaydına eklendi.
- Eski actor cleanup yardımcı dosyası lifecycle isminden ayrılarak `core/actors.py` yapıldı.

## Doğrulamalar

- `pytest -q`: **23 passed**
- `python -m compileall -q src tests`: passed
- `l4stack --config-dir config validate`: 14 sensors, localization runtime contract passed
- `l4stack --config-dir config coverage`: 360° camera azimuth coverage passed
- Message immutability ve deterministic sequence: passed
- Bounded channel overflow/blocking: passed
- Atomic snapshot version/wait: passed
- Lifecycle dependency order ve reverse shutdown: passed
- Priority executor queued ordering: passed
- Input stale, execution budget ve output period ihlalleri: passed
- Health staleness ve lineage trace: passed
- Localization runtime output parent/snapshot/channel/health: passed
- Expired localization input rejection: passed
- WGS-84 local tangent plane round-trip: passed
- Sentetik 5 m/s sabit hızlı ESKF benchmark RMSE: yaklaşık **0.20 m**
- 100 m GNSS outlier NIS gate tarafından reddedildi

## Ortam sınırlamaları

- CARLA sunucusu artifact ortamında bulunmadığından gerçek actor spawn ve callback
  entegrasyon koşusu burada yapılmadı.
- `ruff` executable bu ortamda kurulu olmadığından lokal lint koşusu çalıştırılamadı;
  GitHub Actions workflow'u `ruff check .` çalıştırır.
- Python thread scheduler hard real-time garanti vermez. Runtime deadline gözlemi ve
  priority sıralaması sağlar; üretim ECU zamanlama garantisi değildir.
