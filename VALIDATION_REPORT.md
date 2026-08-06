# Validation Report

Date: 2026-08-06

## Değişiklik kapsamı

- Runtime algılama modülleri ve bunlara ait veri tipleri kaldırıldı.
- Semantic LiDAR, semantic/instance sensör referansları kaldırıldı.
- CARLA actor pose/velocity kullanan lokalizasyon adaptörü kaldırıldı.
- GNSS/IMU planar error-state EKF eklendi.
- Ground truth yalnızca test benchmark referansı olarak sınırlandı.

## Doğrulamalar

- `pytest -q`: **6 passed**
- `python -m compileall -q src tests`: passed
- `l4stack --config-dir config validate`: 14 sensors, GNSS+IMU required
- `l4stack --config-dir config coverage`: 360° camera azimuth coverage passed
- WGS-84 local tangent plane round-trip: passed
- Sentetik 5 m/s sabit hızlı benchmark RMSE: **0.20 m**
- 100 m GNSS outlier NIS gate tarafından reddedildi
- `git diff --check`: passed

`ruff` artifact ortamında kurulu olmadığı için burada çalıştırılamadı. CARLA sunucusu da
bulunmadığından gerçek actor spawn ve sensor callback entegrasyon koşusu yapılmadı.
