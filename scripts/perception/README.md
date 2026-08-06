# Perception Yardımcı Scriptleri

- `mock_backend.py`: gerçek model kurulmadan JSONL readiness/inference protokolünü test
  eden deterministic backend.

Örnek:

```bash
export PYTHONPATH="$PWD/src"
python scripts/perception/mock_backend.py --kind maptrv2
```

Normal model süreçleri `models/perception/<model>/run_backend.sh` üzerinden başlatılır.
Gerçek model runner'ı ilgili izole environment içinde `JsonlBackendServer` protokolünü
uygulamalıdır.
