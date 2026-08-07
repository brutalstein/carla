# TLD-READY — RTX 5090 TensorRT

## Kaynak

- https://github.com/KASTEL-MobilityLab/traffic-light-detection

Ağırlık:

```text
model/traffic_lights_yolov8x.pt
```

Runtime engine:

```text
model/traffic_lights_yolov8x-sm120.engine
```

TLD-READY critical sınıftadır ve ön üç kamerada 10 Hz çalışır. Küçük inference işi MPS
altında ağır BEV/MapTR kernel'leriyle kontrollü overlap edebilir.

## Export

TLD worker image'ı PyTorch 2.12.1/CUDA 13.0 ve Ultralytics 8.4.104 içerir:

```bash
docker compose -f infra/perception/docker-compose.cuda.yml build tld-worker
docker compose -f infra/perception/docker-compose.cuda.yml up -d tld-worker
```

Engine hedef RTX 5090'da FP16 olarak export edilir. Export sonrası gerçek CARLA trafik
ışığı görüntüleriyle state/pictogram sınıf eşlemesi doğrulanmalıdır.

## Worker

```bash
export L4STACK_TLD_READY_COMMAND='docker exec -i l4stack-perception-tld <tld-jsonl-worker>'
```

Output bbox'ları `CAMERA_PIXEL`, state `RED/YELLOW/GREEN/UNKNOWN`, confidence ve varsa
ego relevance alanlarını taşımalıdır. Runtime CPU fallback kabul etmez.

Lisans: TLD-READY/Ultralytics AGPL/ticari lisans koşulları ürünleştirmeden önce hukuk
incelemesi gerektirir.
