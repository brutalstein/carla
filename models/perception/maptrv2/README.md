# MapTRv2 — RTX 5090 Modern Port

## Kaynak

- Resmî repo: https://github.com/hustvl/MapTR
- Branch: `maptrv2`
- Checkpoint: MapTRv2 R50 BEVPool 24 epoch nuScenes

Dosya:

```text
model/maptrv2_r50_bevpool_24ep.pth
```

## Kritik uyumluluk notu

Resmî kurulum Python 3.8, PyTorch 1.9 ve MMCV 1.4 dönemine aittir. Bu binary stack RTX
5090 SM 12.0 için uygun değildir. Host veya runtime'da eski CUDA/PyTorch kurulmaz.

RTX 5090 port ortamı:

```text
Python 3.12
PyTorch 2.12.1 + CUDA 13.0
TORCH_CUDA_ARCH_LIST=12.0
MMCV CUDA ops source build
Modern MMEngine/MMDetection/MMDetection3D API portu
```

Checkout:

```bash
git clone --branch maptrv2 https://github.com/hustvl/MapTR.git \
  models/perception/maptrv2/external/MapTR
```

PyTorch worker image'ı:

```bash
docker compose -f infra/perception/docker-compose.cuda.yml build maptr-worker
docker compose -f infra/perception/docker-compose.cuda.yml up -d maptr-worker
```

MMCV'nin PyTorch 2.12/CUDA 13 hazır wheel'i yoksa kaynak koddan derlenir. Derleme başarılı
olsa bile MapTRv2'nin registry/config/runner API farkları port edilmeden checkpoint
çalışmayabilir. Port ancak şu gerçek kontrollerden sonra tamamlanmış sayılır:

1. `torch.cuda.get_device_capability() == (12, 0)`.
2. Checkpoint eksiksiz yüklenir; beklenmeyen missing/unexpected key raporu incelenir.
3. Altı gerçek CARLA kamera shared-memory girdisi okunur.
4. En az bir gerçek vector-map output üretilir.
5. 100 warm-up + 500 ölçüm inference benchmark'ı alınır.
6. Sonuç ve dependency SHA'ları `model/rtx5090_port_verified.json` dosyasına yazılır.

Bu doğrulama dosyası zorunlu artifact'tır; olmadan model READY değildir.

## Worker

```bash
export L4STACK_MAPTRV2_COMMAND='docker exec -i l4stack-perception-maptr <maptrv2-jsonl-worker>'
```

Worker `torch.inference_mode`, autocast FP16/BF16, persistent CUDA buffer ve
`cudaMallocAsync` allocator kullanmalıdır. CPU inference yolu yasaktır.
