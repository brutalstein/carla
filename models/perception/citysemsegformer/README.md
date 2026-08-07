# NVIDIA CitySemSegFormer — TensorRT 11.1

## Kaynak

- NVIDIA NGC CitySemSegFormer model kartı:
  https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/citysemsegformer

Dosyalar:

```text
model/citysemsegformer.onnx
model/citysemsegformer-sm120.engine
model/labels.txt
```

## Engine

Engine RTX 5090 üzerinde TensorRT 11.1/CUDA 13.3 ile FP16 üretilir. ONNX input binding
adı ve shape'i `trtexec` tarafından okunmalı; tahmin edilmemelidir. Engine build sonrası
`trtexec --loadEngine ... --useCudaGraph` benchmark'ı ve gerçek ön kamera smoke testi
yapılır.

## Worker

```bash
export L4STACK_CITYSEMSEGFORMER_COMMAND='docker exec -i l4stack-perception-trt <citysem-jsonl-worker>'
```

Üç kamera aynı process içinde batch veya ardışık CUDA stream ile işlenebilir. Maskeler
JSON içine gömülmez; worker output shared-memory raster referansı döndürür. Model
opportunistic sınıftadır ve varsayılan hız 2 Hz'dir.

## Output lease

Kamera maskeleri `WorkerOutputStore` üzerinden yayınlanır. Worker server
`release_handler=output_store.release` kullanır; host snapshot değişimi/expiry sırasında
protokol v2 release mesajı gönderir.
