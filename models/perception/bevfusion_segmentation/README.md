# BEVFusion Segmentation — RTX 5090

## Kaynak

- https://github.com/mit-han-lab/bevfusion

Dosyalar:

```text
model/bevfusion-seg.pth
model/bevfusion-seg-sm120.engine
```

Bu model opportunistic sınıftadır. BEVFusion Detection ve TLD-READY deadline baskısı
oluşturduğunda global GPU admission controller segmentation frame'ini atlayabilir.

## Engine

Engine TensorRT 11.1 / CUDA 13.3 / RTX 5090 üzerinde yeniden üretilir. Custom operator
ve preprocessing kodu orijinal model config'iyle aynı olmalıdır. Raster output `shm://`
veya `cuda-ipc://` artifact olarak döndürülmeli; büyük maskeler JSON içine gömülmemelidir.

## Worker

```bash
export L4STACK_BEVFUSION_SEGMENTATION_COMMAND='docker exec -i l4stack-perception-trt <bevfusion-seg-jsonl-worker>'
```

Worker startup'ta engine deserialize eder, buffer'ları önceden ayırır ve CPU fallback
yapmaz. Varsayılan hız 2 Hz'dir.

## Output lease

Worker, `WorkerOutputStore` ile sabit-slotlu mask ring'i kullanır. `JsonlBackendServer`
`release_handler=output_store.release` ile başlatılır. Host protokol v2 `release` onayı
almadan slot tekrar kullanılamaz.
