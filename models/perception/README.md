# Perception Model Yerleşimi — RTX 5090 CUDA-Only

Bu dizin model ağırlıklarını, target-SM120 engine dosyalarını ve üçüncü taraf kaynak
kodlarını birbirinden ayırır. Büyük dosyalar Git'e eklenmez.

## Kurulum sırası

1. `docs/RTX5090_CUDA_SETUP.md` ile host, driver, Docker, MPS ve image'ları kurun.
2. Her modelin resmî ağırlığını kendi `model/` dizinine koyun.
3. TensorRT modellerinde engine'i hedef RTX 5090 üzerinde üretin.
4. Worker command'ini model README'sindeki environment değişkenine yazın.
5. `l4stack --config-dir config cuda-doctor` çalıştırın.
6. `l4stack --config-dir config perception-doctor` çalıştırın.
7. Yalnız bir modeli etkinleştirip 400 gerçek CARLA frame'i benchmark edin.
8. Başarılı modelden sonra sıradaki modeli açın.

## Ortak dizin sözleşmesi

```text
models/perception/<model>/
├── external/       üçüncü taraf repo checkout'u; Git'e girmez
├── model/          checkpoint, ONNX ve engine; Git'e girmez
├── README.md
└── run_backend.sh  CUDA guard üzerinden persistent worker başlatır
```

`run_backend.sh` içinde sentetik/mock yol yoktur. İlgili `L4STACK_*_COMMAND` tanımlı
değilse veya CUDA görünür değilse process fail-fast kapanır.

## Online veri

Model worker'ı frame dosyası okumaz. İstek içindeki `shm://` URI'larını
`l4stack.perception.shared_memory.open_shared_artifact()` ile açar. Worker, memoryview
üzerinden NumPy/Torch view oluşturmalı ve pinned buffer/CUDA stream'e async H2D copy
yapmalıdır. Request tamamlanmadan referans saklanmamalıdır; future dönüşünde host slotu
serbest bırakabilir.

## Readiness zorunluluğu

Worker modeli ve engine'i startup'ta yükler. `ping` yanıtında gerçek CUDA durumu:

```json
{
  "device": "cuda",
  "cuda_available": true,
  "cpu_fallback": false,
  "model_loaded": true,
  "compute_capability": 12.0,
  "precision": "fp16"
}
```

olmalıdır. CPU, boş model veya eski GPU yanıtı kabul edilmez.

## Worker output shared memory

Raster üreten worker'lar `l4stack.perception.WorkerOutputStore` kullanır ve
`JsonlBackendServer(..., release_handler=store.release)` ile protokol v2 release
sözleşmesini uygular. Worker-owned slot host onayı gelmeden overwrite edilmez.
