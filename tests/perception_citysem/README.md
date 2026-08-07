# CitySemSegFormer RTX 5090 gerçek TensorRT gözlem testi

Bu klasör yalnız **gerçek model/engine/görüntü** ile çalışan manuel GPU acceptance testidir.
Mock perception çıktısı üretmez. Ana `pytest` suite'ine otomatik dahil değildir; RTX 5090,
TensorRT ve model dosyaları olmayan CI makinelerini etkilemez.

## Tasarım

Test yolu production hedefiyle aynı temel performans ilkelerini kullanır:

- engine yalnız bir kez deserialize edilir,
- tek persistent TensorRT execution context kullanılır,
- input/output GPU buffer'ları bir kez ayrılır,
- host staging buffer'ları bir kez ayrılır ve `cudaHostRegister` ile pinned yapılır,
- tek CUDA stream yeniden kullanılır,
- frame başına model yükleme, `cudaMalloc`, process oluşturma veya disk ara-frame yazımı yoktur,
- H2D -> `execute_async_v3` -> D2H aynı stream üzerinde sıralanır.

Bu test henüz ana CARLA shared-memory worker protokolü değildir. Önce gerçek engine'in
preprocess + inference + semantic-mask sözleşmesini ve RTX 5090 latency'sini izole olarak
kanıtlar. Başarılı sonuçtan sonra aynı runner mantığı persistent perception worker'a taşınır.

## Gereken yerel dosyalar

Git tarafından bilinçli olarak izlenmezler:

```text
models/perception/citysemsegformer/model/citysemsegformer.onnx
models/perception/citysemsegformer/model/citysemsegformer-sm120-fp32.engine
models/perception/citysemsegformer/model/labels.txt
```

ONNX beklenen gerçek binding:

```text
input   [-1, 3, 1024, 1820]  FP32
output  [-1, 1024, 1820, 1]  INT64 class id
```

## Conda ortamı

Yeni makinede:

```bash
conda env create -f tests/perception_citysem/environment.yml
```

Mevcut `perception_trt` ortamında bağımlılıklar zaten kuruluysa yeniden oluşturmayın.

## Kurulumu doğrula

```bash
conda activate perception_trt
python tests/perception_citysem/verify_install.py
```

## Engine yeniden üretimi

Sadece hedef RTX 5090 üzerinde gerektiğinde:

```bash
conda activate perception_trt
python tests/perception_citysem/build_engine.py
```

Bu script mevcut doğrulanmış yol için FP32 referans engine üretir. FP16/ModelOpt parity ayrı
bir acceptance adımıdır; başarısı ölçülmeden FP16 gibi raporlanmaz.

## Tek gerçek görüntü ile OpenCV

```bash
bash tests/perception_citysem/run.sh --image /absolute/path/to/real_frame.png
```

Pencerede yan yana:

```text
RGB | semantic class-id mask | alpha overlay
```

Ayrıca preprocess, H2D+TensorRT+D2H GPU-pipeline, postprocess ve toplam latency görünür.

Bir ekran görüntüsü kaydetmek için:

```bash
bash tests/perception_citysem/run.sh \
  --image /absolute/path/to/real_frame.png \
  --save output/citysem_test.png
```

## Gerçek video / canlı kamera

```bash
bash tests/perception_citysem/run.sh --video /absolute/path/to/real_video.mp4
```

veya:

```bash
bash tests/perception_citysem/run.sh --camera-index 0
```

`q` veya `ESC` ile çıkılır. İlk 5 frame varsayılan olarak warm-up'tır.

Ekranı açmadan 200 gerçek frame latency benchmark:

```bash
bash tests/perception_citysem/run.sh \
  --video /absolute/path/to/real_video.mp4 \
  --warmup 20 \
  --max-frames 200 \
  --no-display
```

Çıkışta mean/p50/p95/p99 yazılır. `gpu_pipeline_ms`, H2D + TensorRT enqueue/execute + D2H
ve stream synchronization süresidir; yalnız kernel latency diye yorumlanmamalıdır.

## Preprocess sözleşmesi

OpenCV BGR frame önce ONNX'in gerçek `1024x1820` çözünürlüğüne resize edilir, RGB'ye
çevrilir ve NVIDIA TAO SegFormer normalizasyonu uygulanır:

```text
mean RGB = [123.675, 116.28, 103.53]
std  RGB = [58.395, 57.12, 57.375]
```

Çıktı 19 Cityscapes sınıfının class-id maskesidir. `labels.txt` sırası runtime'da okunur.

## Kabul kriterleri

Bu test için minimum kabul:

1. `verify_install.py` RTX 5090 / SM 12.0 ve engine'i doğrular.
2. Gerçek görüntüde output shape `1024x1820` class-id maskedir.
3. Maskedeki class id'ler labels aralığındadır; belirgin bozuk/tek-sınıf çıktı gözlemlenmez.
4. 20 warm-up + en az 200 gerçek frame benchmark tamamlanır.
5. p50/p95/p99 ve VRAM kullanımı kaydedilir.
6. Sonuç kabul edilmeden ana perception runtime'a bağlanmaz.

NVIDIA TensorRT 11.x runtime sözleşmesi `set_tensor_address` + `execute_async_v3` kullanır.
Buffer'lar inference tamamlanana kadar canlı tutulur; bu harness buffer'ları runner ömrü boyunca
sabit tutar.
