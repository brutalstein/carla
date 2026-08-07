# RTX 5090 CUDA-Only Perception Kurulumu

Bu belge, `brutalstein/carla` perception katmanını tek RTX 5090 32 GiB üzerinde
çalıştırmak için referans host ve container kurulumunu tanımlar. Runtime'da CPU
inference fallback'i yoktur. CPU yalnız CARLA callback yönetimi, mesajlaşma ve sensör
byte'larını shared-memory ring'e kopyalamak için kullanılır.

## 1. Sabitlenen platform

```text
OS                          Ubuntu 24.04 LTS x86-64
GPU                         NVIDIA GeForce RTX 5090, compute capability 12.0
Minimum driver              610.43.02
TensorRT runtime            11.1.0, CUDA 13.3
Modern PyTorch worker       2.12.1 + torchvision 0.27.1, CUDA 13.0
Python                      3.12
NVIDIA Container Toolkit   1.19.1
Online frame transport      POSIX shared memory, /dev/shm
Multi-process GPU runtime   CUDA MPS
```

Sürümler `infra/perception/versions.env` dosyasında tek kaynaktan tutulur.
TensorRT engine dosyaları GPU mimarisi ve TensorRT sürümüne bağlıdır; başka GPU veya
TensorRT sürümünde üretilmiş `.engine` dosyası kullanılmaz.

## 2. Host ön koşulları

BIOS/host tarafında:

- Resizable BAR açık olmalıdır.
- RTX 5090 ana PCIe x16 slotunda çalışmalıdır.
- Sistem RAM'i en az 64 GiB önerilir.
- Proje ve CARLA için NVMe SSD kullanılmalıdır.
- `/dev/shm` içinde en az 512 MiB boş alan bulunmalıdır.
- Wayland/Xorg seçimi inference'ı etkilemez; GPU'da başka ağır süreç bulunmamalıdır.

Driver otomatik kurulmaz. Önce NVIDIA paket yöneticisiyle en az `610.43.02` sürücüsü
kurulmalı ve yeniden başlatma sonrası şu komut başarılı olmalıdır:

```bash
nvidia-smi
```

## 3. Repository ve host audit

```bash
git clone https://github.com/brutalstein/carla.git
cd carla

./scripts/perception/setup_rtx5090_host.sh
```

Audit modu hiçbir paket değiştirmez. RTX 5090 ve driver doğrulandıktan sonra:

```bash
./scripts/perception/setup_rtx5090_host.sh --apply
```

Bu işlem:

1. Docker ve Compose eklentisini kurar.
2. NVIDIA Container Toolkit `1.19.1` paketlerini aynı sürümde pinler.
3. Docker runtime'ını `nvidia-ctk` ile yapılandırır.
4. CUDA MPS daemon'unu başlatır.
5. Gerçek GPU, VRAM, MPS ve `/dev/shm` kontrollerini çalıştırır.

Kurulumdan sonra Docker grup üyeliği için oturumu kapatıp yeniden açın.

## 4. CUDA MPS

Başlatma:

```bash
./scripts/perception/start_mps.sh
```

Doğrulama:

```bash
echo get_server_list | nvidia-cuda-mps-control
nvidia-smi
```

Durdurma:

```bash
./scripts/perception/stop_mps.sh
```

Script önce `EXCLUSIVE_PROCESS` compute mode'u dener. GeForce sürücüsü bunu
reddederse MPS `DEFAULT` modda devam eder ve uyarı verir. MPS, model process'lerinin
ayrı CUDA context scheduling maliyetini azaltır. Fatal CUDA hatası MPS kullanan başka
client'ları etkileyebileceği için her worker kontrollü shutdown yapmalıdır.

## 5. Container image'ları

```bash
docker compose -f infra/perception/docker-compose.cuda.yml build

docker compose -f infra/perception/docker-compose.cuda.yml up -d
```

Oluşan worker tabanları:

```text
l4stack/perception-tensorrt:11.1-cu13.3
  NVIDIA 13.3.0-tensorrt-devel Ubuntu 24.04 tabanı; trtexec, C++ headers ve
  BEVFusion Detection/Segmentation ile CitySemSegFormer engine runtime

l4stack/perception-tld:8.4.104-cu13.0
  TLD-READY ağırlığı, Ultralytics export ve TensorRT runtime

l4stack/perception-pytorch:2.12.1-cu13.0
  MapTRv2 RTX 5090 port/build ortamı
```

Container'lar `ipc: host` kullanır. Bunun nedeni host process'in oluşturduğu POSIX
shared-memory segmentlerinin container içinde aynı adla açılmasıdır. Compose ayrıca
container'ları host MPS daemon'unu başlatan kullanıcıyla aynı UID/GID altında çalıştırır.
`setup_rtx5090_host.sh --apply`, `infra/perception/.env` dosyasını otomatik üretir.

## 6. Model dosyaları

Ağırlıkları ilgili README'deki resmî kaynaktan indirip şu dizinlere koyun:

```text
models/perception/bevfusion_detection/model/bevfusion-det.pth
models/perception/bevfusion_segmentation/model/bevfusion-seg.pth
models/perception/maptrv2/model/maptrv2_r50_bevpool_24ep.pth
models/perception/tld_ready/model/traffic_lights_yolov8x.pt
models/perception/citysemsegformer/model/citysemsegformer.onnx
models/perception/citysemsegformer/model/labels.txt
```

Model ağırlıkları Git'e eklenmez.

## 7. RTX 5090 üzerinde engine üretimi

Engine dosyaları hedef makinede üretilir:

```text
bevfusion-det-sm120.engine
bevfusion-seg-sm120.engine
traffic_lights_yolov8x-sm120.engine
citysemsegformer-sm120.engine
```

Her engine build sonrasında gerçek input ile smoke test yapılmalı ve builder/runtime
sürümleri kaydedilmelidir. `trtexec` için temel doğrulama:

```bash
docker exec -it l4stack-perception-trt trtexec --help
```

CitySemSegFormer ONNX engine örneği, gerçek input binding adları modelden okunarak:

```bash
docker exec -it l4stack-perception-trt bash
trtexec \
  --onnx=/workspace/models/perception/citysemsegformer/model/citysemsegformer.onnx \
  --saveEngine=/workspace/models/perception/citysemsegformer/model/citysemsegformer-sm120.engine \
  --fp16 \
  --builderOptimizationLevel=5 \
  --useCudaGraph \
  --profilingVerbosity=detailed
```

Binding shape'leri sabit değilse `--minShapes`, `--optShapes`, `--maxShapes` modelin
gerçek giriş isimleriyle verilmelidir. Rastgele shape kullanılmaz.

BEVFusion, custom CUDA/TensorRT plugin kullandığı için engine üretimi NVIDIA
CUDA-BEVFusion deposunun export/build hattıyla yapılır. Engine dosyası oluşması tek
başına yeterli değildir; CARLA calibration ve gerçek sensör artifact'ıyla inference
smoke testi zorunludur.

TLD-READY ağırlığı önce ONNX/TensorRT'e export edilir. Export sırasında `device=0`,
`half=True` ve sabit ön işleme çözünürlüğü kullanılır. Runtime CPU fallback'i kabul
etmez.

MapTRv2'nin resmî ortamı eski PyTorch/MMCV sürümlerine bağlıdır ve RTX 5090 SM 12.0
ile doğrudan uyumlu değildir. Bu nedenle eski environment host'a kurulmaz. MapTRv2:

1. PyTorch `2.12.1+cu130` tabanında açılır.
2. MMCV CUDA ops kaynak koddan `TORCH_CUDA_ARCH_LIST=12.0` ile derlenir.
3. Eski MapTRv2 config/registry/import API'leri modern OpenMMLab API'lerine port edilir.
4. Resmî checkpoint yükleme raporu alınır.
5. Gerçek altı kamera girdisiyle smoke test geçer.
6. Sonuç `models/perception/maptrv2/model/rtx5090_port_verified.json` içine yazılır.

Bu doğrulama dosyası bulunmadan `perception-doctor` MapTRv2'yi hazır saymaz.

## 8. Worker komutları

Her worker persistent JSONL process olmalıdır. Model constructor sırasında GPU'ya yüklenir;
her frame'de yeniden yüklenmez. Örnek environment yapısı:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

export L4STACK_BEVFUSION_DETECTION_COMMAND='docker exec -i l4stack-perception-trt <gerçek-worker-komutu>'
export L4STACK_BEVFUSION_SEGMENTATION_COMMAND='docker exec -i l4stack-perception-trt <gerçek-worker-komutu>'
export L4STACK_TLD_READY_COMMAND='docker exec -i l4stack-perception-tld <gerçek-worker-komutu>'
export L4STACK_CITYSEMSEGFORMER_COMMAND='docker exec -i l4stack-perception-trt <gerçek-worker-komutu>'
export L4STACK_MAPTRV2_COMMAND='docker exec -i l4stack-perception-maptr <gerçek-worker-komutu>'
```

Client container içinde `CUDA_VISIBLE_DEVICES` elle yeniden eşlenmez; GPU seçimi Docker
device reservation ve MPS daemon tarafından yapılır. Worker `ping` yanıtında şu gerçek
değerleri vermelidir:

```json
{
  "protocol_version": 2,
  "type": "ready",
  "device": "cuda",
  "cuda_available": true,
  "cpu_fallback": false,
  "model_loaded": true,
  "compute_capability": 12.0,
  "precision": "fp16"
}
```

Bu alanlardan biri yanlışsa runtime modeli ACTIVE yapmaz.

## 9. Doğrulama sırası

```bash
l4stack --config-dir config validate
l4stack --config-dir config cuda-doctor
l4stack --config-dir config perception-doctor
```

Sonra `config/perception.yaml` içinde önce global `enabled`, ardından yalnız bir model
etkinleştirilir. Model başına 400 frame gerçek CARLA koşusu tamamlandıktan sonra diğer
model açılır.

```bash
l4stack --config-dir config run --frames 400
python scripts/perception/benchmark_guard.py output/frames.jsonl \
  --minimum-frames 400 \
  --model-p95-ms bevfusion_detection=80 \
  --model-p95-ms tld_ready=50
```

Benchmark guard boş/sentetik logu kabul etmez ve en az bir gerçek model çıktısı ister.

## 10. Resmî kaynaklar

- CUDA 13.3 release notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- TensorRT 11.1: https://docs.nvidia.com/deeplearning/tensorrt/latest/
- TensorRT install: https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html
- PyTorch versions: https://pytorch.org/get-started/previous-versions/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- CUDA MPS: https://docs.nvidia.com/deploy/mps/latest/
