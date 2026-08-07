# Perception Modelleri — Kurulum Sırası

Model ağırlıkları ve üçüncü taraf repository'ler Git'e eklenmez. Her modelin `model/`
ve `external/` dizini yalnızca yerel kurulum içindir.

## Klasörler

```text
models/perception/
├── bevfusion_detection/
├── bevfusion_segmentation/
├── maptrv2/
├── tld_ready/
└── citysemsegformer/
```

Her klasörde:

- `README.md`: resmî kurulum, ağırlık yolu ve backend sözleşmesi.
- `run_backend.sh`: ana runtime'ın çağırdığı süreç giriş noktası.
- `model/`: checkpoint/ONNX dosyaları.
- `external/`: resmî üçüncü taraf kod tabanı.

## Önerilen sıra

1. Önce ana stack testlerini çalıştır:

   ```bash
   python -m pip install -e '.[dev]'
   pytest -q
   ```

2. Protokolü gerçek model olmadan test et:

   ```bash
   L4STACK_PERCEPTION_MOCK=1 \
     bash models/perception/bevfusion_detection/run_backend.sh
   ```

   Bu komut stdin'de JSONL bekler. Otomatik protokol testi pytest içinde de vardır.

3. Yalnızca kullanacağın modelin README'sini uygula ve dosyaları belirtilen `model/`
   dizinine koy.

4. İzole ortamda gerçek JSONL runner komutunu hazırla ve ilgili environment değişkenini
   tanımla. Örnek:

   ```bash
   export L4STACK_MAPTRV2_COMMAND='conda run -n maptr python /absolute/path/to/maptr_backend.py'
   ```

5. Config ve dosya kontrolü:

   ```bash
   l4stack --config-dir config perception-doctor
   ```

6. `config/perception.yaml` içinde önce tek modeli etkinleştir:

   ```yaml
   perception:
     enabled: true
     models:
       bevfusion_detection:
         enabled: true
   ```

7. CARLA'da kısa bir run yap, `output/frames.jsonl` içindeki deadline/health/pipeline
   sayaçlarını incele. Daha sonra diğer modelleri tek tek etkinleştir.

## Backend komut değişkenleri

```text
L4STACK_BEVFUSION_DETECTION_COMMAND
L4STACK_BEVFUSION_SEGMENTATION_COMMAND
L4STACK_MAPTRV2_COMMAND
L4STACK_TLD_READY_COMMAND
L4STACK_CITYSEMSEGFORMER_COMMAND
```

Komutun başlattığı program stdin/stdout JSONL protokolünü uygulamalıdır. Protokol
ayrıntıları `docs/PERCEPTION_ARCHITECTURE.md` içindedir.

## Model ağırlıkları neden Git'te değil?

- GitHub dosya boyutu ve clone maliyeti.
- Üçüncü taraf lisans/dağıtım şartları.
- Checkpoint sürümünün config manifest'iyle açıkça yönetilmesi.
- Model ve kod güncellemesinin ana stack commit geçmişinden ayrılması.
