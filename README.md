# توليد شيخوخة الوجه التتابعية باستخدام GAN

موديل **StarGAN متعدّد المجموعات العمرية** لتوليد صور تقدّم العمر (Face Aging Progression) من صورة وجه مُدخلة. مولّد واحد `G(x, c)` يحوّل الوجه إلى أي **مجموعة عمرية** هدف `c`، فنحصل على **تتابع شيخوخة** (الإدخال → 0-19 → 20-29 → 30-39 → 40-49 → 50+).

- التدريب على **UTKFace** (يحتوي العمر في اسم كل ملف).
- حجم الصور **456×456**.
- حفظ **الأوزان + صورة عيّنات لكل إيبوك**، وتتبّع **أفضل أوزان** تلقائيًا (أقل FID على التحقق).
- مقاييس التقييم: **FID, SSIM, LPIPS, PSNR, CSIM**.
- مُحسّن لـ **Google Colab / Kaggle (GPU)**، ويعمل أيضًا على CPU/Apple MPS.

> ملاحظة واقعية: التدريب الكامل للوصول إلى أوزان مثالية يستغرق **ساعات إلى أيام** على GPU واحد. ابدأ بإعداد `configs/debug.yaml` للتأكد أن كل شيء يعمل، ثم انتقل إلى `configs/utkface_456.yaml`.

---

## 1) البنية المعمارية باختصار

- **المولّد (Generator):** Encoder (تصغير) → 6 كتل Residual → Decoder (تكبير). نُدخل صورة الوجه + خريطة المجموعة العمرية الهدف (one-hot موسّعة مكانيًا) ونُخرج الوجه بعد الشيخوخة. الـ Cycle-consistency يضمن إمكانية الرجوع للأصل ويحافظ على الهوية.
- **المميّز (Discriminator):** PatchGAN له رأسان: `out_src` (حقيقي/مزيّف) و`out_cls` (تصنيف المجموعة العمرية، عبر AdaptiveAvgPool + Linear ليتوافق مع أي حجم مثل 456).
- **دوال الخسارة:** Adversarial (hinge افتراضيًا، أو WGAN-GP)، تصنيف المجموعة (Cross-Entropy)، إعادة بناء دورية (L1)، وخسارة هوية (FaceNet) لرفع CSIM.

```
الإدخال x ──► G(x, c=الهدف) ──► وجه مُشيَّخ
                    │
                    └──► G(الناتج, c=الأصل) ──► إعادة البناء (Cycle / حفظ الهوية)
```

## 2) هيكل المشروع

```
face-model/
├── configs/
│   ├── utkface_456.yaml      # الإعداد الأساسي (456×456)
│   └── debug.yaml            # إعداد سريع للتجربة (128×128، 3 إيبوك)
├── src/
│   ├── config.py  utils.py  data.py  models.py
│   ├── identity.py  losses.py  metrics.py  train.py
├── scripts/
│   ├── prepare_utkface.py    # فحص الداتا ست وتوزيع الأعمار
│   ├── train.py              # تشغيل التدريب
│   ├── generate.py           # توليد شيخوخة تتابعية من صورة
│   └── evaluate.py           # حساب المقاييس الخمسة
├── notebooks/colab_train.ipynb
├── requirements.txt
└── outputs/<run_name>/       # تُنشأ أثناء التدريب
    ├── weights/
    │   ├── epoch_000/ G.pth D.pth     ◄── أوزان كل إيبوك
    │   ├── epoch_001/ ...
    │   ├── best_G.pth  best_D.pth     ◄── الأوزان المثالية (أقل FID)
    │   └── latest_ckpt.pth            ◄── لاستئناف التدريب
    ├── samples/epoch_000.png ...      ◄── صور كل إيبوك (تتابع الأعمار)
    ├── logs/ loss_log.csv  config.json
    └── eval/ metrics_test.json
```

## 3) التثبيت

```bash
pip install -r requirements.txt
# مهم: ثبّت facenet-pytorch بدون تبعيات حتى لا يخفّض إصدار torch:
pip install facenet-pytorch --no-deps
```
على Colab/Kaggle يكون torch مثبّتًا مسبقًا — يكفي تثبيت بقية الحزم. إذا تعذّر تثبيت `facenet-pytorch` سيستمر التدريب لكن **بدون** خسارة الهوية ومقياس CSIM (تحذير فقط).

ملاحظات:
- مقياس **FID** يتطلب حزمة `torch-fidelity` (مضمّنة في `requirements.txt`).
- على **Apple MPS** يُحسب FID تلقائيًا على CPU (لأن MPS لا يدعم float64)؛ على CUDA يعمل على الـGPU مباشرة. بقية المقاييس تعمل على الجهاز نفسه.

## 4) تجهيز الداتا ست UTKFace

نزّل UTKFace وضعه في `data/UTKFace/` (الأسماء بصيغة `25_0_0_20170116.jpg`):

```bash
# عبر Kaggle API:
kaggle datasets download -d jangedoo/utkface-new
unzip -q utkface-new.zip -d data/UTKFace

# تحقّق من الداتا وتوزيع المجموعات العمرية:
python scripts/prepare_utkface.py --data-root data/UTKFace
```

## 5) التدريب

```bash
# تجربة سريعة أولًا (يُنصح بها):
python scripts/train.py --config configs/debug.yaml --data-root data/UTKFace

# التدريب الكامل 456×456:
python scripts/train.py --config configs/utkface_456.yaml --data-root data/UTKFace

# استئناف بعد انقطاع:
python scripts/train.py --config configs/utkface_456.yaml --resume
```
يُحفظ بعد **كل إيبوك**: مجلد أوزان `weights/epoch_XXX/`، وصورة عيّنات `samples/epoch_XXX.png` (كل صف = شخص: الإدخال ثم كل مجموعة عمرية)، ويُحدَّث `best_G.pth` عند تحسّن FID على التحقق.

## 6) التوليد (شيخوخة تتابعية)

```bash
# كل المجموعات العمرية لصورة واحدة:
python scripts/generate.py --config configs/utkface_456.yaml \
  --weights outputs/utkface_456/weights/best_G.pth \
  --input my_face.jpg --output results/

# أعمار أكبر فقط (40-49 و 50+) لمجلد صور، مع اقتصاص الوجه:
python scripts/generate.py --config configs/utkface_456.yaml \
  --weights outputs/utkface_456/weights/best_G.pth \
  --input ./faces/ --output results/ --targets 3,4 --align
```
المخرجات: صورة لكل مجموعة `{name}_to_{age}.png` + شريط مجمّع `{name}_aging_strip.png`. فهرس المجموعات: `0=0-19, 1=20-29, 2=30-39, 3=40-49, 4=50+`.

## 7) التقييم بالمقاييس الخمسة

```bash
python scripts/evaluate.py --config configs/utkface_456.yaml \
  --weights outputs/utkface_456/weights/best_G.pth --split test
```
يطبع النتائج ويحفظ `outputs/<run>/eval/metrics_test.json`.

| المقياس | يقيس | الاتجاه الأفضل | البروتوكول هنا |
|---|---|---|---|
| **FID**   | واقعية التوزيع | ↓ أقل | لكل مجموعة: صور حقيقية بعمرها مقابل صور مُولّدة لذلك العمر، ثم المتوسط |
| **SSIM**  | تشابه بنيوي | ↑ أعلى | الناتج مقابل الإدخال |
| **PSNR**  | نسبة الإشارة للضجيج | ↑ أعلى | الناتج مقابل الإدخال |
| **LPIPS** | مسافة إدراكية | ↓ أقل | الناتج مقابل الإدخال |
| **CSIM**  | حفظ الهوية (تشابه كوني للوجه) | ↑ أعلى | تضمين FaceNet للإدخال مقابل الناتج |

## 8) التشغيل على Colab

افتح `notebooks/colab_train.ipynb`، أو اتبع: تفعيل GPU → ربط Google Drive (لحفظ الأوزان) → تثبيت المتطلبات → تحميل UTKFace → التدريب مع `--output-root /content/drive/MyDrive/face_aging_out`.

## 9) ملاحظات الذاكرة (456×456 على 16GB)

- ابدأ بـ `batch_size: 4`؛ إذا ظهر **CUDA out of memory** فاخفضه إلى 2 (أو 1).
- أبقِ `use_amp: true` (الدقة المختلطة تقلّل الذاكرة كثيرًا على CUDA).
- `adv_loss: wgan-gp` أو `r1_gamma > 0` **يعطّل AMP** (يحتاج اشتقاقًا مزدوجًا) — استخدم `hinge` للحفاظ على AMP.
- لتوفير القرص: اضبط `keep_last_n_epochs: 10` لإبقاء آخر 10 إيبوكات فقط (مع `best_G.pth` دائمًا).

## 10) حل المشكلات

- **"No valid UTKFace images"**: تأكد أن الملفات بصيغة `العمر_..._..._تاريخ.jpg` تحت `--data-root`.
- **CSIM = NaN / "facenet unavailable"**: ثبّت `facenet-pytorch --no-deps`.
- **FID = NaN لمجموعة**: عدد الصور الحقيقية في تلك المجموعة قليل جدًا؛ زِد `max_eval_images` أو وسّع حدود `age_bins`.
- **بطء شديد على الماك**: هذا متوقع على MPS بحجم 456؛ استخدم Colab/Kaggle للتدريب الفعلي.
