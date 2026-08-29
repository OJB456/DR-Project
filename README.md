# RetinaScreen

## Explainable AI for diabetic retinopathy screening in rural India

RetinaScreen is a research prototype for **AI-assisted screening** of diabetic
retinopathy (DR) from fundus photographs. It is designed around a rural PHC
workflow: capture → image quality gate → local preprocessing → 5-class model →
Grad-CAM explanation → referable-DR flag → specialist review queue.

It is not a diagnosis, does not replace an ophthalmologist, is not clinically
validated or regulator-approved, and must not be used as the sole basis for
patient care.

## What is implemented

- Non-destructive audit of the local APTOS dataset with nested-directory mapping.
- Lazy PyTorch dataset loading; image pixels are read one at a time.
- Conservative black-border crop, resize, and optional mild CLAHE preprocessing.
- EfficientNet-B0 transfer-learning classifier with five DR grades.
- Training-only inverse-frequency class weights, AdamW, validation, checkpointing,
  early stopping, reproducible seeds, CPU fallback, and CUDA AMP when available.
- Untouched test evaluation with accuracy, balanced accuracy, macro/weighted
  precision/recall/F1, per-class metrics, QWK, referable sensitivity/specificity,
  confusion matrix, and classification report.
- Genuine Grad-CAM from model activations and gradients.
- Classical quality gate covering blur, brightness, contrast, field of view, and
  image integrity. Thresholds are prototype engineering thresholds, not clinical
  thresholds.
- Streamlit dashboard with cached local model loading, probability display,
  confidence safety flag, HTML/JSON screening reports, and medical disclaimer.
- Simulink construction specification in `docs/simulink_architecture.md`; no
  MATLAB installation is required for the Python system.

## Dataset

The authoritative dataset is already present at `archive/` and is never
downloaded by the application. The verified mapping is:

| Split | Labels | Images | Mapping |
| --- | ---: | ---: | --- |
| Train | 2,930 | 2,930 | `train_1.csv` → `train_images/train_images` |
| Validation | 366 | 366 | `valid.csv` → `val_images/val_images` |
| Test | 366 | 366 | `test.csv` → `test_images/test_images` |

The audit found zero missing, extra, corrupt, duplicate-ID, or cross-split
ID-overlap findings. Class counts and sample dimensions are recorded in
[`artifacts/dataset_report.json`](artifacts/dataset_report.json). APTOS image
labels do not provide lesion masks; Grad-CAM is the explainability mechanism,
not lesion segmentation.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dataset and `archive.zip` are ignored by Git. Do not move, duplicate,
unzip, or commit them.

## Verify the dataset and runtime

```powershell
python scripts/inspect_dataset.py
python scripts/sanity_check.py
python -m pytest -q
```

The sanity check verifies one sample from every split, one batch, and the
EfficientNet output shape `[B, 5]`.

## Train

For the full reproducible training run:

```powershell
python scripts/train_model.py --config config.yaml
```

Useful bounded CPU smoke options are `--epochs 1 --max-train-samples 64
--max-val-samples 32`. Set `pretrained: false` in `config.yaml` for a fully
offline initialization; with `pretrained: true`, torchvision may download the
ImageNet initialization once during training. The app itself never downloads
anything.

Outputs:

- `artifacts/checkpoints/best_model.pt`
- `artifacts/checkpoints/last_model.pt`
- `artifacts/training_history.json`
- `artifacts/training_config.json`
- `artifacts/visualizations/`

## Evaluate

Evaluation always uses the official untouched test CSV/image split:

```powershell
python scripts/evaluate_model.py --config config.yaml --checkpoint artifacts/checkpoints/best_model.pt
```

Outputs are `artifacts/metrics.json`,
`artifacts/classification_report.txt`, and
`artifacts/confusion_matrix.png`.

The recorded runtime evidence is a one-epoch, 64-image CPU smoke baseline,
not a production-quality training run. Its actual full-test results were:

- Accuracy: 47.27%; balanced accuracy: 24.50%; QWK: 0.3730.
- Referable sensitivity: 68.61%; specificity: 72.49%; precision: 59.87%;
  F1: 63.95%.

The project targets (>90% sensitivity and >85% specificity) were **not met** by
this smoke baseline. No target values are fabricated or hard-coded.

## Run the demo

After a checkpoint exists locally:

```powershell
python -m streamlit run app.py
```

Open the displayed local URL, upload a fundus image, and review the quality
gate, predicted grade, actual class probabilities, referable flag, Grad-CAM
heatmap/overlay, and downloadable report. If the quality gate rejects an image,
the model result is withheld and recapture is recommended. A low model
confidence is explicitly flagged and specialist review is recommended.

The referable rule is: grades 0–1 non-referable, grades 2–4 referable. A
non-referable screening result does not establish that disease is absent.

## Project layout

```text
app.py                         Streamlit review dashboard
pipeline.py                    Backward-compatible CLI facade
config.yaml                    Relative paths and thresholds
src/data                       Lazy dataset and validation
src/preprocessing              Fundus preprocessing
src/models                     EfficientNet-B0 and checkpoints
src/training                   Training and sanity checks
src/evaluation                 Metrics and plots
src/quality                    Image quality gate
src/explainability             Grad-CAM
src/inference                  One-call screening API
src/reporting                  HTML/JSON reports
scripts/                       Dataset, training, evaluation, sanity entrypoints
tests/                         Automated core tests
docs/simulink_architecture.md  MATLAB/Simulink specification
```

## Limitations and next improvements

The available executed model evidence is deliberately limited to a bounded
CPU smoke run, so its metrics are weak and not clinically meaningful. Next
steps are a full multi-epoch run on suitable GPU hardware, validation-only
referable threshold tuning, calibration analysis, controlled augmentation/loss
experiments, external validation, and ophthalmologist-led clinical review.
APTOS image-level labels do not support claims about lesion localization.
