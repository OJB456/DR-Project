# Development Status

## Audit date

2026-08-28

## Status legend

- `[ ]` not started
- `[~]` in progress / limited evidence
- `[x]` complete for the implemented prototype
- `[!]` blocked or not yet achieved

## Feature status

- `[x]` Dataset: local `archive/` audited in place; no download, move, copy, deletion, or full-image RAM load.
- `[x]` Data pipeline: CSV-to-nested-directory mapping, lazy `APTOSDataset`, RGB conversion, transforms, and error handling.
- `[x]` Preprocessing: conservative black-border crop, resize, optional mild CLAHE, configurable in `config.yaml`.
- `[x]` Model: torchvision EfficientNet-B0 with five-class classifier and offline fallback when pretrained weights are unavailable.
- `[~]` Training: AdamW, training-only class weights, validation, checkpoint resume, early stopping, reproducibility, and CPU/CUDA AMP support implemented. A bounded CPU smoke run was executed; a full 15-epoch run remains for suitable hardware.
- `[x]` Evaluation: untouched test-set metrics, per-class metrics, QWK, referable metrics, confusion matrix, and classification report generated.
- `[x]` Grad-CAM: genuine activation/gradient-based heatmap tested on a real local image; original, heatmap, and overlay saved.
- `[x]` Quality gate: blur, brightness, contrast, field of view, integrity, 0–100 score, reasons, and ACCEPT/REJECT policy.
- `[x]` Inference: quality gate → preprocessing → model → probabilities → referable flag → Grad-CAM → report.
- `[x]` Streamlit: cached local model, upload flow, safety states, probabilities, explanation, and report downloads.
- `[x]` Reporting: self-contained HTML and JSON reports with case ID, timestamp, quality, result, recommendation, Grad-CAM, and disclaimer.
- `[x]` Testing: 7 automated core tests passing.
- `[x]` Simulink: honest architecture specification added; MATLAB/Simulink was not claimed or emulated.
- `[x]` Documentation: README, config, `.env.example`, Git ignore rules, and rural deployment notes.

## Verified dataset evidence

`artifacts/dataset_report.json` was refreshed with the explicit nested paths:

- Train: 2,930 labels / 2,930 PNG images; classes `{0: 1434, 1: 300, 2: 808, 3: 154, 4: 234}`.
- Validation: 366 labels / 366 PNG images; classes `{0: 172, 1: 40, 2: 104, 3: 22, 4: 28}`.
- Test: 366 labels / 366 PNG images; classes `{0: 199, 1: 30, 2: 87, 3: 17, 4: 33}`.
- All mappings verified; zero missing/extra/corrupt images, duplicate IDs, or cross-split ID overlap.
- Content duplicate hashing was intentionally not repeated because it requires reading every byte of the approximately 9 GB local archive.

## Executed runtime evidence

Environment: Python 3.11.6, PyTorch 2.13.0+cpu, torchvision 0.28.0+cpu, CUDA unavailable.

Sanity check passed: samples from all three splits produced `[3, 224, 224]`, a batch produced `[4, 3, 224, 224]`, and EfficientNet-B0 produced `[4, 5]` logits.

Executed bounded smoke training: one epoch, 64 train samples, 32 validation samples,
96px training input, CPU. Training-only class weights were
`[0.408647, 1.953333, 0.725248, 3.805195, 2.504273]`.

Smoke training metrics:

- Train loss: 1.6404
- Validation loss: 1.5895
- Validation accuracy: 18.75%
- Validation QWK: 0.0496

The resulting checkpoint was evaluated on the untouched 366-image test split:

- Accuracy: 47.27%
- Balanced accuracy: 24.50%
- Macro precision / recall / F1: 23.20% / 24.50% / 23.42%
- Weighted precision / recall / F1: 48.92% / 47.27% / 47.89%
- Quadratic weighted kappa: 0.3730
- Referable sensitivity: 68.61%
- Referable specificity: 72.49%
- Referable precision / recall / F1: 59.87% / 68.61% / 63.95%

The SIH target (>90% sensitivity and >85% specificity) is not met by this
smoke baseline. These values are actual executed metrics and are not clinical
validation.

Real-image inference passed: `DR-DEMO` quality `100.0/100 ACCEPT`; model result
was low confidence and correctly produced a specialist-review recommendation,
Grad-CAM artifacts, and HTML/JSON reports. Streamlit started on localhost and
`/_stcore/health` returned `ok`.

## Known limitations

- No GPU was available; only bounded CPU smoke training was practical here.
- No threshold tuning, calibration, external validation, clinical validation,
  or regulatory assessment has been performed.
- The low smoke baseline must not be presented as a deployed screening model.
- APTOS image-level labels do not contain lesion masks; Grad-CAM is model
  influence visualization, not proof of lesions.
- MATLAB/Simulink was not installed/verified, so only its construction
  specification is included.
