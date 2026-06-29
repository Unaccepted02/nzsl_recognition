# NZSL Recognition Experiment Log

## 2026-06-23 Transformer robustness update

### Motivation

The original Transformer showed much lower held-out test performance than the classical sklearn baseline. The likely causes were small data volume, noisy/high-dimensional holistic keypoints, train/test domain shift, and an oversized Transformer configuration for roughly one thousand samples.

### Code changes

- Added checkpoint-aware torch inference preprocessing in `src/predict.py`.
- Updated Streamlit torch inference to pass raw extracted keypoint sequences and let `predict_sequence_torch` apply the correct model-specific preprocessing.
- Updated `src/train_transformer.py` with optional keypoint preprocessing:
  - repair missing hand/face landmark blocks,
  - normalize by shoulder scale,
  - trim low-motion sequence edges,
  - resample back to fixed sequence length for batching.
- Recomputed Transformer feature standardization from the preprocessed training split when this option is enabled.
- Added optional class-balanced sampler.
- Reduced default Transformer capacity:
  - `d_model`: 256 -> 128
  - `num_layers`: 3 -> 2
  - `nhead`: 8 -> 4
  - `dim_feedforward`: 512 -> 256
  - `dropout`: 0.2 -> 0.4
  - `weight_decay`: 0 -> 1e-3

### Planned training command

```powershell
.venv\Scripts\python.exe -m src.train_transformer `
  --processed_dir starter_nzsl\data\processed `
  --out_dir starter_nzsl\models_transformer_optimized `
  --epochs 30 `
  --batch_size 32 `
  --augment `
  --temporal_crop `
  --keypoint_preprocess `
  --balanced_sampler
```

### Evaluation note

The first evaluation remains the strict filename-held-out split in `starter_nzsl/data/processed`, where `*-test-*` samples are not used for training. This is intentionally harder than the 7-fold CV setup and is kept for comparison with earlier Transformer results.

### Result

Optimized Transformer output:

- Model: `starter_nzsl/models_transformer_optimized/best_transformer.pt`
- Report: `starter_nzsl/reports_transformer_optimized/metrics.json`
- Best validation macro F1 during training: `0.9865`
- Strict held-out test accuracy: `0.0909`
- Strict held-out test macro F1: `0.0603`
- Strict held-out top-3 accuracy: `0.1688`
- Strict held-out top-5 accuracy: `0.2338`

Baseline Transformer comparison:

- Baseline strict held-out test accuracy: `0.0649`
- Baseline strict held-out test macro F1: `0.0193`
- Baseline strict held-out top-5 accuracy: `0.2208`

### Interpretation

The optimized Transformer improved over the previous Transformer on the strict held-out test split, but the gain was small. Validation performance remained very high while held-out performance remained low, which supports the earlier conclusion that the main problem is domain shift between the training/validation recordings and the filename-held-out test recordings, not only Transformer capacity. The smaller Transformer and keypoint preprocessing reduce overfitting/noise somewhat, but do not solve distribution mismatch.

## 2026-06-23 Transformer 7-fold CV setup

### Motivation

The project decision changed to ignore the original filename-based `test` tag for subsequent experiments. This matches the sklearn `cv7_all` setup and treats all 1015 samples as one dataset for stratified cross-validation.

### Code changes

- Added `src/train_transformer_cv.py`.
- The script uses all rows in `starter_nzsl/data/processed/labels.csv` regardless of the existing `split` column.
- Evaluation uses stratified 7-fold cross-validation.
- Each fold trains an optimized small Transformer with:
  - keypoint preprocessing,
  - augmentation,
  - temporal crop,
  - class-balanced sampler,
  - `d_model=128`,
  - `num_layers=2`,
  - `nhead=4`,
  - `dropout=0.4`,
  - `weight_decay=1e-3`.
- After CV, a final Transformer is trained on all samples for Streamlit/demo use.

### Planned command

```powershell
.venv\Scripts\python.exe -m src.train_transformer_cv `
  --processed_dir starter_nzsl\data\processed `
  --out_dir starter_nzsl\models_transformer_cv7_all `
  --reports_dir starter_nzsl\reports_transformer_cv7_all `
  --folds 7 `
  --epochs 20 `
  --final_epochs 20 `
  --augment `
  --temporal_crop `
  --keypoint_preprocess `
  --balanced_sampler
```

### Result

Transformer 7-fold CV completed using all 1015 samples and ignoring the original filename-based `test` split.

- Report: `starter_nzsl/reports_transformer_cv7_all/cv_summary.json`
- Fold results: `starter_nzsl/reports_transformer_cv7_all/cv_results.csv`
- Final all-data model: `starter_nzsl/models_transformer_cv7_all/final_all/best_transformer.pt`
- Mean accuracy: `0.9320`
- Accuracy std: `0.0156`
- Mean macro F1: `0.9328`
- Macro F1 std: `0.0150`

### Interpretation

Under stratified 7-fold CV, the optimized Transformer performs similarly to the sklearn CV baseline. This confirms that the Transformer architecture can learn the current vocabulary when the original `*-test-*` domain split is ignored. The previous poor Transformer results were therefore mainly caused by the filename-held-out domain shift rather than by the Transformer being unable to model the signs.
