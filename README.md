# NZSL Recognition Prototype

A proof-of-concept isolated sign language recognition (ISLR) system for New Zealand Sign Language (NZSL), focused on transport, accessibility, and daily communication vocabulary for Auckland Transport.

## Overview

This project implements an end-to-end pipeline for recognising isolated NZSL signs from short video clips:

1. **Keypoint extraction** — MediaPipe Holistic extracts a 315-dimensional per-frame feature vector (pose, hands, face subset) from raw video
2. **Preprocessing** — sequences are resampled to 60 frames, keypoints are repaired and shoulder-scale normalised
3. **Model training** — classical ML (Random Forest, SVM) and Transformer encoder classifiers trained via stratified 7-fold cross-validation
4. **Evaluation** — stratified 7-fold CV reporting accuracy and macro-F1
5. **Demo** — Streamlit web app for uploading a clip and viewing top predictions

Under stratified 7-fold cross-validation on a 23-class, 1015-sample dataset:

| Model | Mean Accuracy | Mean Macro-F1 |
|---|---|---|
| SVM | 0.9133 | 0.9153 |
| Random Forest | 0.9291 | 0.9289 |
| Transformer encoder | **0.9320** | **0.9328** |

## Data policy

- Training data uses verified NZSL sources only (primary: [NZSL Online](https://www.nzsl.nz/), CC BY-NC-SA 4.0)
- Every sample must be traceable to a source URL and licence via `starter_nzsl/data/metadata/`

## Setup

```bash
cd nzsl_recognition
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline commands

```powershell
# 1. Clean and manifest raw NZSL clips
.venv\Scripts\python.exe starter_nzsl\scripts\build_clean_raw_dataset.py `
  --raw_dir starter_nzsl\data\raw `
  --out_dir starter_nzsl\data\raw_cleaned `
  --manifest starter_nzsl\reports_analysis\cleaning_manifest.json

# 2. Keypoint extraction (MediaPipe Holistic -> 315-D per-frame, 60 frames/clip)
.venv\Scripts\python.exe -m src.extract_keypoints `
  --raw_dir starter_nzsl\data\raw_cleaned `
  --out_dir starter_nzsl\data\processed `
  --num_frames 60

# 3. Preprocessing and labels
.venv\Scripts\python.exe -m src.preprocess `
  --processed_dir starter_nzsl\data\processed `
  --out_dir starter_nzsl\models

# 4. Classical ML 7-fold CV
.venv\Scripts\python.exe -m src.train_sklearn_cv `
  --processed_dir starter_nzsl\data\processed `
  --out_dir starter_nzsl\models_cv7_all `
  --reports_dir starter_nzsl\reports_cv7_all `
  --folds 7

# 5. Transformer 7-fold CV
.venv\Scripts\python.exe -m src.train_transformer_cv `
  --processed_dir starter_nzsl\data\processed `
  --out_dir starter_nzsl\models_transformer_cv7_all `
  --reports_dir starter_nzsl\reports_transformer_cv7_all `
  --folds 7 --epochs 20 --final_epochs 20 `
  --augment --temporal_crop --keypoint_preprocess --balanced_sampler

# 6. Streamlit demo
run_starter_demo.bat
```

## Demo

```bat
run_starter_demo.bat
```

Then open `http://localhost:8501`.

## Repository structure

```
src/                        # Core pipeline modules
starter_nzsl/
  app/                      # Streamlit demo
  config/                   # Vocabulary, augmentation policy, source registry
  data/metadata/            # Verified sample manifests (video files not tracked)
  docs/                     # Manual capture guide, experiment log
  reports_analysis/         # Class distribution and model comparison figures
  reports_cv7_all/          # Classical ML 7-fold CV results
  reports_transformer_cv7_all/  # Transformer 7-fold CV results
  reports/                  # BiLSTM strict-split diagnostic
  reports_sklearn/          # Random Forest strict-split diagnostic
  reports_transformer_optimized/  # Transformer strict-split diagnostic
  scripts/                  # Data preparation and reporting scripts
requirements.txt
run_starter_demo.bat
```

