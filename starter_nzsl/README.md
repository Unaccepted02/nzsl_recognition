# NZSL Starter Prototype

Verified NZSL-only training workspace. All data must be traceable to a documented source and licence.

## Scope

- Language: NZSL only
- Vocabulary target: 30+ classes
- Domain focus: public transport, mobility, accessibility, and common interactions
- Output artifacts: accuracy, macro-F1, confusion matrix, Streamlit demo

## Structure

```text
starter_nzsl/
  app/
    streamlit_app.py
  config/
    augmentation_policy.json
    bsl_nzsl_cognate_candidates.csv
    priority_batch_01.csv
    verified_nzsl_sources.csv
    verified_nzsl_vocab_30plus.csv
  data/
    metadata/
      verified_samples_master.csv
      verified_samples_template.csv
      manual_capture_manifest.csv
    processed/
    raw/
  docs/
    experiment_log.md
    manual_capture_guide.md
  reports/
  reports_analysis/
  reports_cv7_all/
  reports_sklearn/
  reports_transformer_cv7_all/
  reports_transformer_optimized/
  scripts/
    build_clean_raw_dataset.py
    download_seed_samples.py
    generate_report_assets.py
    init_verified_workspace.py
    seed_nzsl_online_batch_01.py
    seed_nzsl_online_batch_02.py
    seed_nzsl_online_batch_03.py
```

## Source policy

Use [`config/verified_nzsl_sources.csv`](./config/verified_nzsl_sources.csv) as the source registry.

- `approved`: data may be ingested once the clip-level label is verified
- `review_required`: inspect licensing and clip suitability before training
- `pending`: reserved for future capture workflows

`NZSL Online` is the primary source. Community and archive sites can be added only after rights and clip suitability are documented.

## Vocabulary policy

Use [`config/verified_nzsl_vocab_30plus.csv`](./config/verified_nzsl_vocab_30plus.csv) as the target vocabulary file.

- `status=active`: collect now
- `status=planned`: collect after the first pass
- `target_samples`: rough clip count goal per label

The first collection pass is fixed in [`config/priority_batch_01.csv`](./config/priority_batch_01.csv).

## Sample metadata policy

Every clip must appear in [`data/metadata/verified_samples_template.csv`](./data/metadata/verified_samples_template.csv) with:

- source id
- source URL
- license
- local path
- NZSL verification flag
- verifier name or initials

If a sample cannot be traced, do not train on it.

## Augmentation policy

The default policy is documented in [`config/augmentation_policy.json`](./config/augmentation_policy.json).

- Safe by default: temporal crop, frame jitter, speed perturbation, noise, scale jitter, keypoint dropout
- Disabled by default: horizontal flip, large rotation, cross-sign mixup

## Workflow

Initialize a fresh workspace:

```bash
.venv\Scripts\python.exe -m starter_nzsl.scripts.init_verified_workspace
```

After placing verified NZSL clips in `starter_nzsl/data/raw/`:

```bash
# Extract keypoints
.venv\Scripts\python.exe -m src.extract_keypoints ^
  --raw_dir starter_nzsl\data\raw_cleaned ^
  --out_dir starter_nzsl\data\processed ^
  --num_frames 60

# Preprocess
.venv\Scripts\python.exe -m src.preprocess ^
  --processed_dir starter_nzsl\data\processed ^
  --out_dir starter_nzsl\models

# Classical ML 7-fold CV
.venv\Scripts\python.exe -m src.train_sklearn_cv ^
  --processed_dir starter_nzsl\data\processed ^
  --out_dir starter_nzsl\models_cv7_all ^
  --reports_dir starter_nzsl\reports_cv7_all ^
  --folds 7

# Transformer 7-fold CV
.venv\Scripts\python.exe -m src.train_transformer_cv ^
  --processed_dir starter_nzsl\data\processed ^
  --out_dir starter_nzsl\models_transformer_cv7_all ^
  --reports_dir starter_nzsl\reports_transformer_cv7_all ^
  --folds 7 --epochs 20 --final_epochs 20 ^
  --augment --temporal_crop --keypoint_preprocess --balanced_sampler

# Evaluate
.venv\Scripts\python.exe -m src.evaluate ^
  --processed_dir starter_nzsl\data\processed ^
  --model_type sklearn ^
  --model_path starter_nzsl\models\sklearn_rf.joblib ^
  --reports_dir starter_nzsl\reports_sklearn
```

Run the demo:

```bash
run_starter_demo.bat
```
