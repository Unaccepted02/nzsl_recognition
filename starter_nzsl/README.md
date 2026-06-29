# NZSL Starter Prototype

This module is the **only active training workspace** for the project. It is designed for a verified NZSL-only pipeline with documented sources, explicit licenses, and an expandable `30+` class vocabulary focused on transport, accessibility, and daily communication.

## Scope

- Language: NZSL only
- Vocabulary target: 30+ classes
- Domain focus: public transport, mobility, accessibility, and common interactions
- Input priority: uploaded short videos
- Output artifacts: train/val/test split, accuracy, macro F1, confusion matrix, Streamlit demo

## Structure

```text
starter_nzsl/
  app/
    streamlit_app.py
  config/
    augmentation_policy.json
    auslan_nzsl_cognate_candidates.csv
    bsl_nzsl_cognate_candidates.csv
    mm_wlauslan_selected_glosses.csv
    verified_nzsl_sources.csv
    verified_nzsl_vocab_30plus.csv
  data/
    metadata/
      manual_capture_manifest.csv
      verified_samples_template.csv
    processed/
    raw/
  docs/
    manual_capture_guide.md
  models/
  reports/
  scripts/
    build_dataset.py
    init_verified_workspace.py
    prepare_mm_wlauslan_aux.py
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

The first collection pass is fixed in [`config/priority_batch_01.csv`](./config/priority_batch_01.csv). Use [`data/metadata/batch_01_manifest_seed.csv`](./data/metadata/batch_01_manifest_seed.csv) to track search progress and verified sample counts.

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

## Auxiliary planning files

- [`data/metadata/manual_capture_manifest.csv`](./data/metadata/manual_capture_manifest.csv): priority NZSL clips to record next
- [`docs/manual_capture_guide.md`](./docs/manual_capture_guide.md): recording and consent guidance
- [`config/bsl_nzsl_cognate_candidates.csv`](./config/bsl_nzsl_cognate_candidates.csv): BSL-to-NZSL auxiliary mapping candidates for future transfer experiments
- [`config/auslan_nzsl_cognate_candidates.csv`](./config/auslan_nzsl_cognate_candidates.csv): Auslan-to-NZSL auxiliary mapping candidates for transfer experiments
- [`config/mm_wlauslan_selected_glosses.csv`](./config/mm_wlauslan_selected_glosses.csv): curated MM-WLAuslan glosses used in the current Auslan transfer run

## MM-WLAuslan auxiliary source

MM-WLAuslan is the current Auslan auxiliary source. The official dataset page provides a Google Drive download link and states that the dataset follows the Creative Commons BY-NC-SA 4.0 license:

- Dataset page: <https://uq-cvlab.github.io/MM-WLAuslan-Dataset/>
- Download page: <https://uq-cvlab.github.io/MM-WLAuslan-Dataset/docs/en/dataset-download>

The current local integration uses the official Train/Valid label JSON files and `Annotation/Pose/Train & Valid/pose_cam1.pkl`. It converts the official Halpe-136 pose arrays into the local 315-D MediaPipe-like sequence layout.

## Clean-start workflow

Initialize a fresh workspace:

```bash
cd nzsl_recognition
.venv\Scripts\python.exe -m starter_nzsl.scripts.init_verified_workspace
```

After you have verified NZSL clips and placed them in `starter_nzsl/data/raw/`, run:

```bash
.venv\Scripts\python.exe -m src.extract_keypoints ^
  --raw_dir starter_nzsl/data/raw ^
  --out_dir starter_nzsl/data/processed ^
  --num_frames 60

.venv\Scripts\python.exe -m src.preprocess ^
  --processed_dir starter_nzsl/data/processed ^
  --out_dir starter_nzsl/models
```

Train a baseline:

```bash
.venv\Scripts\python.exe -m src.train_lstm ^
  --processed_dir starter_nzsl/data/processed ^
  --out_dir starter_nzsl/models ^
  --epochs 30 ^
  --batch_size 16 ^
  --augment ^
  --temporal_crop
```

Prepare the MM-WLAuslan auxiliary pose subset after downloading the official annotation files and `pose_cam1.pkl`:

```bash
.venv\Scripts\python.exe -m starter_nzsl.scripts.prepare_mm_wlauslan_aux ^
  --annotations_dir aux_auslan/mm_wlauslan/annotations ^
  --pose_pkl aux_auslan/mm_wlauslan/pose/pose_train_valid_cam1.pkl ^
  --selected_glosses starter_nzsl/config/mm_wlauslan_selected_glosses.csv ^
  --out_dir aux_auslan/data/processed ^
  --num_frames 60
```

Transfer experiment with the prepared Auslan auxiliary subset:

```bash
.venv\Scripts\python.exe -m src.train_transfer_lstm ^
  --target_processed_dir starter_nzsl/data/processed ^
  --target_out_dir starter_nzsl/models ^
  --aux_processed_dir aux_auslan/data/processed ^
  --mapping_csv starter_nzsl/config/mm_wlauslan_selected_glosses.csv ^
  --out_dir starter_nzsl/models_transfer_auslan ^
  --pretrain_epochs 10 ^
  --finetune_epochs 20 ^
  --batch_size 16 ^
  --augment ^
  --temporal_crop
```

Notes for auxiliary training:

- Auxiliary labels are filtered through the mapping CSV and remapped into the NZSL label space.
- `recommended_use=pretrain` rows are used for transfer learning.
- `reference_only` and `do_not_use` rows are excluded from training.
- Final validation and test remain NZSL-only.

Current transfer result:

- Auxiliary source: MM-WLAuslan pose, 16 mapped labels, 224 auxiliary samples
- NZSL target set: 36 labels, 131 samples
- Report: [`reports_transfer_auslan/metrics.json`](./reports_transfer_auslan/metrics.json)
- Accuracy: `0.0833`
- Macro F1: `0.0439`

This is below the current NZSL-only baseline, so it should be treated as an experimental transfer baseline rather than an improvement. The likely cause is modality/domain mismatch between MM-WLAuslan pose and NZSL Online MediaPipe keypoints, plus the very small NZSL target set.

Evaluate:

```bash
.venv\Scripts\python.exe -m src.evaluate ^
  --processed_dir starter_nzsl/data/processed ^
  --model_type lstm ^
  --model_path starter_nzsl/models/best_lstm.pt ^
  --reports_dir starter_nzsl/reports
```

Alternative inference and evaluation modes:

```bash
.venv\Scripts\python.exe -m src.predict ^
  --input_path starter_nzsl/data/raw/hello/kia_ora.6351.finalexample1.sp.r480x360.mp4 ^
  --model_type template ^
  --model_path starter_nzsl/models/best_lstm.pt ^
  --processed_dir starter_nzsl/data/processed

.venv\Scripts\python.exe -m src.evaluate ^
  --processed_dir starter_nzsl/data/processed ^
  --model_type hybrid ^
  --hybrid_base_model lstm ^
  --model_path starter_nzsl/models/best_lstm.pt ^
  --reports_dir starter_nzsl/reports_hybrid
```

- `template`: nearest-prototype matching on hand and wrist landmark trajectories
- `hybrid`: blends ML probabilities with template matching and a small rule-based gesture scorer

Run the demo:

```bash
.venv\Scripts\python.exe -m streamlit run starter_nzsl/app/streamlit_app.py
```
