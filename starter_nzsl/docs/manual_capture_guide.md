# Manual Capture Guide

This guide is for recording additional **verified NZSL** samples to fill gaps in the starter vocabulary.

## Priority

Use [`data/metadata/manual_capture_manifest.csv`](../data/metadata/manual_capture_manifest.csv) as the source of truth.

Record in this order:

1. `ferry`
2. `taxi`
3. `station`
4. `platform`
5. `ticket`
6. `hello (kia ora)`
7. `thank_you`

## Minimum recording standard

- Record in landscape orientation
- Keep the signer framed from mid-torso to above the head
- Both hands and face must stay visible for the full clip
- Use plain background and even lighting
- Avoid strong backlight, cluttered backgrounds, and motion blur
- Each clip should contain one isolated target sign with a short neutral pause before and after

## Per-clip target

- Duration: `2` to `4` seconds
- One target sign per clip
- Natural tempo preferred
- For each signer, include a few mild natural variations in speed and emphasis

## Diversity target

For each high-priority label:

- At least `2` signers, preferably `3`
- At least `4` clips per signer
- Try to vary:
  - clothing colour
  - recording room
  - small differences in tempo

## Consent

Only record or use clips where the signer has explicitly agreed that the clips can be used for this academic NZSL recognition project.

## Suggested filename pattern

Use:

```text
{label}.{signer_id}.{clip_index}.mp4
```

Examples:

```text
taxi.s01.01.mp4
taxi.s01.02.mp4
hello.s02.01.mp4
```

Place files under:

```text
starter_nzsl/data/raw/{label}/
```

## Metadata

For every recorded clip, add a row to:

[`data/metadata/verified_samples_master.csv`](../data/metadata/verified_samples_master.csv)

Recommended fields:

- `sample_id`
- `label`
- `source_id = manual_capture`
- `license = project-specific`
- `signer_id`
- `local_relpath`
- `verified_nzsl = true`
- `verified_by`
- `verification_notes`

## Immediate goal

The current dataset bottleneck is not pipeline code. It is low per-class sample count.

The fastest improvement path is:

1. record the seven priority labels above
2. rerun keypoint extraction
3. rerun preprocessing, training, and evaluation
