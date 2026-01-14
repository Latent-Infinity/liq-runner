# Pipeline Spec (Runner)

This document defines a minimal, model-agnostic pipeline spec for orchestration. It
is intended to keep liq-runner as a coordinator only.

## Spec schema (JSON)

```json
{
  "name": "ssl_finetune",
  "artifact_root": "artifacts",
  "stages": [
    {
      "name": "ingest",
      "description": "Ingest and normalize raw data",
      "inputs": [],
      "outputs": ["raw_candles"],
      "enabled": true,
      "config": {}
    }
  ]
}
```

## Default stage order

This template is intended for SSL + finetune models (transformers, JEPA, masked
time-series pretraining, and similar families).

- ingest
- features
- datasets
- pretrain
- finetune
- eval_dev
- eval_lockbox
- publish

## Hashing

Use `compute_config_hash(spec)` from `liq.runner.pipeline_spec` to hash the spec
for audit logs and registry metadata.

## Dry-run

Use `dry_run_steps(spec)` to return an ordered list of enabled stages and their
inputs/outputs without executing any stage logic.
