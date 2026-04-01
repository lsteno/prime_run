# RLM Trace Pipeline

This folder contains a standalone pipeline for:

- generating raw RLM traces for a model
- comparing multiple `prompt_variant` system prompts on the same slice of data

It is intentionally separate from `rlm_rlvr` training configs and does not depend on the RL training pipeline.

Use it in two modes:

- single prompt variant: generate traces only
- multiple prompt variants: run the same examples for each prompt and compare them

## What It Writes

For each run:

- `comparison.json`: machine-readable prompt comparison summary
- `comparison.md`: compact prompt comparison table
- `<prompt_variant>/records.jsonl`: full rollout records with traces and segments
- `<prompt_variant>/successful_records.jsonl`: records without runtime errors
- `<prompt_variant>/summary.json`: per-prompt aggregate metrics

Each record includes:

- question and acceptable answers
- final answer
- exact-match flag
- optional judge score
- recursion and token metrics
- root steps
- recursive traces
- flattened segments

## Run

Run it from the `rlm_rlvr` project environment so the local env dependencies are available:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/run.py --config pipelines/rlm_traces/examples/compare_prompts_qwen3_4b.toml
```

## Config Shape

The pipeline config is one TOML file with four sections:

- `[model]`: the generation endpoint
- `[dataset]`: dataset id, split, seed, and sample count
- `[rollout]`: prompt variants and runtime settings
- `[judge]`: optional semantic judge endpoint

If `prompt_variants` contains one entry, the run is trace generation.
If `prompt_variants` contains multiple entries, the run is also a prompt-comparison sweep.

The pipeline currently supports only `openai_chat_completions` endpoint aliases from `configs/endpoints.toml`.

## Design Notes

- One script, one config, one output folder per run.
- Prompt comparison and trace generation share the same code path.
- No dependency on RL configs or orchestrator configs.
- Prompt comparison is just "run the same slice for each prompt variant and summarize the resulting records".
