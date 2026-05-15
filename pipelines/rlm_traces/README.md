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

For GLM-5 teacher traces over the `sft_traces` split, start with the 3-example smoke:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/run.py --config pipelines/rlm_traces/examples/generate_glm5_sft_smoke.toml
```

If the traces look usable, resume/generate all 302 examples with:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/run.py --config pipelines/rlm_traces/examples/generate_glm5_sft_full.toml
```

Then export accepted, judged-correct trainable turns to a private Prime SFT dataset:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/export_sft.py \
  --input outputs/rlm_traces/glm5-sft-full/sanjaya_text_depth1_llm_only_v1/successful_records.jsonl \
  --output-dir outputs/rlm_traces/glm5-sft-full/sft_dataset \
  --dataset-id lsteno/rlm-rlvr-glm5-sft
```

Use `--no-push` for local tokenization/export validation only.

## Progress

For the current GPT-5.4 + Vertex Flash-Lite full SFT trace run:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/progress.py
```

For another run:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/progress.py \
  --run-dir outputs/rlm_traces/<run-name> \
  --variant sanjaya_text_depth1_llm_only_v1
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

Set top-level `max_workers > 1` with `worker_backend = "process"` for local REPL trace generation. Rollout workers must be separate OS processes because the local REPL temporarily mutates process-global cwd and stdout/stderr during code execution. The parent process remains the only writer for JSONL/summary outputs. Batched `llm_query_batched` calls still use threads inside each rollout process, so API subcalls remain concurrent.

## Design Notes

- One script, one config, one output folder per run.
- Prompt comparison and trace generation share the same code path.
- No dependency on RL configs or orchestrator configs.
- Prompt comparison is just "run the same slice for each prompt variant and summarize the resulting records".
