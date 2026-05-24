# RLM Trace Pipeline

This folder contains a standalone pipeline for:

- generating raw RLM traces for a model
- comparing multiple `prompt_variant` system prompts on the same slice of data
- generating teacher traces for SFT warmup
- retrying failed teacher traces with a stronger root model
- curating accepted traces into SFT-ready rows

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
- optional curated outputs such as `records.curated.jsonl`, `sft_rows.strict.jsonl`, and `quality_report.md`

Each record includes:

- question and acceptable answers
- final answer
- exact-match flag
- optional judge score
- recursion and token metrics
- root steps
- recursive traces
- flattened segments
- prompt messages for trainable RLM segments when enabled
- plain subcall prompts and responses for auditability

## Run

Run it from the `rlm_rlvr` project environment so the local env dependencies are available:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/run.py --config pipelines/rlm_traces/examples/compare_prompts_qwen3_4b.toml
```

For GLM-5 teacher traces over the `sft_traces` split, start with the 3-example smoke:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/run.py --config pipelines/rlm_traces/examples/generate_glm5_sft_smoke.toml
```

If the traces look usable, resume/generate the original 302-example split with:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/run.py --config pipelines/rlm_traces/examples/generate_glm5_sft_full.toml
```

The stronger current teacher path uses GPT-5.4 as root, Vertex Gemini Flash-Lite
as the plain subcall model, and Vertex Gemini 3 Flash as judge:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/run.py --config pipelines/rlm_traces/examples/generate_prime_gpt54_vertex_flash_lite_sft_full.toml
```

For the expanded `sft_traces` split, generate only missing examples and
automatically retry GPT-5.4 failures with GPT-5.5:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/run_missing_with_retry.py \
  --config pipelines/rlm_traces/examples/generate_prime_gpt54_gpt55_vertex_flash_lite_sft_missing_auto_retry.toml
```

The retry driver treats a primary trace as failed when it has a runtime error,
is not accepted by exact match or judge, or uses no plain LLM subcalls.

Curate the combined good trace set:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/curate_sft_traces.py \
  --input outputs/rlm_traces/combined-good-sft-traces-v1/records.jsonl \
  --output-dir outputs/rlm_traces/combined-good-sft-traces-v1/curated-v1
```

For full-SFT warmup, export the reviewed curated v2 traces as one multi-turn
conversation row per strict trace. Do not train directly on per-turn
`sft_rows.strict.jsonl`: Prime masks by message role, so assistant turns inside
the prompt would be trained again.

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/export_sft.py \
  --input outputs/rlm_traces/combined-good-sft-traces-v2/curated-v2-reviewed/records.curated.jsonl \
  --output-dir outputs/rlm_traces/combined-good-sft-traces-v2/sft_dataset_v2_conversations \
  --dataset-id lsteno/rlm-rlvr-sft-v2-conversations \
  --format conversation \
  --validate-tokenization \
  --push
```

The conversation export keeps REPL feedback as `user` messages so Prime masks
tool/environment outputs. Plain `llm_query*` subcalls are not exported as
separate completions.

For paper-style SFT, export one row per root RLM decision with full history in
the prompt and only the next assistant action in the completion:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/export_sft.py \
  --input outputs/rlm_traces/combined-good-sft-traces-v2/curated-v2-reviewed/records.curated.jsonl \
  --output-dir outputs/rlm_traces/combined-good-sft-traces-v2/sft_dataset_v3_per_root_turn \
  --dataset-id lsteno/rlm-rlvr-sft-v3-per-root-turn \
  --format per_root_turn \
  --validate-tokenization \
  --push
```

For Qwen3-8B, use the same per-root-turn data but add explicit non-thinking
chat-template kwargs:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/export_sft.py \
  --input outputs/rlm_traces/combined-good-sft-traces-v2/curated-v2-reviewed/records.curated.jsonl \
  --output-dir outputs/rlm_traces/combined-good-sft-traces-v2/sft_dataset_v3_per_root_turn_qwen3_8b_nonthinking \
  --dataset-id lsteno/rlm-rlvr-sft-v3-per-root-turn-qwen3-8b-nonthinking \
  --format per_root_turn \
  --tokenizer Qwen/Qwen3-8B \
  --chat-template-kwargs-json '{"enable_thinking": false}' \
  --validate-tokenization \
  --push
```

The export script can still export accepted trainable turns from raw
`successful_records.jsonl` files when explicitly requested with
`--format per_turn`:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/export_sft.py \
  --input outputs/rlm_traces/glm5-sft-full/sanjaya_text_depth1_llm_only_v1/successful_records.jsonl \
  --output-dir outputs/rlm_traces/glm5-sft-full/sft_dataset \
  --dataset-id lsteno/rlm-rlvr-glm5-sft \
  --format per_turn
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

The missing-run driver and `run.py` both print live progress. `progress.py`
is useful when the process is detached or running on a pod.

## Config Shape

The pipeline config is one TOML file with four sections:

- `[model]`: the generation endpoint
- `[dataset]`: dataset id, split, seed, and sample count
- `[rollout]`: prompt variants and runtime settings
- `[judge]`: optional semantic judge endpoint

If `prompt_variants` contains one entry, the run is trace generation.
If `prompt_variants` contains multiple entries, the run is also a prompt-comparison sweep.

The pipeline currently supports only `openai_chat_completions` endpoint aliases from `configs/endpoints.toml`.

Set top-level `max_workers > 1` with `worker_backend = "process"` for local REPL trace generation. Rollout workers must be separate OS processes because the local REPL temporarily mutates process-global cwd and stdout/stderr during code execution. The parent process remains the only writer for JSONL/summary outputs. Batched `llm_query_batched` calls still use threads inside each rollout process, so API subcalls remain concurrent. Batched recursive `rlm_query_batched` child calls execute serially by default for local REPL safety; use `recursive_rlm_batch_mode = "thread"` only as an explicit advanced override.

For Vertex-backed subcalls, configs may include:

```toml
[llm_subcall]
provider = "vertex"
model = "gemini-3.1-flash-lite"
vertex_project_env = "GOOGLE_CLOUD_PROJECT"
vertex_location = "global"
thinking_level = "medium"
empty_response_max_attempts = 3
```

Empty Vertex subcall responses are retried rather than silently accepted as
useful evidence.

## Design Notes

- One script, one config, one output folder per run.
- Prompt comparison and trace generation share the same code path.
- No dependency on RL configs or orchestrator configs.
- Prompt comparison is just "run the same slice for each prompt variant and summarize the resulting records".
- Raw records are intentionally verbose. They are the audit log for prompts,
  model outputs, REPL feedback, subcall prompts, subcall responses, token counts,
  and final answers.
- SFT export should include only trainable RLM-owned turns. Plain LLM subcalls
  are trace/cost data, not student completions.
