# Local SFT Pipeline

This folder contains local `prime-rl` SFT configs for warming the RLM base model
before RLVR.

## Files

- `local_8xrtx6000ada_48gb_qwen3_4b_instruct_glm5_traces.toml`: current conservative full-SFT config for `Qwen/Qwen3-4B-Instruct-2507` on 8x RTX 6000 Ada 48GB, using 32k sequence length and `pack_function = "cat"`
- `local_8xrtx6000ada_48gb_qwen3_4b_instruct_curated_v2.toml`: full-SFT config for the reviewed curated v2 multi-turn conversation export, aligned to the RL base model and using assistant-only loss masking
- `local_8xa100_80gb_qwen3_4b_instruct_curated_v3_per_root_turn.toml`: paper-style per-root-turn full-SFT config for `Qwen/Qwen3-4B-Instruct-2507` on 8x A100 80GB
- `local_8xa100_80gb_qwen3_8b_nonthinking_curated_v3_per_root_turn.toml`: paper-style per-root-turn full-SFT config for `Qwen/Qwen3-8B` on 8x A100 80GB, using a derived non-thinking chat-template dataset
- `local_h100x8_qwen3_4b.toml`: older 8xH100 SFT config for `Qwen/Qwen3-4B-Thinking-2507`
- `../../scripts/rlm_sft/run_local.sh`: launcher that wraps `uv run sft`
- `../../scripts/rlm_sft/run_curated_v2_with_upload.sh`: launcher that runs the curated v2 SFT config and uploads the latest `weights/step_*` directory to Hugging Face after a successful run

## Dataset Contract

The dataset must be loadable through Hugging Face `datasets.load_dataset(...)`
and expose `prompt` and `completion` columns in the standard Prime SFT format.

For the current RLM trace pipeline, do not train directly on per-turn
`sft_rows.strict.jsonl` files. Prime's SFT loader masks by message role, so
assistant turns inside a per-turn prompt can be trained again. Export one
multi-turn conversation row per strict trace instead:

```text
outputs/rlm_traces/combined-good-sft-traces-v2/sft_dataset_v2_conversations
```

The conversation export uses:

- `prompt`: initial system/context-metadata/task messages only
- `completion`: assistant REPL actions/finals plus masked `user` feedback from
  the environment
- loss mask: assistant messages only; system/user/tool messages stay masked

Plain `llm_query*` subcalls are not separate student completions. Their prompts
and results appear only through assistant code and masked REPL feedback.

Recommended split layout:

- `train`: main SFT data
- `eval`: held-out validation data

The launcher overrides `data.name` and `val.data.name`, so the same config can be reused for different datasets.

For Qwen3-8B experiments, export a derived per-root-turn dataset with
`chat_template_kwargs = {"enable_thinking": false}` on every row. The official
dense 8B checkpoint is the hybrid `Qwen/Qwen3-8B`; its own chat template can
still inject empty thinking markers in supervised assistant history, so the 8B
SFT config intentionally trains the 8B weights with the compatible
`Qwen/Qwen3-4B-Instruct-2507` non-thinking tokenizer/template.

## Curated V2 Export

From the repo root:

```bash
uv run --project environments/rlm_rlvr python pipelines/rlm_traces/export_sft.py \
  --input outputs/rlm_traces/combined-good-sft-traces-v2/curated-v2-reviewed/records.curated.jsonl \
  --output-dir outputs/rlm_traces/combined-good-sft-traces-v2/sft_dataset_v2_conversations \
  --dataset-id lsteno/rlm-rlvr-sft-v2-conversations \
  --format conversation \
  --validate-tokenization \
  --push
```

Use `--no-push` for local inspection. Full tokenization validation is explicit
because it scans every row with the Qwen chat template and fails rows longer
than 32768 tokens.

## Launch

From the repo root:

```bash
./scripts/rlm_sft/run_local.sh <hf-dataset-name> [output-dir] [wandb-run-name]
```

Example:

```bash
./scripts/rlm_sft/run_local.sh lsteno/rlm-rlvr-sft ../outputs/rlm-rlvr-sft-qwen3-4b rlm-rlvr-sft-qwen3-4b
```

For the reviewed curated v2 full-SFT run, start from `prime-rl`:

```bash
cd prime-rl
../scripts/rlm_sft/run_curated_v2_with_upload.sh \
  ../configs/rlm_sft/local_8xrtx6000ada_48gb_qwen3_4b_instruct_curated_v2.toml \
  ../outputs/rlm-rlvr-sft-curated-v2-qwen3-4b-instruct-8xrtx6000ada \
  lsteno/Qwen3-4B-Instruct-2507-RLM-SFT-v2
```

The upload step requires `HF_TOKEN` and only runs after the training command
exits successfully.

## Notes

- The current RL branch uses `Qwen/Qwen3-4B-Instruct-2507`; keep SFT and RL base model aligned unless running an explicit model-family ablation.
- The curated v2 RTX 6000 Ada config uses full SFT, not LoRA, so RL can start from a real warmed-up checkpoint.
- If long traces cause OOM, reduce `data.seq_len`, `data.batch_size`, or `data.micro_batch_size`, then retry.
