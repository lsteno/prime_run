# Local SFT Pipeline

This pipeline uses the local `prime-rl` SFT runner on a single 8xH100 node.

## Files

- `local_h100x8_qwen3_4b.toml`: default 8xH100 SFT config for `Qwen/Qwen3-4B-Thinking-2507`
- `../../scripts/rlm_sft/run_local.sh`: launcher that wraps `uv run sft`

## Dataset Contract

The dataset must be loadable through Hugging Face `datasets.load_dataset(...)` and expose `prompt` and `completion` columns in the standard Prime SFT format.

Recommended split layout:

- `train`: main SFT data
- `eval`: held-out validation data

The launcher overrides `data.name` and `val.data.name`, so the same config can be reused for different datasets.

## Launch

From the repo root:

```bash
./scripts/rlm_sft/run_local.sh <hf-dataset-name> [output-dir] [wandb-run-name]
```

Example:

```bash
./scripts/rlm_sft/run_local.sh lsteno/rlm-rlvr-sft ../outputs/rlm-rlvr-sft-qwen3-4b rlm-rlvr-sft-qwen3-4b
```

## Notes

- The config assumes a single node with 8 H100 GPUs.
- The default model is the same 4B thinking family used by the current RL branch.
- If long traces cause OOM, reduce `data.seq_len` or `data.micro-batch-size`, then retry.
