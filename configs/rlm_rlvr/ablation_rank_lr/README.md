# LoRA Rank/LR RL Ablation: Depth-1 LLM-Only

Generated configs for the Qwen3-4B RLM RLVR LoRA rank/LR grid at depth 1.

This ablation intentionally trains only the root RLM with plain LLM subcalls:

- `prompt_variant = "sanjaya_text_depth1_llm_only_v1"` so the system prompt does not mention `rlm_query` or child RLMs.
- Runtime `max_depth = 0` because the root call is depth 0 and plain LLM subcalls are recorded as depth 1.
- Any accidental `rlm_query*` call is blocked from spawning a recursive child and is downgraded by the runtime to a plain LLM call.

## Matrix

| LoRA rank | LoRA alpha | Learning rates |
|---:|---:|---|
| 4 | 8 | 5e-7, 1e-5, 1e-4 |
| 16 | 32 | 5e-7, 1e-5, 1e-4 |
| 64 | 128 | 5e-7, 1e-5, 1e-4 |

Every config runs for 150 steps from `Qwen/Qwen3-4B-Instruct-2507`, with the Sanjaya depth-1 LLM-only prompt, Vertex Gemini subcalls/judge, and adaptive correct-only cost penalty.

## Hardware Shape

The generated configs use the active 8-GPU shape:

- 8 GPUs total.
- 4 GPUs for vLLM rollout inference.
- 4 GPUs for LoRA trainer.
- Trainer `cp = 4`, `tp = 1`, `dp_replicate = 1`.
- vLLM `dp = 4`, `tp = 1`.
- Trainer/orchestrator `seq_len = 65536`, vLLM `max_model_len = 65536`.
- Activation checkpointing uses `freq = 2`, checkpointing every other layer. This is a conservative middle point between full checkpointing (`freq = 1`) and no checkpointing, keeping OOM risk low while reducing recompute overhead.
- `max_async_level = 1` and `max_off_policy_steps = 2`, keeping the rollout pipeline overlapped while discarding rollouts that drift too far from the trainer policy.

Use these on 8x A100 80GB, 8x A100 40GB, H100/H200, or comparable 8-GPU nodes. The directory name says `8xa10080` because that is the preferred pod target, but the config is intentionally the same logical 8-GPU shape as the current working 8xA100 budgeted config.

## Rank 4 Note

Prime-RL's dense Qwen LoRA layer falls back to non-grouped LoRA matmuls when rank is not divisible by 8. Rank 4 is still a valid true-rank ablation, but may be slower than rank 16/64.

## Running

From the repo root on the pod:

```bash
DRY_RUN=1 scripts/rlm_rlvr/run_rank_lr_ablation_grid.sh
```

Then run selected indexes:

```bash
START_INDEX=1 END_INDEX=3 scripts/rlm_rlvr/run_rank_lr_ablation_grid.sh
```

The launcher runs configs sequentially and writes launcher logs to:

```text
outputs/rlm_rank_lr_ablation_launcher_logs/
```

To run a single config directly:

```bash
cd prime-rl
uv run rl @ ../configs/rlm_rlvr/ablation_rank_lr/qwen3_4b_instruct_sanjaya_depth1_llmonly_r064_a128_lr1e-5_s150_8xa10080.toml
```

## Summary

After runs and standardized evals are available:

```bash
uv run python scripts/rlm_rlvr/summarize_rank_lr_ablation.py \
  --manifest configs/rlm_rlvr/ablation_rank_lr/manifest.csv \
  --eval-root outputs/rlm_rank_lr_ablation_evals \
  --out outputs/rlm_rank_lr_ablation_summary/summary.csv
```

## Regeneration

```bash
uv run python scripts/rlm_rlvr/generate_rank_lr_ablation_configs.py
```
