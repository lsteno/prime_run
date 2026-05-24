# Full-FT RLM RLVR Pilot: Depth-1 LLM-Only

This directory contains the full-parameter RLVR pilot config for `Qwen/Qwen3-4B-Instruct-2507`.

The config is derived from the r64 LoRA depth-1 LLM-only runs, but removes LoRA and uses full-model NCCL weight broadcast so each policy update can sync the whole 4B model to vLLM without writing full filesystem checkpoints every step.

## Config

```text
qwen3_4b_instruct_sanjaya_depth1_llmonly_fullft_lr1e-6_s150_8xa10080_bal35f40v1.toml
```

Key choices:

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Dataset: `data/beeg_agents_balanced_35_40_25_frames40_v1`
- Prompt: `sanjaya_text_depth1_llm_only_v1`
- Runtime depth: `max_depth = 0`, so the root policy can use plain `llm_query` subcalls but not recursive child RLM calls.
- Hardware shape: 8 GPUs total, 4 vLLM inference GPUs and 4 trainer GPUs.
- Run length: 150 steps.
- Eval: every 25 steps, 64 examples, 1 rollout/example.
- Trainer: full FT, no `[trainer.model.lora]`.
- LR: `1e-6`, AdamW, constant schedule, `weight_decay = 0.0`, `max_norm = 1.0`.
- KL mismatch coefficient: `kl_tau = 1e-3`, matching the LoRA baseline for this first pilot.
- Sequence shape: `seq_len = 49152`, `cp = 2`, `tp = 1`, `dp_replicate = 1`.
- Dtypes: `optimization_dtype = "bfloat16"`, `reduce_dtype = "bfloat16"`.
- Activation checkpointing: `[trainer.model.ac] freq = 2`.
- Weight sync: top-level `[weight_broadcast] type = "nccl"`.
- Inference API servers: `api_server_count = 4`, matching vLLM `dp = 4`; this is required for the full-model NCCL receiver ranks to be distinct.
- Async: `max_async_level = 1`, `max_off_policy_steps = 2`, required by Prime-RL's NCCL full-weight broadcast validation.

## Reward And Environment

The run keeps the stable infrastructure from the LoRA runs:

- 32 train env workers, 16 eval env workers.
- Background group scoring.
- Attempt logging.
- Worker timeout/restart machinery.
- `drop_group_on_first_timeout = false`.
- Gemini Flash-Lite plain subcalls through Vertex.
- Gemini Flash judge through Vertex.
- Finalization penalty enabled.
- Adaptive efficiency penalty enabled with `beta_max = 0.15`, `gamma = 1.0`, and `solve_rate_floor = 0.25`.

Unlike the high-LR rescue runs, this pilot uses:

```toml
efficiency_penalty_applies_to = "correct_only"
```

That keeps the full-FT pilot closer to the r64 baseline and avoids adding negative cost shaping as another confound.

## Running

From the repo root on the pod:

```bash
cd prime-rl
uv run rl @ ../configs/rlm_rlvr/full_ft/qwen3_4b_instruct_sanjaya_depth1_llmonly_fullft_lr1e-6_s150_8xa10080_bal35f40v1.toml
```

Recommended smoke before committing to the full 150 steps:

- Watch the first 3-5 steps.
- Confirm no trainer OOM.
- Confirm NCCL weight update succeeds after the first trainer checkpoint.
- Confirm vLLM receives full model updates, not LoRA adapter updates.
- Check `time/update_weights`, reward, finalization metrics, `mismatch_kl`, entropy, and timeout/restart rates.

If the pilot is stable but weak, try `lr = 2e-6`. If it is unstable, try `lr = 5e-7` or increase `kl_tau` to `3e-3`.
