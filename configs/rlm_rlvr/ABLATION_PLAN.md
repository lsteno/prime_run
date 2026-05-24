# RLM RLVR Ablation Plan

## Summary

- Standardize the main study on `Qwen/Qwen3-4B-Thinking-2507`.
- Target self-managed local training on rented 4-8 GPU A100/H100 pods.
- Keep the study focused on recursive RLVR, cost-aware reward shaping, recursion depth, and LoRA rank/LR behavior.
- Drop async futures, `D_seq`, and rsLoRA from the study goals.
- Drop GEPA prompt optimization for this phase; standardize on the Sanjaya text RLM prompt.
- End with an eval-only benchmark pass on the original RLM paper benchmark family: CodeQA, BrowseComp-Plus 1K, OOLONG, OOLONG-Pairs, and optionally S-NIAH for length-scaling diagnostics.

# TODO
- ✅ Parallelize llm query batched
- ✅ _raise_if_subcall_prompt_too_large(...) is stupid and slow and _fit_messages_to_prompt_budget(...) cuts away context! Bad! Simplify the guard by using character count estimates and remove _fit_messages_to_prompt_budget(...) to avoid loss of context (just raise error if prompt too big) also raise single error for multiple too long calls (e.g. batched call)
- ✅ Update tracking of recursion to distinguish between llm subcalls and rlm subcalls
- ✅ Only train on root turns
- ✅ Strengthen how we track which subcall generated what so model is rewarded/trained exactly on the context it had at a certain time, this should fix the error we get where it says output is more than seq_len
- ✅ Make sure responses from tools are masked so we don't train on tool/REPL outputs
- ✅ Diagnose why we don't get sub-rlm calls, is it broken or do the models just not use it?
- ✅ Still need to add cost awerness in the reward!!!

## Scope Decisions

- No async research claim:
  - keep recursive batched calls serial and deterministic
  - do not implement futures
  - do not report `D_seq`
  - log wall-clock time as an operational metric only
- No rsLoRA:
  - use standard Prime LoRA scaling
  - set `trainer.model.lora.alpha = 2 * rank` unless a smoke run shows instability
  - interpret rank as a capacity/optimization ablation, not as a scaling-law ablation
- No GEPA prompt optimization:
  - use `sanjaya_text_v1` as the default training prompt
  - keep prompt changes out of the main ablation matrix unless diagnostics show the prompt is the blocker
- Cost awareness remains central:
  - implement active shaped reward before long training
  - use generated tokens across root turns and recursive subcalls as the single cost source
  - do not directly penalize number of subcalls in the first pass

## Environment And Config Changes Needed

- Add env arg `cost_reward_alpha`.
- Keep `prompt_variant` as a compatibility selector, with `sanjaya_text_v1` as the default and `default` as an alias.
- Make the reward formula explicit:
  - `cost_k = total_model_tokens / 1024`
  - `reward_shaped = reward_correctness / (1 + cost_reward_alpha * cost_k)`
  - incorrect or empty-answer rollouts stay at `0`
- Log metrics:
  - `reward_correctness`
  - `reward_shaped`
  - `total_model_tokens`
  - `total_env_tokens`
  - `used_recursion`
  - `num_subcalls`
  - `max_depth_reached`
- Define the main 4B config from the current long/deep 9B shape, changing model-specific and hardware-specific fields:
  - model/tokenizer: `Qwen/Qwen3-4B-Thinking-2507`
  - `inference_mode = "local"`
  - `repl_backend = "local"`
  - `use_token_client = false`
  - `seq_len = 16384`
  - `inference.model.max_model_len = 16384`
  - 4 GPU layout: `num_train_gpus = 2`, `num_infer_gpus = 2`
  - 8 GPU layout: `num_train_gpus = 4`, `num_infer_gpus = 4`

## Run Matrix

### Stage 0: Bring-Up

- Smoke run:
  - 2-5 steps
  - `rank = 16`
  - `alpha = 32`
  - `lr = 1e-6`
  - `max_depth = 2`
  - `cost_reward_alpha = 0.0`
- Calibration run:
  - 20-30 steps
  - trace export enabled
  - same settings as smoke unless memory or throughput requires adjustment
- Success gate:
  - rollouts complete
  - recursive calls work
  - token accounting is nonzero
  - reward has variance
  - checkpoint and trace export work

Estimated resources:

- 4 GPUs: 1-4 wall-hours.
- 8 GPUs: 1-3 wall-hours.
- Use measured step time from this stage to update all later estimates.

### Stage 1: Sanjaya Prompt Calibration

- Eval only, no RL updates.
- Fixed prompt: `sanjaya_text_v1`.
- Depth caps:
  - `max_depth = 1`
  - `max_depth = 2`
- Run 30-50 eval samples per depth setting.
- Select the depth setting for training by correctness first, then generated-token cost, then recursion/subcall behavior.

Estimated resources:

- 4 GPUs: 3-8 wall-hours total.
- 8 GPUs: 2-5 wall-hours total.

### Stage 2: Cost Reward Pilot

- Train short runs with fixed LoRA.
- Fixed settings:
  - winning prompt
  - `rank = 16`
  - `alpha = 32`
  - `lr = 1e-6`
  - `max_depth = 2`
  - 75-100 steps
  - `rollouts_per_example = 4` if stable; use 2 only for pilot troubleshooting
- Cost alphas:
  - `0.0`
  - `0.01`
  - `0.02`
  - `0.05`

Selection rule:

- Pick the smallest nonzero `cost_reward_alpha` that reduces generated-token cost without a material correctness drop.
- If all nonzero alphas hurt correctness materially, keep `0.0` for the main rank/LR study and report cost shaping as negative evidence.

Estimated resources per 100-step run:

- 4x H100: 5-12 wall-hours.
- 8x H100: 3-8 wall-hours.
- 4x A100: 8-20 wall-hours.
- 8x A100: 5-14 wall-hours.

### Stage 3: LoRA Rank/LR Ablation

- Use the winning prompt and cost alpha.
- Primary grid:
  - ranks: `{4, 16, 64}`
  - learning rates: `{5e-7, 1e-6, 5e-6}`
  - LoRA alpha: `{8, 32, 128}` respectively
  - 200 steps per run
- Optional high-rank confirmation:
  - `rank = 128`
  - `alpha = 256`
  - best LR only
  - run only if `rank = 64` beats or approaches the best lower-rank run

Selection rule:

- Primary metric: validation correctness.
- Tie-breakers: shaped reward, total generated tokens, recursion efficiency.
- Report convergence speed separately from final quality.

Estimated resources for the 9-run primary grid:

- 4x H100: roughly 180-430 GPU-hours.
- 8x H100: roughly 190-510 GPU-hours, with lower calendar time.
- 4x A100: roughly 290-720 GPU-hours.
- 8x A100: roughly 320-900 GPU-hours.

### Stage 4: Depth Ablation

- Use the best rank/LR/cost-alpha settings.
- Train:
  - `max_depth = 0`
  - `max_depth = 1`
  - `max_depth = 2`
- Optional:
  - `max_depth = 3` only if depth 2 uses recursion productively.
- Length:
  - 150-200 steps per depth setting.

Interpretation:

- `max_depth = 0`: REPL and plain LLM calls only.
- `max_depth = 1`: closest to the original shallow recursive setting.
- `max_depth >= 2`: tests whether learned recursive policies benefit from deeper child RLMs.

Estimated resources:

- 2-3 extra training runs beyond the already selected main config.
- Similar per-run cost to Stage 3.

## Final Benchmark Pass

- Run eval only with the best trained config.
- Do not train on these benchmark examples.
- Add dataset adapters or parquet conversion for the original RLM paper benchmark family:
  - CodeQA from LongBench-v2
  - BrowseComp-Plus 1K
  - OOLONG `trec_coarse`
  - OOLONG-Pairs
  - optional S-NIAH length sweep
- First validate each adapter with 25-50 examples.
- Final pass should target the paper's sample sizes where feasible:
  - BrowseComp-Plus uses 150 sampled instances in the paper
  - OOLONG/OOLONG-Pairs/S-NIAH should preserve their native scoring and length settings where available

Compare:

- base Qwen3 4B
- prompt-only RLM with the Sanjaya prompt
- best RL-trained RLM

Report:

- benchmark-native accuracy or F1
- generated-token cost
- `num_subcalls`
- `used_recursion`
- `max_depth_reached`
- wall-clock time

## Test And Validation Gates

- Prompt tests:
  - Sanjaya prompt and `default` alias preserve REPL code-block parsing
  - recursion calls remain documented
  - `FINAL(...)` and `FINAL_VAR(...)` termination remains clear
- Reward tests:
  - exact match still yields `reward_correctness = 1`
  - judge score still fills `reward_correctness`
  - shaped reward matches the formula for multiple `cost_reward_alpha` values
  - incorrect rollouts remain `0`
- Runtime tests:
  - root and recursive generated tokens contribute to `total_model_tokens`
  - local chat completions still use message-based `/chat/completions`
  - serial recursive batching remains deterministic
- Run gates:
  - Stage 0 must pass before prompt tuning
  - Stage 1 prompt winner chosen before cost pilots
  - Stage 2 alpha winner chosen before the rank/LR grid
  - Stage 3 winner chosen before depth ablations
  - final paper-benchmark pass only after the main config is stable

## Assumptions

- Main model is `Qwen/Qwen3-4B-Thinking-2507`.
- Main hardware is rented 4-8 GPU A100/H100 pods.
- Preferred deployment is 4 GPUs with `2 train / 2 infer` or 8 GPUs with `4 train / 4 infer`.
- The study intentionally excludes async futures, `D_seq`, GEPA prompt optimization, and rsLoRA.
- Cost is measured by generated model tokens across root and recursive calls.
- Original RLM paper benchmark adapters are not currently present in the repo and must be added before the final benchmark pass.
