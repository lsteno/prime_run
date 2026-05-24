# RLM RLVR Run Report

Run: `qwen3-4b-instruct-sanjaya-medium-4xa100-80gb-fastseq-budgeted`

Trainer W&B: `msxmkp3i`
Orchestrator W&B: `27b6050r`

Local analysis copy: `outputs/remote_analysis/rlm-rlvr-qwen3-4b-instruct-sanjaya-medium-4xa100-80gb-fastseq-budgeted`

## Executive Summary

The run completed cleanly: 100 orchestrator steps, 100 trainer steps, final checkpoint and final eval. It trained 6,400 rollout samples across about 31.7M total tokens.

The training loop was stable. There is no sign of optimizer divergence: mismatch KL stayed around 0.0014-0.0038, grad norm stayed mostly around 0.07-0.12 with one spike to 0.1514, peak memory was flat at 35.4 GiB, and entropy did not collapse. Loss is noisy and sometimes slightly negative, which is expected for this RL objective and was not itself a failure signal.

The reward improved, especially late. Mean training reward moved from about 0.516 in steps 0-24 to 0.596 in steps 75-99, and 0.634 over steps 90-99. Final training step reward was 0.7802. Raw correctness shows the same direction: about 51.7% early, 59.8% in steps 75-99, and 63.6% in steps 90-99.

Final eval improved versus interval evals but is still modest: Avg@1/Pass@1 was 0.3594 on 64 eval examples. Interval evals were roughly 0.25 at step 25, 0.265 at step 50, 0.344 at step 75, and 0.359 final. So there is real movement, but not enough to call the learned policy strong.

The main behavioral issue is tool/decomposition use. The model used the REPL essentially always, used `llm_query` in only 369/6400 rollouts, and used `rlm_query` in only 2/6400 rollouts. The `llm_query` rollouts were much less accurate than no-subcall rollouts: 31.2% correctness with LLM subcalls versus 55.8% without subcalls. The two RLM-recursive rollouts were both wrong. This run mostly trained direct Python/REPL solving, not recursive decomposition.

## Configuration Context

Key active settings:

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Hardware: 4x A100 80GB pod, with trainer using 2 processes and inference using vLLM data parallelism 2
- Trainer `seq_len`: 65,536
- Orchestrator `batch_size`: 64
- `rollouts_per_example`: 4
- `max_concurrent`: 32
- `oversampling_factor`: 1.0
- `max_off_policy_steps`: 8
- `rollout_timeout_seconds`: 1000
- `repl_timeout_seconds`: 300
- `repl_fast_timeout_seconds`: 60
- Subcall budget enabled: 40 total, 40 batched
- Efficiency penalty coefficient: 0.0001
- Reward behavior: incorrect/no-answer -> 0; correct -> `max(0, 1 - efficiency_penalty)`

The 65k sequence length was enough for the actual trainable segments in this run. The maximum trainable segment observed was 12,611 tokens, p99 was 4,230 tokens, and p95 was 2,752 tokens. The high prompt-token counts mostly came from subcalls and context carried into tools, not from trainable RLM completion tokens.

## Runtime And Throughput

The orchestrator runtime was about 33,214s, or 9.23 hours. The sum of per-step orchestrator times was 32,705s. Trainer per-step time summed to 33,837s.

Trainer step-time summary:

- Mean: 338.4s
- Median: 272.6s
- p90: 536.2s
- p95: 669.0s
- Max: 1624.1s at step 23
- Last 10 mean: 245.9s

Orchestrator step-time summary:

- Mean: 327.0s
- Median: 277.6s
- p90: 522.2s
- p95: 676.1s
- Max: 1836.0s at step 23
- Last 10 mean: 225.6s

The run got faster late. The last 10 trainer steps averaged 245.9s with about 4,755 tokens/s and 68% MFU. That is much better than the early run, and confirms the shorter 65k sequence length plus lower context parallel pressure was a good change. Step 23 was the clear runtime outlier due to rollout timeouts and rescheduling.

The orchestrator became trainer-limited near the end. It repeatedly paused waiting for checkpoint availability once it was more than 4 steps ahead. That means inference/rollout generation was no longer the dominant bottleneck in the final phase; training/backprop and checkpoint cadence were.

## Reward Progression

Training reward and correctness by quarter:

| Steps | Rollouts | Mean reward | Correctness | Zero reward frac |
|---|---:|---:|---:|---:|
| 0-24 | 1600 | 0.5160 | 0.5169 | 0.4831 |
| 25-49 | 1600 | 0.5254 | 0.5262 | 0.4738 |
| 50-74 | 1600 | 0.5329 | 0.5337 | 0.4662 |
| 75-99 | 1600 | 0.5962 | 0.5975 | 0.4025 |
| 90-99 | 640 | 0.6344 | 0.6359 | 0.3641 |

This is a real upward trend, though with noisy per-step rewards. The first three quarters were mostly flat around 0.52-0.53; the late run moved up to about 0.60, with the last 10 steps around 0.63.

Step-level orchestrator reward:

- Mean: 0.5426
- Median: 0.5306
- Min: 0.3588
- p90: 0.6257
- p95: 0.6709
- Max: 0.7802

Important detail: raw rollout reward is binary-ish after the reward fix. Incorrect/no-answer examples get 0. Correct answers get just under 1 because the efficiency penalty is small. So training reward is almost exactly an accuracy proxy now.

Global reward distribution across all rollouts:

- Mean: 0.5426
- Median: 0.9967
- p90: 0.9991
- Max: 0.9997
- Min: 0.0

The median being near 1 while the mean is about 0.54 is expected from a mostly 0/1 reward distribution with a slight correct-answer penalty.

## Eval Progression

Interval/final evals from orchestrator logs:

| Eval point | Avg@1 | Pass@1 | No-response | Mean completion length | Truncated |
|---|---:|---:|---:|---:|---:|
| Step 25 | 0.2500 | n/a | 0.0% | 4265.78 | 3.1% |
| Step 50 | 0.2651 | n/a | 0.0% | 5065.31 | 3.1% |
| Step 75 | 0.3438 | 0.3438 | 0.0% | 3899.08 | 0.0% |
| Final | 0.3594 | 0.3594 | 0.0% | 4504.00 | 6.2% |

The eval curve improved, but only by about +0.11 absolute from step 25 to final. That suggests the run learned something transferable, but not a large behavior shift.

The final eval had higher truncation than step 75, with max completion length 21,714. That is worth watching, but it did not cause no-response failures.

## Trainer Stability

Trainer summary:

| Metric | Mean | Min | Median | p95 | Max | Last 10 mean |
|---|---:|---:|---:|---:|---:|---:|
| Loss | 0.0076 | -0.0024 | 0.0071 | 0.0154 | 0.0211 | 0.0082 |
| Entropy | 0.1866 | 0.1561 | 0.1870 | 0.2042 | 0.2133 | 0.1795 |
| Mismatch KL | 0.0017 | 0.0014 | 0.0017 | 0.0022 | 0.0038 | 0.0017 |
| Grad norm | 0.0906 | 0.0690 | 0.0885 | 0.1081 | 0.1514 | 0.0927 |
| Throughput | 3441.9 | 0.0 | 3413.0 | 4760.1 | 4787.0 | 4754.8 |
| MFU | 49.3% | 0.0% | 48.8% | 68.1% | 68.5% | 68.1% |
| Peak mem | 35.4 GiB | 35.4 | 35.4 | 35.4 | 35.4 | 35.4 |

No divergence signs:

- KL stayed very small and flat.
- Grad norm stayed stable.
- Peak memory stayed flat.
- Entropy was noisy but did not collapse.
- Loss scale was small and did not blow up.

The only standout trainer event was the long step 23, caused by rollout delays rather than optimizer instability.

## Difficulty Filtering And Effective Groups

The run used `rollouts_per_example=4` and produced about 16 prompt groups per step. Across the whole run there were 1,596 groups.

Group outcome by quarter:

| Steps | Groups | Mixed groups | All-correct | All-wrong | Mixed fraction |
|---|---:|---:|---:|---:|---:|
| 0-24 | 400 | 355 | 45 | 0 | 88.8% |
| 25-49 | 399 | 343 | 56 | 0 | 86.0% |
| 50-74 | 400 | 350 | 50 | 0 | 87.5% |
| 75-99 | 397 | 328 | 69 | 0 | 82.6% |
| 90-99 | 159 | 126 | 33 | 0 | 79.2% |

This is healthy for GRPO-style learning: most groups had within-group reward variance, so they produced useful preference signal. There were no all-wrong groups, which likely reflects the difficulty filter/buffer selecting solvable examples. All-correct groups increased late, which is expected as the policy improves but also means the batch becomes slightly less informative over time.

Recommendation: keep difficulty filtering on, but monitor mixed-group fraction. If all-correct groups rise above roughly 30-40%, raise difficulty or refresh the sampling distribution.

## Token And Length Behavior

Across all rollouts:

| Metric | Mean | Median | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|
| Prompt tokens | 20,013 | 11,581 | 43,761 | 107,051 | 1,394,933 |
| Completion tokens | 1,131 | 765 | 2,864 | 4,584 | 40,658 |
| Total tokens | 21,144 | 12,375 | 46,552 | 108,094 | 1,396,296 |
| Trainable tokens | 1,087 | 761 | 2,752 | 4,230 | 12,611 |
| Turns | 4.25 | 4 | 9 | 9 | 9 |
| Subcalls | 0.425 | 0 | 1 | 8 | 40 |

The prompt-token explosion remains real but rare. p99 prompt tokens were about 107k, and the maximum was 1.39M. These are mostly subcall-driven. The trainable tokens stayed moderate, so the trainer did not need the large old 262k sequence length. The current 65k sequence length looks conservative enough for training on this dataset.

Top prompt explosions included:

- Step 91, `frames-0119`: 1,394,933 prompt tokens, 31 LLM subcalls, correct.
- Step 56, `oolong-000071`: 1,027,566 prompt tokens, 30 LLM subcalls, incorrect.
- Step 48, unknown source: 978,791 prompt tokens, 28 LLM subcalls, incorrect.
- Step 31, `frames-0766`: 873,976 prompt tokens, 40 LLM subcalls, incorrect.

The key point: large prompt-token use is not consistently bad. Some prompt-heavy traces were correct, but on average subcall-heavy traces underperformed and cost much more wall time/tokens.

## Tool Usage And Decomposition

Global tool usage:

- REPL used: 6,399/6,400 rollouts
- Any LLM subcall: 369/6,400 rollouts, 5.8%
- Any RLM subcall: 2/6,400 rollouts, 0.03%
- Mean subcalls per rollout: 0.425
- Max subcalls in a rollout: 40

Outcome by usage:

| Usage | n | Correctness | Mean reward | Zero reward frac | Mean prompt tokens | Mean generation seconds |
|---|---:|---:|---:|---:|---:|---:|
| No subcalls | 6029 | 55.8% | 0.5572 | 44.2% | 15,342 | 117.1s |
| Any `llm_query` | 369 | 31.2% | 0.3076 | 68.8% | 95,631 | 201.6s |
| Any `rlm_query` | 2 | 0.0% | 0.0000 | 100.0% | 149,506 | 657.3s |
| REPL | 6399 | 54.4% | 0.5427 | 45.6% | 20,016 | 122.1s |

This is the strongest behavioral finding. The model mostly avoided recursive decomposition. When it used `llm_query`, those traces were substantially less accurate and more expensive than direct REPL-only traces. When it used `rlm_query`, both observed cases failed.

LLM usage by quarter:

| Steps | LLM subcall rollouts | LLM correctness | RLM subcall rollouts | RLM correctness |
|---|---:|---:|---:|---:|
| 0-24 | 91 | 30.8% | 1 | 0.0% |
| 25-49 | 98 | 25.5% | 0 | n/a |
| 50-74 | 99 | 34.3% | 0 | n/a |
| 75-99 | 81 | 34.6% | 1 | 0.0% |
| 90-99 | 27 | 40.7% | 0 | n/a |

LLM subcall use declined slightly late, while correctness of those traces improved. Still, direct solving remained the dominant and better-performing behavior.

Interpretation: the current reward/data does not yet make delegation a consistently rewarded strategy. The model can get enough reward by inspecting context and doing direct Python searches. Since subcalls are risky and often expensive, the policy has no clear incentive to explore them.

## Dataset And Task Breakdown

By source:

| Source | n | Correctness | Reward | LLM use | RLM use | Mean prompt tokens |
|---|---:|---:|---:|---:|---:|---:|
| oolong | 5212 | 57.9% | 0.5778 | 2.6% | 0.02% | 14,150 |
| frames | 434 | 41.5% | 0.4114 | 41.5% | 0.23% | 59,767 |
| unknown/None | 406 | 41.6% | 0.4143 | 13.6% | 0.0% | 51,773 |
| longcodeu | 348 | 33.0% | 0.3298 | 0.0% | 0.0% | 21,204 |

By task group:

| Group | n | Correctness | Reward | LLM use | RLM use | Mean prompt tokens |
|---|---:|---:|---:|---:|---:|---:|
| counting | 1863 | 67.5% | 0.6740 | 0.5% | 0.0% | 10,629 |
| user | 1862 | 54.9% | 0.5482 | 3.8% | 0.05% | 15,488 |
| timeline | 852 | 43.1% | 0.4301 | 4.2% | 0.0% | 18,012 |
| cross | 635 | 58.1% | 0.5803 | 2.7% | 0.0% | 15,371 |
| None | 1188 | 39.1% | 0.3885 | 19.8% | 0.08% | 45,739 |

Counting tasks are currently too easy for direct REPL methods, and they dominate success. Longcodeu/frames/unknown tasks are harder and generate more prompt pressure. If the goal is decomposition, future curricula should increase the share of tasks where direct global search is insufficient and where independent subproblem analysis pays off.

## Timeouts, Slowness, And Failure Cases

Warnings and notable logs:

- Rollout timeout warnings: 254
- Busy event-loop warnings: 8
- Syntax warnings from generated regex strings: 92
- Orchestrator checkpoint pauses: 12
- W&B large-string serialization warnings: 32

Stop conditions:

- `has_final_env_response`: 5,994 rollouts
- `max_turns_reached`: 406 rollouts
- Truncated fraction: 0.64%
- Errors recorded in trace JSON: 0

The timeout mechanism did its job: slow rollouts were killed/re-scheduled and the run completed. Step 23 was the major timeout cluster and took 1836s on the orchestrator side. It had many repeated "Rollout timeout ... after 1000.0s, re-scheduling" warnings.

The slowest traces were not always subcall traces. Several step 23 and step 79 slow rollouts had no subcalls and still ran for 900-990s, usually with ordinary REPL loops or repeated long generations. The worst subcall slow traces were mostly frames examples with many LLM calls.

Examples:

- Step 23, `oolong-001134`: about 993s, no subcalls, one incorrect and one correct rollout.
- Step 79, unknown source: about 981s, no subcalls, max turns, incorrect.
- Step 78, `frames-0468`: about 949s, 36 LLM subcalls, incorrect.
- Step 78, `frames-0673`: about 908s, 4 RLM subcalls, incorrect.

The REPL/event-loop logs were mostly healthy outside bursts. During high pending-task periods, event-loop lag spiked into seconds and occasionally about one minute. This is consistent with high environment concurrency and heavy generated code, not a persistent server failure.

## W&B Logging Notes

The run logged samples every step and final samples at the end. The local final summary shows the samples table accumulated 800 rows, which is consistent with 8 sampled rows per step over 100 steps.

The W&B "reward min" metric can be misleading because some logged reward metrics are aggregates over prompt groups, not raw per-rollout minima. Raw rollout reward definitely contained zeros throughout the run; final-quarter zero-reward fraction was still 40.25%, and steps 90-99 still had 36.41% zero-reward rollouts.

Large W&B serialization warnings came from table cells containing long strings/traces. This is not a training failure, but it can make W&B UI sluggish and artifact upload heavier.

## Main Insights

1. The run is technically healthy.
   The trainer did not diverge, memory was stable, KL/grad norm were controlled, and the run completed all steps and evals.

2. The config changes helped throughput.
   The 65k sequence length was enough for actual trainable segments and dramatically better than the earlier 262k setup. Last-10 trainer MFU reached about 68%.

3. Reward improved, especially late.
   The final 25 steps were meaningfully better than the first 75. The final eval improved to 0.3594 but remains far below a strong result.

4. The model mostly learned direct REPL solving.
   REPL use was essentially universal. LLM subcalls were rare and weak. RLM recursive calls were almost nonexistent.

5. Difficulty filtering is doing useful work.
   Most groups had mixed outcomes and thus useful gradient. There were no all-wrong groups. All-correct groups increased late but remained manageable.

6. Prompt explosion exists but is not the trainer bottleneck.
   Extreme prompt-token traces exist, but trainable tokens were modest. The practical issue is rollout wall time and poor subcall quality, not trainer sequence length.

7. The late bottleneck became training/checkpoint availability.
   The orchestrator repeatedly paused waiting for trainer checkpoints, so simply increasing rollout concurrency is unlikely to help late-run throughput.

## Recommendations For Future Runs

1. Keep `seq_len=65536` for this family of runs.
   The observed max trainable segment was 12.6k, p99 4.2k. There is no evidence this run needed 262k training sequence length.

2. Keep the current nonnegative reward rule.
   Incorrect/no-answer -> 0 and correct -> `max(0, 1 - efficiency_penalty)` produced sane reward curves. The small efficiency penalty coefficient is not dominating correctness.

3. Keep difficulty filtering, but track mixed-group fraction.
   Mixed groups are still high enough. If all-correct groups rise materially, increase difficulty rather than just training longer on the same distribution.

4. Do not increase rollout concurrency from here.
   The orchestrator was already ahead of the trainer late in the run. Higher concurrency would mostly add pressure and probably more prompt/token spikes.

5. If optimizing speed, focus on trainer throughput and checkpoint cadence.
   Last-10 MFU was good but the trainer still dominated wall time. Potential next knobs are checkpoint interval/weight broadcast overhead, trainer parallelism, packing efficiency, and possibly using more trainer GPUs if inference is no longer the bottleneck.

6. Add explicit tool-use diagnostics to W&B.
   Log per-step `used_llm_subcalls`, `used_rlm_subcalls`, `num_subcalls`, subcall correctness, prompt tokens, and slowest rollout summaries directly. The traces have the data, but the dashboard should make decomposition failure obvious during the run.

7. Add a decomposition-focused eval split.
   Current training reward can rise through direct REPL solving. Add a held-out eval set where independent subproblem delegation should help, and track accuracy on that split separately.

8. Consider curriculum/data changes before rewarding recursion directly.
   Naively encouraging recursion will likely recreate long rollouts. Better next step: oversample tasks where direct search is weak and sub-analysis is naturally useful, especially frames/long-context multi-hop tasks.

9. Add soft constraints for subcall quality, not subcall count.
   The model should learn "small relevant excerpt -> child result -> verify", not just "call tools more." A possible reward or filter is to penalize subcall prompt bloat or unverified child claims, while leaving correct concise delegation neutral or slightly favored.

10. Keep `rollout_timeout_seconds=1000`, but investigate repeated timeout examples.
    The timeout was useful and did not kill the run. A lower timeout might save time, but several near-1000s rollouts were correct, so lowering aggressively could remove useful hard examples. A better approach is per-REPL idle/progress timeout plus logging of the exact generated code that caused long stalls.

## Suggested Next Run

For the next run, do not change too many knobs at once. Recommended next experiment:

- Same model and base hyperparameters.
- Same `seq_len=65536`.
- Same timeout values.
- Same reward rule.
- Same or slightly longer run length.
- Dataset/curriculum adjusted toward hard decomposition tasks.
- Add dashboard metrics for LLM/RLM usage and correctness.
- Add a decomposition eval split.
- Keep `max_concurrent=32` unless the trainer setup changes.

Stop/continue criteria:

- Continue if final eval improves and mixed-group fraction remains above about 60%.
- Stop early if `rlm_query` usage remains below 1% and decomposition eval does not improve.
- Stop or revise data if direct REPL accuracy improves while LLM/RLM usage accuracy stays far below no-subcall accuracy.
- Reduce pressure only if event-loop lag returns to sustained multi-minute values or timeout reschedules cluster across many steps.

