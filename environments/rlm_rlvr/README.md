# rlm_rlvr

### Overview
- **Environment ID**: `rlm_rlvr`
- **Short description**: Prime-native recursive RLVR environment that trains a single local model on root and recursive RLM turns, while keeping plain LLM subcalls trace-only.
- **Tags**: `rlvr`, `recursive`, `prime-rl`, `multi-turn`

### Datasets
- **Primary dataset(s)**: `lsteno/BEEG-agents` from Hugging Face.
- **Source links**: `dataset_id` / split args (`dataset_train_split`, `dataset_eval_split`).
- **Split sizes**: uses dataset-provided splits; if the eval split is missing, eval falls back to a deterministic 10% train holdout.

### Task
- **Type**: multi-turn
- **Output format expectations (optional)**: assistant text with optional ```repl``` blocks and a final `FINAL(...)` or `FINAL_VAR(...)` answer.
- **Rubric overview**: semantic correctness from an LLM judge, with optional token-cost shaping and monitor metrics for recursion usage and depth.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run rlm_rlvr
```

Configure model and sampling:

```bash
prime eval run rlm_rlvr -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7 -a '{"data_paths": ["./data/my_dataset/train.parquet"], "eval_data_paths": ["./data/my_dataset/eval.parquet"]}'
```

Use the Hugging Face dataset defaults explicitly:

```bash
prime eval run rlm_rlvr -a '{"dataset_id": "lsteno/BEEG-agents", "max_examples": 64, "max_eval_examples": 16}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- This environment reuses the published `rlms` package for prompt construction, parsing, and local REPL execution.
- Install with Prime CLI (`prime env install rlm_rlvr -p /home/coder/prime_run/environments`) to ensure dependencies are available.
- Set `OPENROUTER_API_KEY` so the semantic judge can score outputs. For managed hosted training, add it via `env_file`. For self-managed `prime-rl` runs on Prime Intellect on-demand GPUs, export it directly in the pod shell.
- `repl_backend` currently supports only `"local"`; remote REPL backends are future work.
- The default prompt is `sanjaya_text_v1`. `prompt_variant="default"` is kept as a compatibility alias for the same prompt.
- Local parquet mode is opt-in: pass `data_paths` explicitly. If `eval_data_paths` is omitted, eval defaults to a deterministic 10% holdout from `data_paths`.
- `inference_mode = "hosted"` is for managed hosted training. `inference_mode = "local"` is the standard setting for self-managed `prime-rl` runs on on-demand GPUs.
- In the standard self-managed `prime-rl` path, the launcher handles the local inference base URL and API wiring. You do not need to set `RLM_LOCAL_INFERENCE_BASE_URL` or `RLM_LOCAL_INFERENCE_API_KEY` unless you are overriding the default local server.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_id` | `str \| null` | `"lsteno/BEEG-agents"` | Hugging Face dataset identifier (primary mode) |
| `dataset_train_split` | `str` | `"train"` | Training split name for Hugging Face dataset loading |
| `dataset_eval_split` | `str` | `"eval"` | Evaluation split name; falls back to deterministic holdout if absent |
| `dataset_config` | `str \| null` | `null` | Optional Hugging Face dataset config |
| `dataset_revision` | `str \| null` | `null` | Optional Hugging Face revision/commit pin |
| `data_paths` | `list[str] \| null` | `null` | Optional explicit training parquet paths (local mode) |
| `eval_data_paths` | `list[str] \| null` | `null` | Optional explicit eval parquet paths; if omitted, a 10% holdout is derived from `data_paths` |
| `seed` | `int` | `42` | Dataset shuffle seed |
| `max_examples` | `int` | `-1` | Limit training examples after splitting |
| `max_eval_examples` | `int` | `-1` | Limit eval examples |
| `max_iterations` | `int` | `4` | Recursive reasoning turns per call before forcing a final answer |
| `max_depth` | `int` | `2` | Maximum recursion depth |
| `turn_max_tokens` | `int` | `192` | Max assistant tokens for each recursive reasoning turn |
| `subcall_max_tokens` | `int` | `128` | Max assistant tokens for leaf/plain subcalls |
| `temperature` | `float` | `1.0` | Root and recursive sampling temperature |
| `top_p` | `float` | `1.0` | Root and recursive nucleus sampling |
| `tokenizer_name` | `str \| null` | `null` | Optional tokenizer override; defaults to the rollout model name |
| `prompt_variant` | `str` | `"sanjaya_text_v1"` | System prompt variant. Supported values: `sanjaya_text_v1`, `default` where `default` is a compatibility alias |
| `live_trace_dir` | `str \| null` | `"outputs/rlm_rlvr/live_traces"` | Directory for compact per-sample live traces updated after root and recursive steps. Set to `null` to disable |
| `subcall_prompt_limit_ratio` | `float` | `0.85` | Blocks `llm_query*` and `rlm_query*` prompts whose estimated character size exceeds this fraction of the configured subcall context window, returning a REPL-visible error instead of truncating context |
| `efficiency_penalty_coef` | `float` | `0.02` | Cost-aware shaping coefficient applied only to correct answers. Incorrect/no-answer rollouts receive `0`; correct rollouts receive `max(0, 1 - efficiency_penalty_coef * (rollout_prompt_tokens + rollout_completion_tokens) / 1000)` |
| `inference_mode` | `str` | `"hosted"` | Inference routing mode. Use `hosted` for managed hosted training and `local` for self-managed `prime-rl` on local or on-demand GPUs |
| `inference_base_url` | `str \| null` | `null` | Override the OpenAI-compatible inference endpoint. Usually unset for self-managed `prime-rl`, which wires the local inference server automatically |
| `inference_api_key` | `str \| null` | `null` | Override API key for the inference endpoint. Usually unset for self-managed `prime-rl` local inference |
| `judge_model` | `str` | `"z-ai/glm-5"` | OpenRouter model used for binary semantic judging |
| `judge_base_url` | `str` | `"https://openrouter.ai/api/v1"` | Judge provider base URL |
| `judge_api_key_var` | `str` | `"OPENROUTER_API_KEY"` | Environment variable that stores the judge API key |
| `judge_http_referer` | `str \| null` | `null` | Optional OpenRouter `HTTP-Referer` header |
| `judge_app_title` | `str \| null` | `null` | Optional OpenRouter `X-Title` header |
| `repl_backend` | `str` | `"local"` | RLM REPL backend. Only `local` is currently supported |
| `repl_backend_kwargs` | `dict \| null` | `null` | Reserved for future backend-specific kwargs |
| `repl_timeout_seconds` | `float \| null` | `null` | Optional wall-clock timeout for generated REPL code blocks that call `llm_query*` or `rlm_query*` helpers |
| `repl_fast_timeout_seconds` | `float \| null` | `null` | Optional shorter wall-clock timeout for generated REPL code blocks with no LLM/RLM subcalls |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Zero for incorrect/no-answer rollouts; correct rollouts minus optional token-cost penalty clipped at zero |
| `correctness` | Raw binary judge score before cost shaping |
| `judge_score` | Binary judge score for observability |
| `efficiency_penalty` | Token-cost penalty subtracted from reward |
| `cost_prompt_tokens` | Total prompt tokens consumed across all root turns, recursive turns, and subcalls |
| `cost_completion_tokens` | Total completion tokens consumed across all root turns, recursive turns, and subcalls |
| `cost_total_tokens` | Sum of prompt and completion tokens used for cost shaping |
| `used_repl` | Fraction of rollouts that executed at least one REPL block |
| `used_recursion` | Fraction of rollouts that invoked any `llm_query(...)` or `rlm_query(...)` subcall |
| `used_llm_subcalls` | Fraction of rollouts that used at least one plain `llm_query(...)` subcall |
| `used_rlm_subcalls` | Fraction of rollouts that used at least one recursive `rlm_query(...)` subcall |
| `num_subcalls` | Total number of LLM plus RLM subcalls executed in the rollout |
| `num_llm_subcalls` | Number of plain `llm_query(...)` subcalls executed in the rollout |
| `num_rlm_subcalls` | Number of recursive `rlm_query(...)` subcalls executed in the rollout |
| `max_depth_reached` | Deepest aggregate recursion depth reached, with plain LLM subcalls counted as depth 1 |

### Prime-RL Notes
- The environment emits flattened recursive segments in `rlm_segments` with explicit call provenance. Prime-RL trains only segments marked as RLM-owned turns (`root_turn`, `recursive_turn`, `finalize_turn`) and skips plain `llm_query(...)` subcalls.
- With `prime eval run -s`, completed rollout rows are written incrementally to `environments/rlm_rlvr/outputs/evals/<env>--<model>/<run_id>/results.jsonl`.
- Long in-flight rollouts also update compact live trace files under `outputs/rlm_rlvr/live_traces/<prompt_variant>/<source_id>.json`. These include assistant text, executed code blocks, REPL feedback, final answer state, subcall counters, and compact segment token counts without duplicating full token id arrays.
- Keep `orchestrator.use_token_client = false` for this environment. Recursive rollouts use message-based chat completions; the token-in/token-out endpoint is for linear TITO and prefill flows.
- For local SFT warmup on an 8xH100 node, see `/home/coder/prime_run/configs/rlm_sft/README.md` and `/home/coder/prime_run/configs/rlm_sft/local_h100x8_qwen3_4b.toml`.

### Development

Run the environment tests with uv:

```bash
uv run --project environments/rlm_rlvr --group dev pytest environments/rlm_rlvr/tests -q
```
