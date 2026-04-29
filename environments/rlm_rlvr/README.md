# rlm_rlvr

### Overview
- **Environment ID**: `rlm_rlvr`
- **Short description**: Prime-native recursive RLVR environment that trains a single local model across root turns and recursive subcalls.
- **Tags**: `rlvr`, `recursive`, `prime-rl`, `multi-turn`

### Datasets
- **Primary dataset(s)**: `lsteno/BEEG-agents` from Hugging Face.
- **Source links**: `dataset_id` / split args (`dataset_train_split`, `dataset_eval_split`).
- **Split sizes**: uses dataset-provided splits; if the eval split is missing, eval falls back to a deterministic 10% train holdout.

### Task
- **Type**: multi-turn
- **Output format expectations (optional)**: assistant text with optional ```repl``` blocks and a final `FINAL(...)` or `FINAL_VAR(...)` answer.
- **Rubric overview**: binary semantic correctness from an LLM judge, with monitor metrics for recursion usage and depth.

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
- Use `prompt_variant` to switch between the default upstream prompt and the local balanced prompt variants for inference bakeoffs.
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
| `prompt_variant` | `str` | `"default"` | System prompt variant. Supported values: `default`, `balanced_v1`, `balanced_v2` |
| `efficiency_penalty_coef` | `float` | `0.02` | Reserved for upcoming cost-aware reward shaping; it does not currently change the binary reward |
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

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Binary semantic correctness from the judge (`0` or `1`) |
| `correctness` | Same binary judge score exposed as a metric |
| `judge_score` | Binary judge score for observability |
| `used_repl` | Fraction of rollouts that executed at least one REPL block |
| `used_recursion` | Fraction of rollouts that invoked `rlm_query(...)` |
| `num_subcalls` | Number of recursive subcalls executed in the rollout |
| `max_depth_reached` | Deepest recursive call depth reached |

### Prime-RL Notes
- The environment emits flattened recursive segments in `rlm_segments` so Prime-RL can train both root turns and recursive subcalls.
- For Prime Intellect on-demand pods, start with `/home/coder/prime_run/configs/rlm_rlvr/ondemand_smoke_qwen3_4b.toml`, which keeps `orchestrator.use_token_client = false` and `inference_mode = "local"` for a lower-risk bring-up path.
- After the pod path is stable, use `/home/coder/prime_run/configs/rlm_rlvr/ondemand_long_deep_qwen35_9b.toml` for the longer Qwen 3.5 9B run.
- Keep `orchestrator.use_token_client = false` for this environment. Recursive rollouts use message-based chat completions; the token-in/token-out endpoint is for linear TITO and prefill flows.
- For local SFT warmup on an 8xH100 node, see `/home/coder/prime_run/configs/rlm_sft/README.md` and `/home/coder/prime_run/configs/rlm_sft/local_h100x8_qwen3_4b.toml`.

### Development

Run the environment tests with uv:

```bash
uv run --project environments/rlm_rlvr --group dev pytest environments/rlm_rlvr/tests -q
```
