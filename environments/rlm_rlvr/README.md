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
- This environment reuses the published `rlms` package for prompt construction, parsing, and REPL execution backends.
- Install with Prime CLI (`prime env install rlm_rlvr -p /home/coder/prime_run/environments`) to ensure dependencies are available.
- Set `OPENROUTER_API_KEY` so the semantic judge can score outputs. For hosted training, add it via `env_file`.
- For `repl_backend = "prime"`, set `PRIME_API_KEY` in your environment.
- Local parquet mode is opt-in: pass `data_paths` explicitly. If `eval_data_paths` is omitted, eval defaults to a deterministic 10% holdout from `data_paths`.

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
| `efficiency_penalty_coef` | `float` | `0.02` | Legacy arg kept for config compatibility; it no longer changes the binary reward |
| `inference_mode` | `str` | `"hosted"` | Inference routing mode (`hosted` or `local`) |
| `inference_base_url` | `str \| null` | `null` | Override OpenAI-compatible inference endpoint |
| `inference_api_key` | `str \| null` | `null` | Override API key for inference endpoint |
| `judge_model` | `str` | `"z-ai/glm-4.7-flash"` | OpenRouter model used for binary semantic judging |
| `judge_base_url` | `str` | `"https://openrouter.ai/api/v1"` | Judge provider base URL |
| `judge_api_key_var` | `str` | `"OPENROUTER_API_KEY"` | Environment variable that stores the judge API key |
| `judge_http_referer` | `str \| null` | `null` | Optional OpenRouter `HTTP-Referer` header |
| `judge_app_title` | `str \| null` | `null` | Optional OpenRouter `X-Title` header |
| `repl_backend` | `str` | `"local"` | RLM REPL backend (`local`, `prime`, `docker`, `modal`, `daytona`, `e2b`) |
| `repl_backend_kwargs` | `dict \| null` | `null` | Backend-specific kwargs passed to the selected REPL backend |

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
- Use the configs under `/home/coder/prime_run/configs/rlm_rlvr/` with `orchestrator.use_token_client = true` so recursive subcalls record exact token IDs and logprobs from the same local inference server.
