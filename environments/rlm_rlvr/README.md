# rlm_rlvr

### Overview
- **Environment ID**: `rlm_rlvr`
- **Short description**: Recursive RLVR environment built as a task-specific wrapper around the canonical `verifiers.envs.experimental.rlm_env.RLMEnv`.
- **Tags**: `rlvr`, `recursive`, `prime-rl`, `sandbox`, `multi-turn`

### Architecture
- Uses the canonical Prime Intellect `RLMEnv` worker lifecycle, Python REPL, and sub-LLM orchestration.
- Preserves the task-specific dataset normalization and binary semantic judge used by the original environment.
- Preserves `rlm_segments` and `rlm_trace` rollout fields so Prime-RL can keep training on recursive subcalls.

### Datasets
- **Primary dataset(s)**: `lsteno/BEEG-agents` from Hugging Face.
- **Split sizes**: uses dataset-provided splits; if the eval split is missing, eval falls back to a deterministic 10% train holdout.

### Task
- **Type**: multi-turn Python REPL.
- **Output protocol**: the model updates `answer["content"]` in the REPL and finalizes with `answer["ready"] = True`.
- **Recursion protocol**: the model can call `rlm_query(prompt, max_depth=None)` from the REPL or from sub-LLM tool calls.
- **Rubric overview**: binary semantic correctness from an LLM judge, with monitor metrics for REPL usage and recursive depth.

### Quickstart
Run an evaluation with defaults:

```bash
prime eval run rlm_rlvr
```

Use the Hugging Face dataset defaults explicitly:

```bash
prime eval run rlm_rlvr -a '{"dataset_id": "lsteno/BEEG-agents", "max_examples": 64, "max_eval_examples": 16}'
```

Use local parquet files:

```bash
prime eval run rlm_rlvr -a '{"data_paths": ["./data/my_dataset/train.parquet"], "eval_data_paths": ["./data/my_dataset/eval.parquet"]}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_id` | `str \| null` | `"lsteno/BEEG-agents"` | Hugging Face dataset identifier |
| `dataset_train_split` | `str` | `"train"` | Training split name for Hugging Face loading |
| `dataset_eval_split` | `str` | `"eval"` | Eval split name; falls back to deterministic holdout if absent |
| `dataset_config` | `str \| null` | `null` | Optional Hugging Face dataset config |
| `dataset_revision` | `str \| null` | `null` | Optional Hugging Face revision pin |
| `data_paths` | `list[str] \| null` | `null` | Optional explicit training parquet paths |
| `eval_data_paths` | `list[str] \| null` | `null` | Optional explicit eval parquet paths |
| `seed` | `int` | `42` | Dataset shuffle seed |
| `max_examples` | `int` | `-1` | Limit training examples after splitting |
| `max_eval_examples` | `int` | `-1` | Limit eval examples |
| `max_iterations` | `int` | `4` | Root rollout turn budget and sub-LLM tool-call turn budget |
| `max_depth` | `int` | `2` | Maximum recursive `rlm_query(...)` depth |
| `turn_max_tokens` | `int` | `192` | Default root `max_tokens` if rollout sampling args do not provide one |
| `subcall_max_tokens` | `int` | `128` | Hard cap applied to recursive sub-LLM `max_tokens` |
| `temperature` | `float` | `1.0` | Default sampling temperature if rollout sampling args do not provide one |
| `top_p` | `float` | `1.0` | Default nucleus sampling value if rollout sampling args do not provide one |
| `judge_model` | `str` | `"z-ai/glm-4.7-flash"` | OpenRouter model used for binary semantic judging |
| `judge_base_url` | `str` | `"https://openrouter.ai/api/v1"` | Judge provider base URL |
| `judge_api_key_var` | `str` | `"OPENROUTER_API_KEY"` | Environment variable that stores the judge API key |
| `judge_http_referer` | `str \| null` | `null` | Optional OpenRouter `HTTP-Referer` header |
| `judge_app_title` | `str \| null` | `null` | Optional OpenRouter `X-Title` header |
| `code_execution_timeout` | `int` | `120` | Timeout for canonical RLMEnv code execution |
| `execution_output_char_limit` | `int` | `4000` | Maximum REPL output returned to the model |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Binary semantic correctness from the judge (`0` or `1`) |
| `correctness` | Same binary judge score exposed as a metric |
| `judge_score` | Binary judge score for observability |
| `used_repl` | Fraction of rollouts that executed the Python REPL |
| `used_recursion` | Fraction of rollouts that invoked `rlm_query(...)` |
| `num_subcalls` | Number of recursive `rlm_query(...)` calls executed in the rollout |
| `max_depth_reached` | Deepest recursive call depth reached |

### Prime-RL Notes
- The environment still emits flattened `rlm_segments` so Prime-RL can train on root turns and recursive subcalls.
- `rlm_trace` is retained as a simplified canonical recursive call summary.
- For Prime Intellect on-demand pods, start with `/home/coder/prime_run/configs/rlm_rlvr/ondemand_smoke_qwen3_4b.toml`.
