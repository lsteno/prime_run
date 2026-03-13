# rlm_rlvr

### Overview
- **Environment ID**: `rlm_rlvr`
- **Short description**: Prime-native recursive RLVR environment that trains a single local model across root turns and recursive subcalls.
- **Tags**: `rlvr`, `recursive`, `prime-rl`, `multi-turn`

### Datasets
- **Primary dataset(s)**: Shared parquet schema used by `/home/coder/rl_training/data/train.parquet` and dataset-maker outputs.
- **Source links**: local parquet paths passed through `data_paths` / `eval_data_paths`.
- **Split sizes**: train is shuffled first, then a deterministic held-out eval subset is carved out unless explicit eval parquet paths are supplied.

### Task
- **Type**: multi-turn
- **Output format expectations (optional)**: assistant text with optional ```repl``` blocks and a final `FINAL(...)` or `FINAL_VAR(...)` answer.
- **Rubric overview**: terminal correctness minus a small efficiency penalty proportional to recursive subcalls, with monitor metrics for recursion usage and depth.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run rlm_rlvr
```

Configure model and sampling:

```bash
prime eval run rlm_rlvr -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7 -a '{"data_paths": ["/home/coder/rl_training/data/train.parquet"], "eval_fraction": 0.05}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- This environment now reuses the canonical `rlm` package for prompt construction, parsing, and REPL execution.
- If `rlm` is not importable from your environment, install `rlms` or set `RLM_SOURCE_DIR` to a local checkout such as `/home/coder/rlm` before running eval or training.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `data_paths` | `list[str]` | auto-detect | One or more training parquet paths with the shared RLM schema |
| `eval_data_paths` | `list[str] \| null` | `null` | Optional dedicated eval parquet paths |
| `eval_fraction` | `float` | `0.05` | Held-out eval fraction when `eval_data_paths` is omitted |
| `eval_size` | `int \| null` | `null` | Explicit eval subset size after shuffle |
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
| `efficiency_penalty_coef` | `float` | `0.02` | Small penalty multiplied by `num_subcalls` |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Terminal correctness minus efficiency penalty |
| `correctness` | Exact-match terminal correctness before penalty |
| `used_repl` | Fraction of rollouts that executed at least one REPL block |
| `used_recursion` | Fraction of rollouts that invoked `rlm_query(...)` |
| `num_subcalls` | Number of recursive subcalls executed in the rollout |
| `max_depth_reached` | Deepest recursive call depth reached |

### Prime-RL Notes
- The environment emits flattened recursive segments in `rlm_segments` so Prime-RL can train both root turns and recursive subcalls.
- Use the configs under `/home/coder/prime_run/configs/rlm_rlvr/` with `orchestrator.use_token_client = true` so recursive subcalls record exact token IDs and logprobs from the same local inference server.

