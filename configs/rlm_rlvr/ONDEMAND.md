# RLVR On-Demand Runbook

Use the on-demand configs in this directory for self-managed `prime-rl` runs on Prime Intellect GPU pods. Leave the `hosted_*.toml` files for managed hosted training.

## Configs

- `ondemand_smoke_qwen3_4b.toml`: first bring-up on a single 2-GPU pod
- `ondemand_long_deep_qwen35_9b.toml`: near-parity follow-up to the hosted Qwen 3.5 9B long/deep run

## 1. Pick a 2-GPU secure-cloud pod

```bash
prime availability list --gpu-count 2 --regions united_states --no-group-similar
```

Choose a secure-cloud 2-GPU offer that fits the model and budget. For H100-based bring-up, filtering by `--gpu-type H100_80GB` is the cleanest starting point.

## 2. Create or reuse a persistent disk

```bash
prime disks list
prime disks create --size 500 --name rlm-rlvr-data --id <disk-offer-id>
```

Attach this disk to the pod so checkpoints, caches, and datasets survive pod termination. If you already have a reusable training disk, skip creation and reuse its disk ID.

## 3. Create the pod and attach the disk

```bash
prime pods create --id <gpu-offer-id> --name rlm-rlvr-ondemand --disks <disk-id>
prime pods list
prime pods ssh <pod-id>
```

Before using `prime pods ssh`, make sure the CLI is authenticated and your SSH key path is configured with `prime config set-ssh-key-path`.

## 4. Bootstrap the workspace on the pod

```bash
git clone <your-repo-url> prime_run
cd prime_run
prime lab setup --prime-rl
prime env install rlm_rlvr -p ./environments
```

If you want checkpoints on the attached disk, clone the repo onto that mounted disk or override `output_dir` to a path on the mounted disk before the long run.

## 5. Export only the secrets needed on the pod

The refactored environment uses the canonical `RLMEnv` path and no longer accepts `inference_mode` or local inference endpoint env args. In the standard on-demand path, `prime-rl` handles model serving for rollout generation and recursive subcalls automatically.

Export only the secrets required by your run:

```bash
export OPENROUTER_API_KEY=...
export HF_TOKEN=...
export WANDB_API_KEY=...
```

You do not need `PRIME_API_KEY` for the default `rlm_rlvr` path unless you add extra Prime-hosted integrations outside the canonical environment wrapper.

## 6. Smoke the run before scaling up

Run from the `prime-rl` package directory so the `rl` entrypoint is available from the local project:

```bash
cd prime-rl
uv run rl @ ../configs/rlm_rlvr/ondemand_smoke_qwen3_4b.toml
```

Success criteria for the smoke run:

- inference, orchestrator, and trainer all start
- at least one rollout completes
- trace export files are written
- checkpoints land in the configured `output_dir`

## 7. Promote to the 9B long/deep run

```bash
cd prime-rl
uv run rl @ ../configs/rlm_rlvr/ondemand_long_deep_qwen35_9b.toml
```

If the 9B run OOMs on the first few steps, reduce `seq_len`, `turn_max_tokens`, `subcall_max_tokens`, or `inference.gpu_memory_utilization` before moving to a larger pod shape.
