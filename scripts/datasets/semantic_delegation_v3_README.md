---
pretty_name: BEEG Agents Semantic Delegation V3
language:
  - en
task_categories:
  - text-classification
  - question-answering
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.parquet
      - split: eval
        path: eval.parquet
---

# BEEG Agents Semantic Delegation V3

Private RL training data for semantic delegation experiments. The dataset keeps
the original BEEG Agents train/eval family mixture and replaces Oolong rows with
long-context semantic chunk tasks generated from sources excluded from the
Oolong benchmark.

Semantic rows contain 4, 8, or 16 named chunks. Every chunk has a controlled
65/35 majority over two basic semantic labels, and the model must return either
the complete chunk-to-majority map or a global comparison over 16 chunks.

## Important

`metadata.record_labels`, `metadata.record_chunks`, and
`metadata.chunk_labels` are verifier-only gold fields. Training environments
must not place those fields in model prompts, REPL context, traces, or sample
tables. The visible context contains only record IDs and unlabeled text.

See `manifest.json` and `validation_summary.json` for generation parameters,
source exclusions, split counts, and validation results.
