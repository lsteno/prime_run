---
pretty_name: BEEG Agents Semantic Delegation V4
license: apache-2.0
task_categories:
  - text-classification
  - question-answering
---

# BEEG Agents Semantic Delegation V4

Private training data for verified RLM delegation experiments. It preserves the
35/40/25 Oolong, Frames, and LongCodeU mixture from semantic delegation v3.

Semantic rows support two deterministic child contracts. A child may classify
one complete chunk with a compact label, or classify at least eight verified
records from one chunk with a `record_id -> label` map. Gold record and chunk
labels remain verifier-only metadata and never appear in the visible context.
