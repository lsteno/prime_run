---
pretty_name: BEEG Agents Semantic Delegation V5
license: apache-2.0
task_categories:
  - text-classification
  - question-answering
---

# BEEG Agents Semantic Delegation V5

Private training data for natural semantic delegation experiments. It preserves
the 35/40/25 Oolong, Frames, and LongCodeU mixture from semantic delegation v4.

Semantic contexts contain named sections with ordinary bullet-point texts. The
root may delegate semantic reading naturally; child calls have no required IDs,
schemas, or output protocol. Gold labels are keyed by hidden normalized-text
hashes for verifier-side local credit and never appear in visible task data.
