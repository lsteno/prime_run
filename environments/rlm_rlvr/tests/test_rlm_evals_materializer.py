from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "rlm_eval" / "materialize_rlm_evals_paper.py"
SPEC = importlib.util.spec_from_file_location("materialize_rlm_evals_paper", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
materialize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materialize)


class _FakeDataset(list):
    def filter(self, fn):
        return _FakeDataset([row for row in self if fn(row)])

    def select(self, indices):
        return _FakeDataset([self[index] for index in indices])


def _enc(text: str) -> str:
    raw = text.encode("utf-8")
    key = materialize._derive_xor_key(materialize.BROWSECOMP_PLUS_CANARY, len(raw))
    return base64.b64encode(bytes(a ^ b for a, b in zip(raw, key, strict=True))).decode("ascii")


def test_materializer_decrypts_browsecomp_and_keeps_full_oolong_pair_gold(monkeypatch, tmp_path: Path) -> None:
    def _fake_load_dataset(repo_id, config=None, split=None, streaming=False):
        del split
        if repo_id == materialize.DATASET_ID and config == "longbench_v2_codeqa":
            return _FakeDataset(
                [
                    {
                        "source_id": "code-1",
                        "question": "Q?",
                        "choice_A": "A",
                        "answer": "A",
                        "context": "code context",
                    }
                ]
            )
        if repo_id == materialize.DATASET_ID and config == "browsecomp_plus":
            doc = {"docid": _enc("doc-1"), "url": _enc("https://example.com"), "text": _enc("plain evidence")}
            return _FakeDataset(
                [
                    {
                        "source_id": "browse-1",
                        "query": _enc("Who founded Example University?"),
                        "answer": _enc("Ada Lovelace"),
                        "answer_type": "free_form",
                        "evidence_docs_json": json.dumps([doc]),
                        "gold_docs_json": "[]",
                        "negative_docs_json": "[]",
                    }
                ]
            )
        if repo_id == materialize.DATASET_ID and config == "oolong_trec_coarse":
            return _FakeDataset(
                [
                    {
                        "source_id": "trec-1",
                        "context_len": 131072,
                        "question": "Most common label?",
                        "context_window_text": "trec context",
                        "answer": ["entity"],
                        "answer_type": "classification",
                    }
                ]
            )
        if repo_id == materialize.DATASET_ID and config == "oolong_pairs_contexts":
            return _FakeDataset(
                [
                    {
                        "context_len": 32768,
                        "source_id": "ctx-32768",
                        "context_window_id": 0,
                        "context_window_text": "Question lines without labels",
                    }
                ]
            )
        raise AssertionError((repo_id, config, streaming))

    pairs_path = tmp_path / "oolong-pairs-32768.json"
    pairs_path.write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "question": "List pairs.",
                    "answer": ["(1, 2)", "(3, 4)"],
                    "type": "list_of_answers",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(materialize, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(materialize, "hf_hub_download", lambda **kwargs: str(pairs_path))

    records = materialize.build_records(seed=42, browse_sample_size=10, browse_max_docs=5)
    by_dataset = {row["dataset"]: row for row in records}

    browse = by_dataset["browsecomp_plus"]
    assert browse["prompt"] == "Who founded Example University?"
    assert browse["answer"] == "Ada Lovelace"
    assert "plain evidence" in browse["context"]
    assert "https://example.com" in browse["context"]

    pairs = by_dataset["oolong_pairs"]
    acceptable = json.loads(pairs["acceptable_answers"])
    assert len(acceptable) == 1
    assert json.loads(acceptable[0]) == ["(1, 2)", "(3, 4)"]
    metadata = json.loads(pairs["metadata"])
    assert metadata["gold_pairs_count"] == 2
    assert materialize.validate_records(records)["oolong_pairs_full_gold_rows"] == 1
