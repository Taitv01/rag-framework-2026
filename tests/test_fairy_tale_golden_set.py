"""Consistency checks for the fairy-tale evaluation golden set."""

import json
import re
import unicodedata
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "evals" / "fairy_tales"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip()


def test_golden_set_matches_corpus():
    corpus = {
        path.stem: _normalize(path.read_text(encoding="utf-8"))
        for path in (EVAL_DIR / "corpus").glob("*.md")
    }
    lines = (EVAL_DIR / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]

    assert len(cases) >= 50
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(1 for case in cases if not case["sources"]) >= 5

    for case in cases:
        if not case["sources"]:
            assert not case["evidence"], case["id"]
            continue
        assert set(case["sources"]) <= set(corpus), case["id"]
        texts = [corpus[source] for source in case["sources"]]
        for phrase in case["evidence"]:
            assert any(_normalize(phrase) in text for text in texts), (case["id"], phrase)
