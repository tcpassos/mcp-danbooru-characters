"""Regression test for the tag-vocabulary search (suggest_tags).

Skipped when the precomputed artifact is missing (build it with
scripts/build_tag_vocab.py). Loads the fastembed model once.
"""
from __future__ import annotations

import pytest

from mcp_danbooru.tag_search import EMB_PATH, lex_norm, search

pytestmark = pytest.mark.skipif(
    not EMB_PATH.exists(),
    reason="tag-vocab artifact not built (run scripts/build_tag_vocab.py)",
)

# concept -> a tag that MUST appear in the top-k (high-confidence mappings).
_MUST_HIT = [
    ("viewed from behind", "from behind"),
    ("tight close-up on her face", "close-up"),
    ("a blank, emotionless expression", "expressionless"),
    ("two characters standing back to back", "back-to-back"),
    ("soft god rays streaming through the trees", "light rays"),
]


@pytest.mark.parametrize("concept,expected", _MUST_HIT, ids=[c for c, _ in _MUST_HIT])
def test_concept_returns_expected_tag(concept, expected):
    got = {lex_norm(t) for t in search(concept, k=8)}
    assert lex_norm(expected) in got, f"{expected!r} not in {sorted(got)}"


def test_empty_concept_returns_nothing():
    assert search("   ", k=5) == []


def test_returns_at_most_k():
    assert len(search("a girl smiling in the rain", k=4)) <= 4
