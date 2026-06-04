"""Semantic + lexical search over the Danbooru general-tag vocabulary.

Turns a natural-language concept ("camera angled low, looking up at her") into
native Danbooru tags ("from below", ...) so an image-prompt agent can use the
model's trained vocabulary instead of weak free-text.

Lean runtime: a precomputed embeddings artifact (data/tag_vocab/) + fastembed
(onnx all-MiniLM-L6-v2, no torch) to embed the query + a numpy cosine search with
a LEXICAL-DOMINANT rerank. Literal tag-name containment is near-certain for tag
retrieval, so it outranks the (noisier) vector similarities; the vector arm
handles paraphrase ("god rays" -> light_rays/sunbeam). Regenerate the artifact
with scripts/build_tag_vocab.py.

Calibrated on an 18-concept golden set: recall@5 ~0.61, using the Danbooru
tag-wiki descriptions (patvessel/danbooru-rag-G-v3) as the embedded documents.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_COUNT = 200  # drop tags the model barely saw

_ART_DIR = Path(__file__).resolve().parents[2] / "data" / "tag_vocab"
EMB_PATH = _ART_DIR / "embeddings.npy"
META_PATH = _ART_DIR / "meta.json"


def lex_norm(t: str) -> str:
    """Lexical-match normalization: underscores AND hyphens -> spaces, lowercase."""
    return (t or "").strip().lower().replace("_", " ").replace("-", " ")


def _spaced(t: str) -> str:
    return t.replace("_", " ")


# ---- lazily-loaded state ----
_emb: np.ndarray | None = None
_tags: list[str] | None = None
_counts: list[int] | None = None
_lextags: list[str] | None = None
_maxlog: float = 1.0
_model = None


def _load_artifact() -> None:
    global _emb, _tags, _counts, _lextags, _maxlog
    if _emb is not None:
        return
    if not EMB_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            f"tag-vocab artifact missing under {_ART_DIR}. "
            "Build it with: python scripts/build_tag_vocab.py"
        )
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    _tags = [m["tag"] for m in meta]
    _counts = [int(m["count"]) for m in meta]
    _lextags = [lex_norm(t) for t in _tags]
    _maxlog = math.log10(max(_counts) + 1) if _counts else 1.0
    _emb = np.load(EMB_PATH).astype(np.float32)  # rows are L2-normalized


def _embed_query(text: str) -> np.ndarray:
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=MODEL_NAME)
    v = np.asarray(next(iter(_model.embed([text]))), dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def search(concept: str, k: int = 8, candidates: int = 60) -> list[str]:
    """Return up to `k` native Danbooru tags (spaced) matching `concept`."""
    _load_artifact()
    assert _emb is not None and _tags is not None and _counts is not None and _lextags is not None
    if not concept.strip():
        return []

    q = _embed_query(concept)
    sims = _emb @ q  # cosine (both normalized)

    cn = lex_norm(concept)
    cn_words = set(cn.split())

    # vector arm: top-N candidates by cosine.
    n = min(candidates, len(sims))
    top_idx = np.argpartition(-sims, n - 1)[:n]
    scores: dict[int, float] = {int(i): 0.7 * float(sims[i]) for i in top_idx}

    # lexical arm: high-precision, can introduce tags the vector missed.
    for i, tl in enumerate(_lextags):
        if not tl:
            continue
        tw = tl.split()
        if tl == cn:
            lex = 3.0
        elif tw and set(tw) <= cn_words:
            lex = 2.0 if len(tw) >= 2 else 1.0
        else:
            continue
        base = scores.get(i, 0.7 * float(sims[i]))
        scores[i] = base + lex

    ranked = sorted(
        ((s + 0.05 * (math.log10(_counts[i] + 1) / _maxlog), i) for i, s in scores.items()),
        reverse=True,
    )
    return [_spaced(_tags[i]) for _, i in ranked[:k]]
