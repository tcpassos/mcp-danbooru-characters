"""Build the tag-vocabulary embeddings artifact for the suggest_tags MCP tool.

Embeds the Danbooru tag-wiki description of each general tag (count >= MIN_COUNT)
and writes data/tag_vocab/{embeddings.npy, meta.json}. The server loads only that
artifact at runtime (no dataset / pyarrow / hf_hub needed there).

Inputs:
  - data/danbooru_tags.csv         general-tag list + post counts (shipped)
  - patvessel/danbooru-rag-G-v3    natural-language description per tag (downloaded)

Run (needs the `build` extra: pyarrow, huggingface-hub):
    uv run --extra build python scripts/build_tag_vocab.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mcp_danbooru.tag_search import EMB_PATH, META_PATH, MIN_COUNT, MODEL_NAME, _ART_DIR

ROOT = Path(__file__).resolve().parents[1]
TAGS_CSV = ROOT / "data" / "danbooru_tags.csv"


def load_counts() -> dict[str, int]:
    """General tags (category 0) with count >= MIN_COUNT -> count."""
    counts: dict[str, int] = {}
    with TAGS_CSV.open(encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 3:
                continue
            try:
                if int(row[1]) == 0 and int(row[2]) >= MIN_COUNT:
                    counts[row[0]] = int(row[2])
            except ValueError:
                continue
    return counts


def load_descriptions() -> dict[str, str]:
    """tag -> natural-language description from danbooru-rag-G-v3."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "patvessel/danbooru-rag-G-v3",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(path, columns=["tag", "embed_text"])
    tags = table.column("tag").to_pylist()
    texts = table.column("embed_text").to_pylist()
    return {str(t).strip().lower(): str(x) for t, x in zip(tags, texts) if t and x}


def main() -> None:
    counts = load_counts()
    print(f"{len(counts)} general tags (count >= {MIN_COUNT})")
    descriptions = load_descriptions()
    covered = sum(1 for t in counts if t in descriptions)
    print(f"{len(descriptions)} descriptions; cover {covered}/{len(counts)} "
          f"({100 * covered / len(counts):.0f}%)")

    # Document = the tag's wiki description; fall back to the spaced tag name.
    tags = sorted(counts)
    docs = [descriptions.get(t) or t.replace("_", " ") for t in tags]

    print(f"Embedding {len(docs)} documents with {MODEL_NAME} ...")
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=MODEL_NAME)
    vecs = np.asarray(list(model.embed(docs)), dtype=np.float32)
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8, None)  # L2-normalize

    _ART_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, vecs)
    META_PATH.write_text(
        json.dumps([{"tag": t, "count": counts[t]} for t in tags], ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {EMB_PATH.relative_to(ROOT)}  shape={vecs.shape}")
    print(f"Wrote {META_PATH.relative_to(ROOT)}  ({len(tags)} tags)")


if __name__ == "__main__":
    main()
