"""Parity check for the production tag search (fastembed runtime).

Runs the same 18-concept golden set used to calibrate the approach (recall@5 was
0.611 with chroma's onnx all-MiniLM). Confirms the fastembed runtime + the
precomputed artifact reproduce that. Concepts validated against the danbooru
index; acceptable_tags = any one counts as a hit.

    uv run python scripts/eval_tag_vocab.py
"""
from __future__ import annotations

from mcp_danbooru.tag_search import lex_norm, search

K = 5

# concept -> acceptable canonical tags (any in top-K = hit). PT cases included
# as a multilingual stress test (the agents emit English, so they are not
# representative of real calls).
GOLDEN: list[tuple[str, list[str]]] = [
    ("soft god rays streaming through the trees", ["light_rays", "sunbeam", "dappled_sunlight"]),
    ("rim lighting outlining her hair", ["backlighting"]),
    ("warm golden hour glow", ["sunset", "sunlight"]),
    ("a girl peeking around the corner", ["peeking_out", "peeking"]),
    ("camera angled low, looking up at her", ["from_below"]),
    ("shot from a tilted, off-kilter angle", ["dutch_angle"]),
    ("she arches her back", ["arched_back"]),
    ("sitting with both legs folded to one side, japanese style", ["wariza"]),
    ("viewed from behind", ["from_behind"]),
    ("hand resting on her hip", ["hand_on_own_hip"]),
    ("two characters standing back to back", ["back-to-back"]),
    ("she straddles him on top, facing forward", ["cowgirl_position"]),
    ("a blank, emotionless expression", ["expressionless"]),
    ("a sly, knowing half-smile", ["smirk"]),
    ("dreamy out-of-focus background", ["blurry_background", "bokeh", "depth_of_field"]),
    ("tight close-up on her face", ["close-up", "portrait"]),
    ("garota espiando atras da quina", ["peeking_out", "peeking"]),
    ("luz de fundo contornando o cabelo", ["backlighting"]),
]


def main() -> None:
    hits = 0
    print(f"\n{'CONCEPT':<46}{'HIT':>5}  TOP-{K}")
    print("-" * 100)
    for concept, acceptable in GOLDEN:
        acc = {lex_norm(a) for a in acceptable}
        got = search(concept, k=K)
        hit = any(lex_norm(t) in acc for t in got)
        hits += hit
        print(f"{concept[:44]:<46}{('OK' if hit else '--'):>5}  {', '.join(got)}")
    print("-" * 100)
    print(f"\n  recall@{K}: {hits / len(GOLDEN):.3f}  ({hits}/{len(GOLDEN)})   "
          f"[calibration reference: 0.611]")


if __name__ == "__main__":
    main()
