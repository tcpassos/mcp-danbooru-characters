"""Related tag lookup — local CSV index (primary) with HTTP API fallback.

Core logic is kept separate from the MCP layer so it can be tested in isolation
and reused in other projects without the MCP framework.

Usage:
    import asyncio
    from mcp_danbooru.related_tags import get_related_tags, CooccurrenceIndex, CATEGORY_GENERAL

    # With local CSV (no API key needed):
    index = CooccurrenceIndex(
        cooc_csv="/path/to/danbooru_tags_cooccurrence.csv",
        tags_csv="/path/to/danbooru_tags.csv",
    )
    tags = asyncio.run(get_related_tags("cunnilingus", limit=15, _index=index))
    tags = asyncio.run(get_related_tags("2girls, cunnilingus", limit=15, _index=index))

    # Without CSV (falls back to Danbooru HTTP API):
    tags = asyncio.run(get_related_tags("cunnilingus", limit=15))
"""
from __future__ import annotations

import csv
import os
import re
import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag category constants (Danbooru)
# ---------------------------------------------------------------------------

CATEGORY_GENERAL   = 0  # actions, positions, objects, settings, expressions
CATEGORY_ARTIST    = 1  # illustrator/photographer handles
CATEGORY_COPYRIGHT = 3  # franchise / IP names
CATEGORY_CHARACTER = 4  # character names
CATEGORY_META      = 5  # quality, resolution, format meta-tags

# Default: only general tags — the most useful for prompt building.
DEFAULT_CATEGORIES = [CATEGORY_GENERAL]

# Sentinel: distinguishes "caller didn't pass categories" from "caller passed None to mean all".
_UNSET: list[int] = []  # identity-checked with `is`

DANBOORU_BASE = "https://danbooru.donmai.us"

# Optional API key for authenticated requests (e.g. nsfw related_tag endpoint).
# Read from env at import time; can be overridden per-call via _api_key parameter.
_ENV_API_KEY: str = os.getenv("DANBOORU_API_KEY", "")

# ---------------------------------------------------------------------------
# In-process cache (tag → raw API result list)
# Keyed by normalised tag name; populated lazily on first fetch.
# ---------------------------------------------------------------------------

_CACHE: dict[str, list[dict[str, Any]]] = {}


def clear_cache() -> None:
    """Clear the in-process cache. Useful in tests."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Local CSV index
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_DEFAULT_COOC_CSV = _DATA_DIR / "danbooru_tags_cooccurrence.csv"
_DEFAULT_TAGS_CSV = _DATA_DIR / "danbooru_tags.csv"


class CooccurrenceIndex:
    """Tag co-occurrence index built from danbooru_tags_cooccurrence.csv.

    Scans the CSV lazily per queried tag and caches results in-process.
    This avoids loading the full 98 MB file into memory while still giving
    instant responses for any tag that has been queried before.

    Defaults to the data/ directory bundled with this package — no env vars
    or external configuration needed.

    Args:
        cooc_csv: Path to danbooru_tags_cooccurrence.csv. Defaults to the
                  bundled data/danbooru_tags_cooccurrence.csv.
        tags_csv: Path to danbooru_tags.csv for category filtering. Defaults
                  to the bundled data/danbooru_tags.csv.
    """

    def __init__(
        self,
        cooc_csv: str | Path | None = None,
        tags_csv: str | Path | None = None,
        characters_jsonl: str | Path | None = None,
    ) -> None:
        cooc_csv = cooc_csv or _DEFAULT_COOC_CSV
        tags_csv = tags_csv or _DEFAULT_TAGS_CSV
        characters_jsonl = characters_jsonl or (_DATA_DIR / "characters.jsonl")
        self._cooc_csv = Path(cooc_csv)
        self._tags_csv = Path(tags_csv) if tags_csv else None
        self._characters_jsonl = Path(characters_jsonl)
        self._tag_cache: dict[str, list[tuple[str, float]]] = {}
        self._categories: dict[str, int] | None = None
        self._character_traits: frozenset[str] | None = None

    # Generic noise tags that add no information as pose companions — they are
    # either too broad, belong elsewhere in the prompt, or are meta/safety tags.
    _NOISE_TAGS: frozenset[str] = frozenset({
        "sex", "nude", "naked", "explicit", "uncensored",
        "censored", "mosaic censoring", "bar censor",
    })

    # Subject-count tags are always noise in companion results — they belong in
    # Part 1 of the prompt and are never useful as pose/act companions.
    _SUBJECT_COUNT_TAGS: frozenset[str] = frozenset({
        "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls",
        "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple_boys",
        "1other", "2others", "multiple_others",
        "girl", "boy", "girls", "boys",
    })

    def _load_character_traits(self) -> frozenset[str]:
        if self._character_traits is not None:
            return self._character_traits
        traits: set[str] = set(self._SUBJECT_COUNT_TAGS)
        if self._characters_jsonl.exists():
            try:
                import json as _json
                with self._characters_jsonl.open(encoding="utf-8") as f:
                    for line in f:
                        rec = _json.loads(line)
                        traits.update(rec.get("characteristics", []))
                        traits.update(rec.get("clothing", []))
            except Exception as exc:
                logger.warning("CooccurrenceIndex: failed to load character traits: %s", exc)
        self._character_traits = frozenset(traits)
        return self._character_traits

    def _load_categories(self) -> dict[str, int]:
        if self._categories is not None:
            return self._categories
        cats: dict[str, int] = {}
        if self._tags_csv and self._tags_csv.exists():
            try:
                with self._tags_csv.open(encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        cats[row["tag"]] = int(row["category"])
            except Exception as exc:
                logger.warning("CooccurrenceIndex: failed to load categories: %s", exc)
        self._categories = cats
        return cats

    def _scan(self, tag: str) -> list[tuple[str, float]]:
        """Scan the co-occurrence CSV for all entries involving *tag*."""
        results: list[tuple[str, float]] = []
        try:
            with self._cooc_csv.open(encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) < 3:
                        continue
                    tag_a, tag_b = row[0], row[1]
                    count = float(row[2])
                    if tag_a == tag:
                        results.append((tag_b, count))
                    elif tag_b == tag:
                        results.append((tag_a, count))
        except Exception as exc:
            logger.warning("CooccurrenceIndex: scan failed for %r: %s", tag, exc)
        results.sort(key=lambda x: -x[1])
        return results

    def get_category(self, tag: str) -> int:
        return self._load_categories().get(tag, CATEGORY_GENERAL)

    def query(
        self,
        tags: str | list[str],
        limit: int = 20,
        categories: list[int] | None = None,
        exclude_categories: list[int] | None = None,
        exclude_tags: list[str] | None = None,
        exclude_character_traits: bool = True,
    ) -> list[str]:
        """Return related tags for one or more input tags.

        Follows the same semantics as :func:`get_related_tags`:
        single tag → ranked by co-occurrence count;
        multiple tags → intersection ranked by average position, with union
        fallback when no intersection exists.

        Args:
            exclude_character_traits: When True (default), removes tags that appear
                as ``characteristics`` or ``clothing`` in the character database
                (characters.jsonl). This strips appearance/outfit traits that are
                already handled by the character pipeline, leaving only act-specific,
                positional, and contextual tags.
        """
        normalised_inputs = _parse_input(tags)
        if not normalised_inputs:
            return []

        _categories = DEFAULT_CATEGORIES if categories is None else categories
        _exclude_cats = set(exclude_categories or [])
        _exclude_norm = {_normalise_tag(t) for t in (exclude_tags or [])}
        _char_traits = self._load_character_traits() if exclude_character_traits else frozenset()
        _noise = self._NOISE_TAGS if exclude_character_traits else frozenset()

        def _filter(entries: list[tuple[str, float]]) -> list[tuple[str, float]]:
            out = []
            for related, count in entries:
                if related in _exclude_norm:
                    continue
                if related in _char_traits:
                    continue
                if related in _noise:
                    continue
                cat = self.get_category(related)
                if _categories and cat not in _categories:
                    continue
                if cat in _exclude_cats:
                    continue
                out.append((related, count))
            return out

        # Fetch and cache per-tag results.
        per_tag: list[list[tuple[str, float]]] = []
        for tag in normalised_inputs:
            if tag not in self._tag_cache:
                self._tag_cache[tag] = self._scan(tag)
            per_tag.append(_filter(self._tag_cache[tag]))

        if not any(per_tag):
            return []

        if len(normalised_inputs) == 1:
            return [r.replace("_", " ") for r, _ in per_tag[0][:limit]]

        # Multiple tags: intersection by name, ranked by average position.
        name_ranks: dict[str, list[int]] = {}
        for entries in per_tag:
            for rank, (related, _) in enumerate(entries):
                name_ranks.setdefault(related, []).append(rank)

        n = len(normalised_inputs)
        intersection = {
            name: ranks for name, ranks in name_ranks.items() if len(ranks) == n
        }
        if intersection:
            ranked = sorted(intersection.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
        else:
            ranked = sorted(
                name_ranks.items(),
                key=lambda kv: (-len(kv[1]), sum(kv[1]) / len(kv[1])),
            )
        return [name.replace("_", " ") for name, _ in ranked[:limit]]

    def clear_cache(self) -> None:
        """Evict all per-tag cached results."""
        self._tag_cache.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_tag(tag: str) -> str:
    """Lowercase, strip, replace spaces with underscores (Danbooru API form)."""
    return tag.strip().lower().replace(" ", "_")


def _parse_input(tags: str | list[str]) -> list[str]:
    """Accept a comma-separated string or a list; return normalised tag list.

    String input is split on commas only — spaces within a comma-separated
    item are part of the tag name and normalised to underscores.  To pass
    multiple space-delimited tags as a string, either use commas
    (``"2girls, spread legs"``) or pass a list (``["2girls", "spread legs"]``).
    """
    if isinstance(tags, list):
        raw = tags
    else:
        raw = [t for t in tags.split(",") if t.strip()]
    return [_normalise_tag(t) for t in raw if t.strip()]


async def _fetch_one(
    tag: str,
    client: httpx.AsyncClient,
    raw_limit: int = 100,
    api_key: str = "",
) -> list[dict[str, Any]]:
    """Fetch related tags for *one* normalised tag from the Danbooru API.

    Results are cached in-process so the same tag is never fetched twice per
    server lifetime.  Pass ``client`` so callers (and tests) control the
    HTTP transport.  When ``api_key`` is set it is sent as HTTP Basic Auth
    (Danbooru requires auth for nsfw-category endpoints).
    """
    if tag in _CACHE:
        return _CACHE[tag]

    params: dict[str, str | int] = {"query": tag, "limit": raw_limit}
    auth = (os.getenv("DANBOORU_LOGIN", ""), api_key or _ENV_API_KEY) if (api_key or _ENV_API_KEY) else None

    try:
        resp = await client.get(
            f"{DANBOORU_BASE}/related_tag.json",
            params=params,
            auth=auth,  # type: ignore[arg-type]
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("Danbooru related_tag fetch failed for %r: %s", tag, exc)
        _CACHE[tag] = []
        return []

    # Danbooru returns {"tags": [...]} with each entry having a nested "tag"
    # object and a "cosine_similarity" float.
    entries: list[dict[str, Any]] = data.get("tags", []) if isinstance(data, dict) else data
    _CACHE[tag] = entries
    return entries


def _entry_name(entry: dict[str, Any]) -> str:
    """Extract the tag name from a Danbooru related-tag entry."""
    tag_obj = entry.get("tag", entry)
    return str(tag_obj.get("name", "")).replace("_", " ")


def _entry_category(entry: dict[str, Any]) -> int:
    """Extract the tag category from a Danbooru related-tag entry."""
    tag_obj = entry.get("tag", entry)
    return int(tag_obj.get("category", CATEGORY_GENERAL))


def _entry_similarity(entry: dict[str, Any]) -> float:
    """Extract the cosine similarity score from a Danbooru related-tag entry."""
    return float(entry.get("cosine_similarity", 0.0))


def _apply_filters(
    entries: list[dict[str, Any]],
    categories: list[int],
    exclude_categories: list[int],
    exclude_tags: list[str],
) -> list[dict[str, Any]]:
    """Filter a list of raw API entries by category and tag exclusion rules."""
    exclude_norm = {_normalise_tag(t) for t in exclude_tags}
    result = []
    for entry in entries:
        cat = _entry_category(entry)
        if categories and cat not in categories:
            continue
        if cat in exclude_categories:
            continue
        name_norm = _normalise_tag(_entry_name(entry))
        if name_norm in exclude_norm:
            continue
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_related_tags(
    tags: str | list[str],
    limit: int = 20,
    categories: list[int] | None = _UNSET,  # type: ignore[assignment]
    exclude_categories: list[int] | None = None,
    exclude_tags: list[str] | None = None,
    _client: httpx.AsyncClient | None = None,
    _index: CooccurrenceIndex | None = None,
) -> list[str]:
    """Return tags related to one or more input tags.

    Args:
        tags:
            A single tag string (``"cunnilingus"``), a comma/space-separated
            string (``"2girls, cunnilingus"``), or a list
            (``["2girls", "cunnilingus"]``).
        limit:
            Maximum number of tags to return (default 20).
        categories:
            Danbooru category IDs to *include*.  Defaults to
            ``[CATEGORY_GENERAL]``.  Pass ``None`` to include all categories.
        exclude_categories:
            Danbooru category IDs to always *exclude*, regardless of
            ``categories``.  Defaults to ``[]``.
        exclude_tags:
            Specific tags to remove from the result (e.g. the input tags
            themselves).  Case- and underscore-insensitive.
        _client:
            Optional ``httpx.AsyncClient`` for testing / custom transports.
            A default client is created when ``None``.
        _index:
            Optional :class:`CooccurrenceIndex` built from local CSV files.
            When provided the HTTP API is not called at all — results come
            entirely from the index.  Takes priority over ``_client``.

    Returns:
        A list of related tag strings (spaces, not underscores), ranked by
        relevance.  When multiple input tags are provided the result is the
        *intersection* of each tag's related set, ranked by average similarity
        position.
    """
    # Fast path: use local CSV index when available.
    if _index is not None:
        _cats = DEFAULT_CATEGORIES if categories is _UNSET else ([] if categories is None else list(categories))
        return _index.query(
            tags,
            limit=limit,
            categories=_cats if _cats else None,
            exclude_categories=exclude_categories,
            exclude_tags=exclude_tags,
        )

    normalised_inputs = _parse_input(tags)
    if not normalised_inputs:
        return []

    # _UNSET (not passed) → default to CATEGORY_GENERAL only.
    # None (explicitly passed) → no category filter (include all).
    # list → filter to those categories.
    if categories is _UNSET:
        _categories: list[int] = DEFAULT_CATEGORIES
    elif categories is None:
        _categories = []  # empty → _apply_filters skips the category check
    else:
        _categories = list(categories)
    _exclude_cats = exclude_categories or []
    _exclude_tags = list(exclude_tags or [])

    own_client = _client is None
    client = _client or httpx.AsyncClient()

    try:
        # Fetch related entries for each input tag concurrently.
        fetch_tasks = [_fetch_one(tag, client) for tag in normalised_inputs]
        per_tag_entries: list[list[dict[str, Any]]] = await asyncio.gather(*fetch_tasks)
    finally:
        if own_client:
            await client.aclose()

    # Filter each tag's results independently.
    per_tag_filtered: list[list[dict[str, Any]]] = [
        _apply_filters(entries, _categories, _exclude_cats, _exclude_tags)
        for entries in per_tag_entries
    ]

    if not any(per_tag_filtered):
        return []

    if len(normalised_inputs) == 1:
        # Single tag: return directly ranked by similarity.
        ranked = sorted(per_tag_filtered[0], key=lambda e: -_entry_similarity(e))
        return [_entry_name(e) for e in ranked[:limit]]

    # Multiple tags: intersection ranked by average position across sets.
    # Build name→average_rank mapping.
    name_ranks: dict[str, list[int]] = {}
    for entries in per_tag_filtered:
        for rank, entry in enumerate(entries):
            name = _entry_name(entry)
            name_ranks.setdefault(name, []).append(rank)

    # Keep only names that appear in ALL per-tag result sets.
    n = len(normalised_inputs)
    intersection = {
        name: ranks
        for name, ranks in name_ranks.items()
        if len(ranks) == n
    }

    if not intersection:
        # Fallback: union ranked by how many input tags they appeared in,
        # then by average rank.
        scored = sorted(
            name_ranks.items(),
            key=lambda kv: (-len(kv[1]), sum(kv[1]) / len(kv[1])),
        )
        return [name for name, _ in scored[:limit]]

    ranked = sorted(
        intersection.items(),
        key=lambda kv: sum(kv[1]) / len(kv[1]),
    )
    return [name for name, _ in ranked[:limit]]
