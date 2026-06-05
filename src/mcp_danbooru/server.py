from mcp.server.fastmcp import FastMCP
import json
import os
import re
import asyncio

try:
    # Installed as package (e.g. via pip install -e .)
    from mcp_danbooru.related_tags import (
        CATEGORY_GENERAL,
        CATEGORY_ARTIST,
        CATEGORY_COPYRIGHT,
        CATEGORY_CHARACTER,
        CATEGORY_META,
        CooccurrenceIndex,
        get_related_tags as _get_related_tags,
    )
except ImportError:
    # Run as script: python server.py — sibling file in same directory
    from related_tags import (  # type: ignore[no-redef]
        CATEGORY_GENERAL,
        CATEGORY_ARTIST,
        CATEGORY_COPYRIGHT,
        CATEGORY_CHARACTER,
        CATEGORY_META,
        CooccurrenceIndex,
        get_related_tags as _get_related_tags,
    )

# Load local CSV index from the bundled data/ directory (no env vars needed).
# Falls back to Danbooru HTTP API only if the CSV files are not present.
try:
    _cooc_index: CooccurrenceIndex | None = CooccurrenceIndex()
except Exception:
    _cooc_index = None

mcp = FastMCP(
    "DanbooruCharacters",
    instructions=(
        "This server gives you access to a structured database of 21,000+ Danbooru anime characters. "
        "Use it to build precise image-generation prompts for the Anima model.\n"
        "\n"
        "CORE WORKFLOW:\n"
        "1. get_character_tags(name) — exact lookup when you know the character tag.\n"
        "2. search_characters(query) — fuzzy search when the name is uncertain.\n"
        "3. find_characters_by_trait(traits) — search by visual traits (e.g. '1girl, blue hair, twintails').\n"
        "   If a trait returns suggestions instead of results, retry with a suggested tag name.\n"
        "4. get_character_appearance(name) / get_character_outfit(name) — split appearance vs. outfit\n"
        "   when you want to mix looks from different characters.\n"
        "5. get_character_variants(name) — list alternate versions of a character.\n"
        "6. list_series_characters(series) — list all characters from a franchise.\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "  character_tag: <exact danbooru tag to use in the prompt>\n"
        "  prompt_tags:   <full comma-separated tag string ready for Anima>\n"
        "\n"
        "ANIMA PROMPT TIPS:\n"
        "- Always include the character_tag in your prompt so Anima recognises the character.\n"
        "- Prepend quality boosters before the character tags:\n"
        "  masterpiece, best quality, absurdres, <character tags here>\n"
        "- To depict a specific outfit, append or override the clothing tags from get_character_outfit().\n"
        "- To depict a character with another character's look, combine appearance_tags from one\n"
        "  with outfit_tags from the other."
    ),
)

# ---------------------------------------------------------------------------
# Pre-load — runs once when the server starts
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.lower())

def _display_name(raw_name: str) -> str:
    """Convert 'hatsune_miku' -> 'hatsune miku'."""
    return raw_name.replace("_", " ")

def _character_tag(record: dict) -> str:
    """Return the exact danbooru character tag (display name with spaces)."""
    return _display_name(record["name"])

def _format_entry(record: dict) -> str:
    """Format a single result with clearly separated character_tag and prompt_tags.
    character_tag is the exact tag to reference the character in a prompt.
    franchise_tag is the first copyright tag (underscore format, e.g. sono_bisque_doll_wa_koi_wo_suru).
    prompt_tags is the full comma-separated list ready for Anima."""
    char_tag = _character_tag(record)
    copyrights = record.get("copyright") or []
    franchise_line = f"\nfranchise_tag: {copyrights[0]}" if copyrights else ""
    return f"character_tag: {char_tag}{franchise_line}\nprompt_tags: {_build_tag_string(record)}"

def _build_tag_string(record: dict) -> str:
    """Assemble a prompt-ready tag string from the structured record fields."""
    name = _display_name(record.get("name", ""))
    copyrights = [_display_name(c) for c in record.get("copyright", [])]
    gender = record.get("gender", "")
    characteristics = [_display_name(t) for t in record.get("characteristics", [])]
    clothing = [_display_name(t) for t in record.get("clothing", [])]

    parts = [name] + copyrights
    if gender:
        parts.append(gender)
    parts += characteristics + clothing
    return ", ".join(p for p in parts if p)

def _build_indexes(by_id):
    name_index = {}
    series_index = {}
    trait_index = {}

    for record in by_id.values():
        norm_name = _display_name(record["name"])
        copyrights_disp = [_display_name(c) for c in record.get("copyright", [])]

        text = _normalize(f"{norm_name} {' '.join(copyrights_disp)}")
        for word in set(text.split()):  # set deduplicates words already in name via disambiguation
            if word:
                name_index.setdefault(word, []).append(norm_name)

        for series in copyrights_disp:
            series_norm = _normalize(series)
            if series_norm:
                series_index.setdefault(series_norm, []).append(norm_name)

        all_traits = record.get("characteristics", []) + record.get("clothing", [])
        for tag in all_traits:
            tag_norm = _normalize(_display_name(tag))
            if tag_norm:
                trait_index.setdefault(tag_norm, []).append(norm_name)

        # Index gender separately (e.g. "1girl", "1boy")
        gender = record.get("gender", "")
        if gender:
            trait_index.setdefault(_normalize(gender), []).append(norm_name)

    return name_index, series_index, trait_index

def _load():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "characters.jsonl")
    by_id = {}
    by_name = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            by_id[record["id"]] = record
            by_name[_display_name(record["name"])] = record

    name_idx, series_idx, trait_idx = _build_indexes(by_id)
    return by_id, by_name, name_idx, series_idx, trait_idx

_BY_ID, _BY_NAME, _INDEX, _SERIES_INDEX, _TRAIT_INDEX = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intersect(index, keywords):
    candidates = None
    for word in keywords:
        hits = set(index.get(word, []))
        candidates = hits if candidates is None else candidates & hits
    return candidates

def _rank(candidates, query_words):
    """Sort by relevance score then post_count desc.
    Scores: 3=exact name | 2=name starts with query | 1=all words in name | 0=series match."""
    base_norm = " ".join(query_words)
    scored = []
    for norm_name in candidates:
        record = _BY_NAME.get(norm_name, {})
        name_norm = _normalize(norm_name)
        name_words = set(name_norm.split())
        if name_norm == base_norm:
            score = 3
        elif name_norm.startswith(base_norm):
            score = 2
        elif all(w in name_words for w in query_words):
            score = 1
        else:
            score = 0
        post_count = record.get("post_count", 0)
        scored.append((-score, -post_count, norm_name))
    scored.sort()
    return [norm_name for _, _, norm_name in scored]

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_character_tags(tag: str) -> str:
    """Look up a character's visual tags by their exact Danbooru tag name.
    Accepts either underscored ('hatsune_miku') or spaced ('hatsune miku') format.
    Returns a prompt-ready tag string with name, series, gender, characteristics and clothing."""
    norm = tag.strip().lower().replace("_", " ")
    record = _BY_NAME.get(norm)
    if not record:
        return (
            f"Tag '{tag}' not found. "
            "Use search_characters() to find the correct tag name first."
        )
    return _format_entry(record)


@mcp.tool()
def search_characters(query: str, max_results: int = 5) -> str:
    """Search for Danbooru characters by name or series keywords and return their visual tags.
    Use this when you are not sure of the exact tag name.
    Results are ranked by relevance and post count (popularity).
    max_results: how many matches to return (default 5)."""
    keywords = _normalize(query.replace("_", " ")).split()
    if not keywords:
        return "No keywords provided."

    candidates = _intersect(_INDEX, keywords)
    if not candidates:
        return f"No character found for: {query}."

    ranked = _rank(candidates, keywords)
    results = []
    for norm_name in ranked[:max_results]:
        record = _BY_NAME[norm_name]
        results.append(_format_entry(record))
    return "Matches found:\n\n" + "\n\n".join(results)


@mcp.tool()
def list_series_characters(series_name: str, max_results: int = 20) -> str:
    """List characters from a given series/franchise (e.g. 'touhou', 'fate grand order').
    Results are sorted by post count (most popular first).
    max_results: how many unique character names to return (default 20)."""
    query_words = _normalize(series_name.replace("_", " ")).split()
    if not query_words:
        return "No series name provided."

    matching = []
    for series_norm, norm_names in _SERIES_INDEX.items():
        series_words = series_norm.split()
        if all(w in series_words for w in query_words):
            for norm_name in norm_names:
                record = _BY_NAME.get(norm_name, {})
                matching.append((record.get("post_count", 0), norm_name))

    if not matching:
        return f"No characters found for series: {series_name}."

    seen = set()
    unique = []
    for post_count, norm_name in matching:
        if norm_name not in seen:
            seen.add(norm_name)
            unique.append((post_count, norm_name))
    unique.sort(key=lambda x: -x[0])

    results = [norm_name for _, norm_name in unique[:max_results]]
    header = f"Characters in '{series_name}' ({len(unique)} total, showing {len(results)}):\n"
    return header + "\n".join(f"- {_character_tag(_BY_NAME[r])}" for r in results)


@mcp.tool()
def get_character_variants(character_name: str, max_results: int = 20) -> str:
    """Return all costume/form variants of a character using the relationship graph.
    For example, 'hatsune miku' returns all its children entries (outfit-specific variants).
    max_results: how many variants to return (default 20)."""
    norm = character_name.strip().lower().replace("_", " ")
    record = _BY_NAME.get(norm)
    if not record:
        return (
            f"Character '{character_name}' not found. "
            "Use search_characters() to find the correct name first."
        )

    children_ids = record.get("relationships", {}).get("children", [])
    if not children_ids:
        return f"'{character_name}' has no registered variants in the database."

    variants = []
    for child_id in children_ids:
        child = _BY_ID.get(child_id)
        if child:
            variants.append((child.get("post_count", 0), _character_tag(child)))

    variants.sort(key=lambda x: -x[0])
    lines = [f"- {tag}" for _, tag in variants[:max_results]]
    return f"Variants of '{character_name}' ({len(variants)} found):\n" + "\n".join(lines)


@mcp.tool()
def find_characters_by_trait(traits: str, max_results: int = 10) -> str:
    """Find characters that have specific visual traits.
    Pass comma-separated danbooru tag strings, e.g. 'blue hair, twintails, 1girl'.
    Traits are matched against each character's characteristics, clothing, and gender fields.
    Results are sorted by post count (most popular first).
    If a trait is not found, similar tag suggestions are returned instead of an empty result.
    max_results: how many characters to return (default 10)."""
    raw_traits = [t.strip() for t in traits.split(",") if t.strip()]
    if not raw_traits:
        return "No traits provided."

    candidates = None
    unmatched: list[tuple[str, list[str]]] = []

    for trait in raw_traits:
        trait_norm = _normalize(trait.replace("_", " "))
        hits = set(_TRAIT_INDEX.get(trait_norm, []))
        if not hits:
            words = trait_norm.split()
            similar = [k for k in _TRAIT_INDEX if all(w in k for w in words)]
            similar.sort(key=lambda k: -len(_TRAIT_INDEX[k]))
            unmatched.append((trait, similar[:8]))
        else:
            candidates = hits if candidates is None else candidates & hits

    if unmatched:
        lines = []
        for trait, suggestions in unmatched:
            if suggestions:
                lines.append(f"  '{trait}' not found. Did you mean: {', '.join(suggestions)}")
            else:
                lines.append(f"  '{trait}' not found. No similar tags in the dataset.")
        return "Some traits had no exact match; retry with corrected tag names:\n" + "\n".join(lines)

    if not candidates:
        return f"No characters found with all of these traits: {traits}."

    ranked = sorted(
        candidates,
        key=lambda n: -_BY_NAME.get(n, {}).get("post_count", 0)
    )

    results = []
    for norm_name in ranked[:max_results]:
        record = _BY_NAME[norm_name]
        results.append(f"- {_character_tag(record)}")

    total = len(candidates)
    return f"Characters with [{traits}] ({total} total, showing {min(max_results, total)}):\n" + "\n".join(results)


def _resolve(name_or_tag: str) -> dict | None:
    """Look up a record by display name or underscored tag, return None if not found."""
    norm = name_or_tag.strip().lower().replace("_", " ")
    return _BY_NAME.get(norm)


@mcp.tool()
def get_character_appearance(character_name: str) -> str:
    """Return only the physical appearance tags of a character (gender + characteristics).
    Excludes clothing. Useful for prompts like 'character A with the look of character B'
    or when you want to describe what a character looks like without their outfit."""
    record = _resolve(character_name)
    if not record:
        return (
            f"Character '{character_name}' not found. "
            "Use search_characters() to find the correct name first."
        )
    gender = record.get("gender", "")
    characteristics = [_display_name(t) for t in record.get("characteristics", [])]
    parts = ([gender] if gender else []) + characteristics
    if not parts:
        return f"No appearance data found for '{character_name}'."
    char_tag = _character_tag(record)
    return f"character_tag: {char_tag}\nappearance_tags: " + ", ".join(parts)


@mcp.tool()
def get_character_outfit(character_name: str) -> str:
    """Return only the clothing/outfit tags of a character.
    Useful for prompts like 'character X wearing the outfit of character Y',
    or for mixing appearances and outfits from different characters."""
    record = _resolve(character_name)
    if not record:
        return (
            f"Character '{character_name}' not found. "
            "Use search_characters() to find the correct name first."
        )
    clothing = [_display_name(t) for t in record.get("clothing", [])]
    if not clothing:
        return f"No outfit data found for '{character_name}'."
    char_tag = _character_tag(record)
    return f"character_tag: {char_tag}\noutfit_tags: " + ", ".join(clothing)


@mcp.tool()
def suggest_tags(concept: str, max_results: int = 8) -> str:
    """Convert a natural-language SCENE / POSE / LIGHTING / CAMERA concept into
    native Danbooru tags the image model was actually trained on.

    Use this when you have a free-text idea ("camera angled low looking up",
    "soft god rays through the trees", "she peeks around the corner") and want the
    canonical Danbooru tags that trigger reliably, instead of inventing weak
    free-text. Weave the returned tags into the prompt alongside natural language.

    Best results with a SHORT, concrete concept (a phrase, not a paragraph).

    NOTE: this is for general scene/pose/lighting/camera/action vocabulary. For a
    character's identity and appearance, use search_characters / get_character_tags
    instead (those are authoritative; do not source character tags from here).

    Returns a comma-separated list of canonical Danbooru tags."""
    try:
        try:
            from .tag_search import search  # installed as a package
        except ImportError:
            from tag_search import search  # run as a script: python server.py
        tags = search(concept, k=max_results)
    except FileNotFoundError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # keep the server alive on any embed/index failure
        return f"Error: tag search failed: {exc}"
    if not tags:
        return f"No tags found for: {concept}"
    return ", ".join(tags)


@mcp.tool()
async def get_related_tags(
    tags: str,
    limit: int = 20,
    categories: list[int] | None = None,
    exclude_categories: list[int] | None = None,
    exclude_tags: list[str] | None = None,
    exclude_character_traits: bool | None = None,
) -> str:
    """Return Danbooru tags that co-occur with one or more input tags.

    tags:
        A single tag ("cunnilingus") or comma/space-separated list
        ("2girls, cunnilingus"). Multiple tags produce the *intersection*
        of their related sets — tags relevant to the full combination.
        Falls back to a union ranked by coverage when no intersection exists.
    limit:
        Maximum number of tags to return (default 20).
    categories:
        Danbooru category IDs to include. Defaults to [0] (general tags —
        actions, positions, objects, expressions). Available constants:
        0=general, 1=artist, 3=copyright, 4=character, 5=meta.
        Pass null/None to include all categories.
    exclude_categories:
        Category IDs to always exclude regardless of `categories`.
    exclude_tags:
        Specific tags to remove from the result (case/underscore-insensitive).

    Returns a comma-separated list of related tags, or a message when none
    are found.

    Examples:
        get_related_tags("cunnilingus")
        get_related_tags("2girls, cunnilingus", limit=15)
        get_related_tags("cunnilingus", exclude_tags=["2girls", "group sex"])
        get_related_tags("portrait", categories=[0], limit=10)
    """
    if _cooc_index is not None:
        result = _cooc_index.query(
            tags=tags,
            limit=limit,
            categories=categories,
            exclude_categories=exclude_categories,
            exclude_tags=exclude_tags,
            exclude_character_traits=exclude_character_traits if exclude_character_traits is not None else True,
        )
    else:
        result = await _get_related_tags(
            tags=tags,
            limit=limit,
            categories=categories,
            exclude_categories=exclude_categories,
            exclude_tags=exclude_tags,
        )
    if not result:
        return f"No related tags found for: {tags}"
    return ", ".join(result)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@mcp.prompt()
def generate_image_prompt(character_name: str, extra_tags: str = "") -> str:
    """Build a ready-to-use Anima image-generation prompt for a character.
    character_name: the character's name (can be underscored or spaced).
    extra_tags: optional extra tags to append (e.g. scene, style, pose).
    Returns a structured step-by-step guide that leads to a finished prompt."""
    return (
        f"Build an Anima image-generation prompt for the character '{character_name}'.\n"
        "\n"
        "Steps:\n"
        f"1. Call get_character_tags('{character_name}').\n"
        "   - If the character is not found, call search_characters() with variations of the name.\n"
        "   - Note the character_tag and prompt_tags from the result.\n"
        "2. Decide what to include:\n"
        "   - Full character (appearance + outfit): use prompt_tags directly.\n"
        "   - Appearance only (no outfit): call get_character_appearance() and use appearance_tags.\n"
        "   - Outfit only: call get_character_outfit() and use outfit_tags.\n"
        "3. Assemble the final prompt in this order:\n"
        "   masterpiece, best quality, absurdres, <prompt_tags>, <extra_tags>\n"
        + (f"   Extra tags to include: {extra_tags}\n" if extra_tags else "")
        + "4. Output only the final assembled prompt string, nothing else."
    )


@mcp.prompt()
def find_character_for_scene(
    scene_description: str,
    preferred_series: str = "",
    max_results: int = 5,
) -> str:
    """Find the best-matching character(s) for a described scene or role.
    scene_description: describe the visual traits or role you need (e.g. 'a tall girl with silver hair and a sword').
    preferred_series: optionally restrict to a franchise (e.g. 'fate').
    max_results: how many candidates to return.
    Returns a structured guide for the model to execute the search."""
    steps = (
        f"Find a character that fits this description: '{scene_description}'.\n"
        "\n"
        "Steps:\n"
        "1. Extract visual traits from the description (hair color, eye color, gender, accessories, etc.).\n"
        "   Convert them to danbooru-style tags, e.g. 'silver hair', '1girl', 'sword'.\n"
        "2. Call find_characters_by_trait(traits) with the extracted tags.\n"
        "   - If a trait returns suggestions ('did you mean'), retry with a suggested tag.\n"
        "   - Remove the least specific trait if there are too few results.\n"
    )
    if preferred_series:
        steps += (
            f"3. If results are plentiful, filter by calling list_series_characters('{preferred_series}')\n"
            "   and cross-reference with the trait results.\n"
        )
        steps += "4."
    else:
        steps += "3."
    steps += (
        f" Present the top {max_results} candidates with their character_tag and prompt_tags.\n"
        "   Then ask the user which character to use before generating the final prompt."
    )
    return steps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    # Pre-load the tag-search embedding model in this (main) thread before the
    # async MCP loop starts. Lazy onnxruntime init inside an async tool handler
    # crashes the stdio server (native crash, no traceback). Warming is best-effort:
    # if deps/artifact are missing, suggest_tags returns an error but the rest works.
    import sys
    try:
        try:
            from .tag_search import warm
        except ImportError:
            from tag_search import warm
        warm()
    except Exception as exc:
        print(f"[mcp-danbooru] tag-search warmup skipped: {exc}", file=sys.stderr)
    mcp.run()

if __name__ == "__main__":
    run()
