# mcp-danbooru-characters

An MCP server that lets AI models look up Danbooru anime character visual tags for image generation. Designed to work with the [Anima](https://huggingface.co/Lykon/dreamshaper-8) model family, but compatible with any booru-aware image generator.

## What it does

Given a character name or a set of visual traits, the server returns structured danbooru tag strings ready to paste into an image prompt. It understands relationships between character variants, can split appearance from outfit tags, and guides the model when a tag is ambiguous.

## Dataset

Character data comes from the [booru-characters](https://huggingface.co/datasets/Sn0w123/booru-characters) dataset by **Sn0w123** on Hugging Face. It contains 21,000+ structured Danbooru character entries with gender, characteristics, clothing, copyright, and relationship graph fields.

## Tools

| Tool | Description |
|---|---|
| `get_character_tags(tag)` | Exact lookup by Danbooru tag name. Returns `character_tag` and `prompt_tags`. |
| `search_characters(query, max_results)` | Fuzzy search by name or series. Ranked by relevance and popularity. |
| `find_characters_by_trait(traits, max_results)` | Search by visual traits (e.g. `"1girl, blue hair, twintails"`). Suggests similar tags when a trait is not found. |
| `get_character_appearance(name)` | Returns gender + characteristics only (no outfit). |
| `get_character_outfit(name)` | Returns clothing tags only. |
| `get_character_variants(name, max_results)` | Lists costume/form variants of a character. |
| `list_series_characters(series, max_results)` | Lists all characters from a franchise, sorted by popularity. |

## Prompts

| Prompt | Description |
|---|---|
| `generate_image_prompt(character_name, extra_tags)` | Guided workflow to build a finished Anima prompt for a character. |
| `find_character_for_scene(scene_description, preferred_series, max_results)` | Guided workflow to find a character matching a scene description. |

## Installation

### Option A — run directly with `uvx` (recommended)

```bash
uvx --from git+https://github.com/<your-username>/anima-prompt-generator mcp-danbooru
```

### Option B — install from source

```bash
git clone https://github.com/<your-username>/anima-prompt-generator
cd anima-prompt-generator
pip install -e .
mcp-danbooru
```

## MCP client configuration

Add the server to your MCP client config (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "danbooru-characters": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/<your-username>/anima-prompt-generator",
        "mcp-danbooru"
      ]
    }
  }
}
```

Or if running from a local install:

```json
{
  "mcpServers": {
    "danbooru-characters": {
      "command": "mcp-danbooru"
    }
  }
}
```

## Usage example

Ask your AI assistant:

> "Generate an Anima prompt for Hatsune Miku"

The model will call `generate_image_prompt("hatsune miku")`, follow the workflow steps, and return:

```
masterpiece, best quality, absurdres, hatsune miku, vocaloid, 1girl, long hair, twintails, aqua hair, aqua eyes, detached sleeves, thighhighs, skirt
```

> "Find me a character with purple hair and purple eyes from the Fate series"

The model will call `find_character_for_scene`, extract traits, call `find_characters_by_trait`, cross-reference with `list_series_characters("fate")`, and present the top matches.

## License

MIT
