from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

RelationType = Literal["cooccurrence", "implication"]


@dataclass(frozen=True)
class TagExpansion:
    source_tag: str
    tag: str
    relation_type: RelationType
    weight: float
    provenance: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_tool_json(result: Any) -> dict[str, Any]:
    """Extract exact JSON object returned by a FastMCP text result."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("MCP tool result did not contain a JSON object")


def cooccurrence_expansions(payload: dict[str, Any], *, limit: int = 8) -> list[TagExpansion]:
    source = str(payload.get("tag", {}).get("name") or payload.get("query") or "")
    out: list[TagExpansion] = []
    for row in payload.get("related_tags") or []:
        tag = str(row.get("name") or "")
        if not tag or tag == source:
            continue
        # Preserve upstream's frequency metric. We do not invent a new
        # similarity score here.
        weight = float(row.get("frequency") or 0.0)
        out.append(TagExpansion(
            source_tag=source,
            tag=tag,
            relation_type="cooccurrence",
            weight=weight,
            provenance="booru-pictag:danbooru_search_character",
            metadata={
                "category": row.get("category"),
                "post_count": row.get("post_count"),
                "jaccard_similarity": row.get("jaccard_similarity"),
                "overlap_coefficient": row.get("overlap_coefficient"),
            },
        ))
        if len(out) >= limit:
            break
    return out


def implication_expansions(payload: dict[str, Any], *, limit: int = 8) -> list[TagExpansion]:
    source = str(payload.get("query") or "")
    out: list[TagExpansion] = []
    for row in payload.get("implications") or []:
        if row.get("status") not in (None, "active"):
            continue
        tag = str(row.get("consequent_name") or "")
        if not tag or tag == source:
            continue
        out.append(TagExpansion(
            source_tag=source,
            tag=tag,
            relation_type="implication",
            weight=1.0,
            provenance="booru-pictag:danbooru_get_tag_implications",
            metadata={"implication_id": row.get("id"), "status": row.get("status")},
        ))
        if len(out) >= limit:
            break
    return out


def expanded_preference_tags(cooccurrence: list[TagExpansion], implication: list[TagExpansion], *, limit: int = 8) -> list[str]:
    """Thin composition glue: preserve source tool ordering, dedupe names."""
    ordered = sorted(cooccurrence, key=lambda x: -x.weight) + implication
    seen: set[str] = set()
    tags: list[str] = []
    for item in ordered:
        if item.tag not in seen:
            seen.add(item.tag)
            tags.append(item.tag)
        if len(tags) >= limit:
            break
    return tags
