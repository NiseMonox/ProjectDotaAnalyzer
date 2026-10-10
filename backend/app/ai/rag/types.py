"""Plain data types of the retrieval stack (standard library only)."""
from __future__ import annotations

from dataclasses import dataclass, field

# Where a knowledge chunk comes from, as shown to the user next to a verdict.
SOURCE_TYPES: tuple[str, ...] = (
    "editorial",        # curated hero / matchup / ability / item notes (backend/data/editorial)
    "builtin",          # coaching principles written in knowledge_base / knowledge_chunks
    "operator",         # operator files in backend/data/knowledge
    "opendota_stats",   # learned: OpenDota aggregate statistics
    "match_mining",     # learned: statistics mined from pro matches
    "llm_distilled",    # learned: LLM summary of match statistics
    "video",            # learned: tips distilled from community videos
)
# Third-party prose: it never ranks as critical (priority 1), a block holds at most two such
# entries (knowledge_store) and its title never enters a prompt.
UNTRUSTED_SOURCE_TYPES: frozenset[str] = frozenset({"llm_distilled", "video"})


@dataclass(frozen=True)
class KnowledgeRef:
    """One numbered entry of a knowledge block (``[K1]`` ...).

    ``ref`` is unique within one evaluation (or one agent run); ``title`` and
    ``source_label`` are in the report language.
    """

    ref: str                # "K1"
    chunk_id: str
    title: str
    source_type: str        # one of SOURCE_TYPES
    source_label: str       # localized, e.g. "Curated note" / "编辑整理"
    patch: str = ""         # "" = not patch specific
    score: float = 0.0      # ranking score the entry was selected with
    trusted: bool = True

    def to_payload(self, cited: bool = False) -> dict:
        """The ``sources[]`` item of the report payload."""
        return {
            "id": self.chunk_id,
            "title": self.title,
            "source_type": self.source_type,
            "patch": self.patch,
            "cited": bool(cited),
        }


@dataclass(frozen=True)
class SituationQuery:
    """The closed-set English query of a decision: ``"<role>, <phase>: <topic>"``."""

    kind: str               # topic code, e.g. "death_tf"
    text: str


@dataclass
class SearchResult:
    """Outcome of a free-text knowledge search (:func:`rag.search`)."""

    hits: list = field(default_factory=list)        # [(KnowledgeChunk, score)], best first
    mode: str = "lexical"                           # "hybrid" | "lexical"
    reason: str = ""                                # why the search was lexical ("" when hybrid)
    heroes: list[str] = field(default_factory=list)          # hero tags the search was scoped to
    ignored_heroes: list[str] = field(default_factory=list)  # named, but not in this match
    items: list[str] = field(default_factory=list)           # item tags the search was scoped to
    already_shown: list[str] = field(default_factory=list)   # chunk ids found again that the caller had shown


__all__ = ["KnowledgeRef", "SearchResult", "SituationQuery", "SOURCE_TYPES", "UNTRUSTED_SOURCE_TYPES"]
