"""Search client abstraction for ResearcherAgent.

Backed by the offline `ai_agent_offline_research_corpus_v2` corpus shipped in this repo instead
of a live web API: it is deterministic, has no rate limits, and is scoped to the multi-agent
systems topics this lab is about.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from multi_agent_research_lab.core.schemas import SourceDocument

_CORPUS_DIR_NAME = "ai_agent_offline_research_corpus_v2"
_WORD_RE = re.compile(r"[a-z0-9]+")
_SNIPPET_CHARS = 500


@dataclass(frozen=True)
class _CorpusDocument:
    doc_id: str
    title: str
    url: str | None
    text: str
    topic: str
    tokens: set[str] = field(default_factory=set)


def _find_corpus_dir() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _CORPUS_DIR_NAME
        if candidate.is_dir():
            return candidate
    return None


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[_CorpusDocument, ...]:
    corpus_dir = _find_corpus_dir()
    if corpus_dir is None:
        return ()

    documents: list[_CorpusDocument] = []
    for topic_file in sorted((corpus_dir / "topics").glob("*.json")):
        payload = json.loads(topic_file.read_text(encoding="utf-8"))
        topic_title = payload.get("topic", {}).get("name", topic_file.stem)
        knowledge_base = payload.get("knowledge_base", {})

        for article in knowledge_base.get("knowledge_articles", []):
            text = article.get("content", "")
            documents.append(
                _CorpusDocument(
                    doc_id=article["article_id"],
                    title=f"{topic_title} — {article.get('title', article['article_id'])}",
                    url=None,
                    text=text,
                    topic=topic_title,
                    tokens=_tokenize(f"{article.get('title', '')} {text}"),
                )
            )

        for source in knowledge_base.get("source_documents", []):
            text = source.get("full_text", "")
            documents.append(
                _CorpusDocument(
                    doc_id=source["document_id"],
                    title=source.get("title", source["document_id"]),
                    url=source.get("provenance_url"),
                    text=text,
                    topic=topic_title,
                    tokens=_tokenize(f"{source.get('title', '')} {text}"),
                )
            )

    return tuple(documents)


def _snippet(text: str, query_tokens: set[str]) -> str:
    lowered = text.lower()
    best_pos = 0
    for token in query_tokens:
        pos = lowered.find(token)
        if pos != -1:
            best_pos = max(pos - 100, 0)
            break
    excerpt = text[best_pos : best_pos + _SNIPPET_CHARS].strip()
    return excerpt + ("..." if best_pos + _SNIPPET_CHARS < len(text) else "")


class SearchClient:
    """Keyword search over the offline multi-agent-systems research corpus."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return the top-matching corpus documents for `query`, ranked by keyword overlap."""

        corpus = _load_corpus()
        if not corpus:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for doc in corpus:
            overlap = query_tokens & doc.tokens
            if not overlap:
                continue
            title_bonus = len(query_tokens & _tokenize(doc.title)) * 2
            score = len(overlap) + title_bonus
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[SourceDocument] = []
        for score, doc in scored[:max_results]:
            results.append(
                SourceDocument(
                    title=doc.title,
                    url=doc.url,
                    snippet=_snippet(doc.text, query_tokens),
                    metadata={"corpus_id": doc.doc_id, "topic": doc.topic, "score": score},
                )
            )
        return results
