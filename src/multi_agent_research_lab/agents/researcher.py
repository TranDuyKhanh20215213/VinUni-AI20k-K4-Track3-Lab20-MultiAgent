"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM_PROMPT = (
    "You are a meticulous research assistant. You are given search results (title, source id, "
    "snippet) and must produce concise research notes for the given query. Summarize only what "
    "the sources support, and after every claim reference the source with its bracketed id, "
    "e.g. [source_id]. If the sources do not cover something, say so instead of inventing facts."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self, llm_client: LLMClient | None = None, search_client: SearchClient | None = None
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._search = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        request = state.request
        sources = self._search.search(request.query, max_results=request.max_sources)
        state.sources = sources

        if not sources:
            state.research_notes = (
                "No matching sources were found in the offline corpus for this query. "
                "Proceeding without grounded evidence."
            )
        else:
            source_block = "\n\n".join(
                f"[{doc.metadata.get('corpus_id', idx)}] {doc.title}\n{doc.snippet}"
                for idx, doc in enumerate(sources, start=1)
            )
            user_prompt = (
                f"Query: {request.query}\n"
                f"Audience: {request.audience}\n\n"
                f"Search results:\n{source_block}\n\n"
                "Write research notes (bulleted) summarizing the most relevant findings, "
                "citing sources by their bracketed id."
            )
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.research_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "source_count": len(sources),
                    },
                )
            )

        state.add_trace_event(
            "researcher.run", {"source_count": len(sources), "query": request.query}
        )
        return state
