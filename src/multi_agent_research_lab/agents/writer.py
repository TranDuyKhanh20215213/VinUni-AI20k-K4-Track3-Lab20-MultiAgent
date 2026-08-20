"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a technical writer. Given research notes and analyst findings, synthesize a clear, "
    "well-organized answer for the stated audience. Preserve citation ids ([xyz]) inline where "
    "they support a claim, and end with a short 'Sources' list mapping each id to its title."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        if not state.research_notes:
            raise AgentExecutionError("WriterAgent requires state.research_notes to be set first.")

        sources_block = "\n".join(
            f"[{doc.metadata.get('corpus_id', idx)}] {doc.title}"
            f"{f' ({doc.url})' if doc.url else ''}"
            for idx, doc in enumerate(state.sources, start=1)
        )
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis notes:\n{state.analysis_notes or '(none)'}\n\n"
            f"Available sources:\n{sources_block or '(none)'}\n\n"
            "Write the final answer now."
        )
        response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event("writer.run", {})
        return state
