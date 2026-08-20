from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


class _AlwaysFailResearcher(BaseAgent):
    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        raise AgentExecutionError("boom")


def test_workflow_stops_at_max_iterations_on_repeated_failure() -> None:
    workflow = MultiAgentWorkflow(researcher=_AlwaysFailResearcher())
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    result = workflow.run(state)

    assert result.final_answer is None
    assert result.errors
    assert any("researcher: boom" in error for error in result.errors)
    settings = get_settings()
    assert result.iteration <= settings.max_iterations + 1
