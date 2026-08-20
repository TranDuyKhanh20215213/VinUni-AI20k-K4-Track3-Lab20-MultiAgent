from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_through_pipeline_then_stops() -> None:
    state = _state()
    supervisor = SupervisorAgent()

    supervisor.run(state)
    assert state.route_history[-1] == ROUTE_RESEARCHER

    state.research_notes = "notes"
    supervisor.run(state)
    assert state.route_history[-1] == ROUTE_ANALYST

    state.analysis_notes = "analysis"
    supervisor.run(state)
    assert state.route_history[-1] == ROUTE_WRITER

    state.final_answer = "answer"
    supervisor.run(state)
    assert state.route_history[-1] == ROUTE_DONE
    assert state.iteration == 4


def test_supervisor_retries_a_missing_step() -> None:
    state = _state()
    supervisor = SupervisorAgent()

    supervisor.run(state)
    supervisor.run(state)  # research_notes still None -> route researcher again

    assert state.route_history == [ROUTE_RESEARCHER, ROUTE_RESEARCHER]


def test_supervisor_stops_at_max_iterations_without_success() -> None:
    settings = get_settings()
    state = _state()
    state.iteration = settings.max_iterations
    supervisor = SupervisorAgent()

    supervisor.run(state)

    assert state.route_history[-1] == ROUTE_DONE
    assert any("max_iterations" in error for error in state.errors)
