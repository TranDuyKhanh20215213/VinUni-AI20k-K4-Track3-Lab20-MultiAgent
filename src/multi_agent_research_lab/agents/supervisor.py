"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Deterministic policy: fill in whichever field is still missing (research_notes ->
    analysis_notes -> final_answer), retrying a step if a previous worker failed, and always
    stopping once `max_iterations` is reached so a stuck workflow cannot loop forever.
    """

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""

        settings = get_settings()

        if state.final_answer is not None:
            route = ROUTE_DONE
        elif state.iteration >= settings.max_iterations:
            state.errors.append("Stopped: max_iterations reached without a final answer.")
            route = ROUTE_DONE
        elif state.research_notes is None:
            route = ROUTE_RESEARCHER
        elif state.analysis_notes is None:
            route = ROUTE_ANALYST
        else:
            route = ROUTE_WRITER

        state.record_route(route)
        state.add_trace_event("supervisor.route", {"next": route, "iteration": state.iteration})
        return state
