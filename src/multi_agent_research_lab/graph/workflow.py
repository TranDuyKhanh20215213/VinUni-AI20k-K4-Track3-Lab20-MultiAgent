"""LangGraph workflow skeleton."""

from collections.abc import Callable

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState

_SUPERVISOR = "supervisor"

_CompiledGraph = CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
    ) -> None:
        self._supervisor = supervisor or SupervisorAgent()
        self._researcher = researcher or ResearcherAgent()
        self._analyst = analyst or AnalystAgent()
        self._writer = writer or WriterAgent()
        self._compiled: _CompiledGraph | None = None

    @staticmethod
    def _worker_node(agent: BaseAgent) -> Callable[[ResearchState], ResearchState]:
        def node(state: ResearchState) -> ResearchState:
            try:
                return agent.run(state)
            except AgentExecutionError as exc:
                state.errors.append(f"{agent.name}: {exc}")
                state.add_trace_event(f"{agent.name}.error", {"message": str(exc)})
                return state

        return node

    @staticmethod
    def _route_after_supervisor(state: ResearchState) -> str:
        return state.route_history[-1] if state.route_history else ROUTE_DONE

    def build(self) -> _CompiledGraph:
        """Create a LangGraph graph.

        Nodes: supervisor, researcher, analyst, writer. The supervisor decides the next route
        via a conditional edge; workers always hand control back to the supervisor so it can
        re-evaluate state (and enforce `max_iterations`) after every step.
        """

        graph: StateGraph[ResearchState, None, ResearchState, ResearchState] = StateGraph(
            ResearchState
        )
        graph.add_node(_SUPERVISOR, self._supervisor.run, input_schema=ResearchState)
        # mypy fails to structurally match a plain closure against langgraph's `_Node` Protocol
        # union here (a bound method on the same line above checks fine); verified correct at
        # runtime via `cli.py multi-agent`.
        graph.add_node(
            ROUTE_RESEARCHER,
            self._worker_node(self._researcher),  # type: ignore[arg-type]
            input_schema=ResearchState,
        )
        graph.add_node(
            ROUTE_ANALYST,
            self._worker_node(self._analyst),  # type: ignore[arg-type]
            input_schema=ResearchState,
        )
        graph.add_node(
            ROUTE_WRITER,
            self._worker_node(self._writer),  # type: ignore[arg-type]
            input_schema=ResearchState,
        )

        graph.set_entry_point(_SUPERVISOR)
        graph.add_conditional_edges(
            _SUPERVISOR,
            self._route_after_supervisor,
            {
                ROUTE_RESEARCHER: ROUTE_RESEARCHER,
                ROUTE_ANALYST: ROUTE_ANALYST,
                ROUTE_WRITER: ROUTE_WRITER,
                ROUTE_DONE: END,
            },
        )
        for worker in (ROUTE_RESEARCHER, ROUTE_ANALYST, ROUTE_WRITER):
            graph.add_edge(worker, _SUPERVISOR)

        self._compiled = graph.compile()
        return self._compiled

    def run(self, state: ResearchState) -> ResearchState:
        """Compile (if needed), invoke the graph, and convert the result back to ResearchState."""

        compiled = self._compiled or self.build()
        result = compiled.invoke(state, config={"recursion_limit": 50})
        return ResearchState(**result)
