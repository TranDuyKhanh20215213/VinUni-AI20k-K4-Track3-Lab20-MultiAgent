"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import configure_tracing_provider
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()

_BASELINE_SYSTEM_PROMPT = (
    "You are a single research assistant with no external tools. Research, analyze, and write a "
    "clear answer to the user's query from your own knowledge in one pass. Be upfront about "
    "uncertainty instead of inventing citations."
)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing_provider(settings)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline: one LLM call, no tools, no routing."""

    _init()
    request = _parse_query(query)
    state = run_baseline(request)
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))


def run_baseline(request: ResearchQuery, llm_client: LLMClient | None = None) -> ResearchState:
    """Single-agent baseline used by both the CLI and the benchmark."""

    state = ResearchState(request=request)
    client = llm_client or LLMClient()
    try:
        response = client.complete(
            _BASELINE_SYSTEM_PROMPT,
            f"Query: {request.query}\nAudience: {request.audience}",
        )
    except AgentExecutionError as exc:
        state.errors.append(f"baseline: {exc}")
        return state

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
    state.add_trace_event("baseline.run", {})
    return state


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run baseline and multi-agent on the same query and write a comparison report."""

    _init()
    _parse_query(query)  # validate early so both runners fail fast on bad input
    store = LocalArtifactStore()

    baseline_state, baseline_metrics = run_benchmark(
        "baseline", query, lambda q: run_baseline(_parse_query(q))
    )
    multi_state, multi_metrics = run_benchmark(
        "multi-agent",
        query,
        lambda q: MultiAgentWorkflow().run(ResearchState(request=_parse_query(q))),
    )

    baseline_trace = baseline_state.model_dump_json(indent=2)
    multi_trace = multi_state.model_dump_json(indent=2)
    trace_paths = {
        "baseline": str(store.write_text("trace_baseline.json", baseline_trace)),
        "multi-agent": str(store.write_text("trace_multi_agent.json", multi_trace)),
    }
    report = render_markdown_report([baseline_metrics, multi_metrics], trace_links=trace_paths)
    report_path = store.write_text("benchmark_report.md", report)

    console.print(Panel.fit(report, title="Benchmark Report"))
    console.print(f"Written to {report_path}")


if __name__ == "__main__":
    app()
