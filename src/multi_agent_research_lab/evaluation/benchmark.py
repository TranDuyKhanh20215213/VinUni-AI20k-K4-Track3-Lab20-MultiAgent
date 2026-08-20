"""Benchmark skeleton for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

# Words of final answer per quality point, before saturating at the citation-score cap.
_WORDS_PER_QUALITY_POINT = 60
_MAX_LENGTH_SCORE = 5.0
_MAX_CITATION_SCORE = 5.0


def _citation_coverage(state: ResearchState) -> float | None:
    if not state.sources or not state.final_answer:
        return None
    cited = sum(
        1
        for doc in state.sources
        if (corpus_id := str(doc.metadata.get("corpus_id", ""))) and corpus_id in state.final_answer
    )
    return cited / len(state.sources)


def _quality_score(state: ResearchState, citation_coverage: float | None) -> float:
    if not state.final_answer:
        return 0.0
    words = len(state.final_answer.split())
    length_score = min(words / _WORDS_PER_QUALITY_POINT, _MAX_LENGTH_SCORE)
    if citation_coverage is None:
        # No retrieved sources to cite (e.g. the single-agent baseline): neutral partial credit
        # instead of penalizing a run that was never given anything to cite.
        citation_score = _MAX_CITATION_SCORE / 2
    else:
        citation_score = _MAX_CITATION_SCORE * citation_coverage
    return round(min(length_score + citation_score, 10.0), 2)


def _estimated_cost(state: ResearchState) -> float | None:
    costs = [
        r.metadata["cost_usd"]
        for r in state.agent_results
        if r.metadata.get("cost_usd") is not None
    ]
    return round(sum(costs), 6) if costs else None


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run `runner(query)` once and score latency, cost, quality, citations, and failures."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    failed = bool(state.errors) or state.final_answer is None
    citation_coverage = _citation_coverage(state)
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimated_cost(state),
        quality_score=_quality_score(state, citation_coverage),
        citation_coverage=citation_coverage,
        failure_rate=1.0 if failed else 0.0,
        notes="; ".join(state.errors) if state.errors else "",
    )
    return state, metrics
