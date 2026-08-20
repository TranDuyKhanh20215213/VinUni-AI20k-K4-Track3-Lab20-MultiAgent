"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(
    metrics: list[BenchmarkMetrics], trace_links: dict[str, str] | None = None
) -> str:
    """Render benchmark metrics to markdown, with an optional single/multi-agent comparison."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    by_name = {item.run_name: item for item in metrics}
    if "baseline" in by_name and "multi-agent" in by_name:
        base, multi = by_name["baseline"], by_name["multi-agent"]
        lines += ["", "## Single vs multi-agent"]
        latency_delta = multi.latency_seconds - base.latency_seconds
        direction = "slower" if latency_delta >= 0 else "faster"
        lines.append(
            f"- Multi-agent was {direction} by {abs(latency_delta):.2f}s "
            f"({base.latency_seconds:.2f}s -> {multi.latency_seconds:.2f}s)."
        )
        if base.quality_score is not None and multi.quality_score is not None:
            quality_delta = multi.quality_score - base.quality_score
            lines.append(
                f"- Quality score changed by {quality_delta:+.1f} "
                f"({base.quality_score:.1f} -> {multi.quality_score:.1f})."
            )
        if base.estimated_cost_usd is not None and multi.estimated_cost_usd is not None:
            lines.append(
                f"- Cost: ${base.estimated_cost_usd:.4f} (baseline) vs "
                f"${multi.estimated_cost_usd:.4f} (multi-agent)."
            )

    if trace_links:
        lines += ["", "## Traces"]
        lines += [f"- {name}: `{path}`" for name, path in trace_links.items()]

    return "\n".join(lines) + "\n"
