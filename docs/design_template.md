# Design Template

## Problem

The system takes a free-text research query (e.g. "When does a multi-agent architecture beat a
single agent for research tasks?") plus an audience and must produce a written answer grounded in
evidence, with citations, and no live internet access. It must also let us compare a single-agent
baseline against a multi-agent pipeline on the same query using concrete metrics (latency, cost,
quality, citation coverage, failure rate) instead of eyeballing output quality.

## Why multi-agent?

A single LLM call has to search its own memory, decide what's relevant, and write the answer in
one undifferentiated pass — there's no point where it is forced to look at retrieved evidence
before writing, and no separate step that checks the retrieved evidence for weak/conflicting
claims before those claims reach the final answer. Splitting retrieval (Researcher), critical
analysis (Analyst), and synthesis (Writer) behind an explicit Supervisor makes each step's input
and output an inspectable, testable artifact (`state.sources`, `state.research_notes`,
`state.analysis_notes`), and lets the analysis step flag weak evidence before the writer ever sees
it — something a single undifferentiated prompt has no natural place to do.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Decide the next worker (or stop) from what's already in state; enforce `max_iterations` | `ResearchState` | next route (`researcher`/`analyst`/`writer`/`done`) | Could loop forever without the iteration guardrail |
| Researcher | Retrieve sources from the offline corpus and summarize them into notes | `request.query` | `state.sources`, `state.research_notes` | No matching sources -> notes say so instead of inventing facts; LLM call failure -> caught, logged to `state.errors`, notes stay unset |
| Analyst | Extract key claims, compare viewpoints, flag weak evidence | `state.research_notes` | `state.analysis_notes` | Raises if `research_notes` missing (contract violation); LLM failure caught by the graph node wrapper |
| Writer | Synthesize the final cited answer for the audience | `research_notes`, `analysis_notes`, `sources` | `state.final_answer` | Raises if `research_notes` missing; LLM failure caught by the graph node wrapper |

## Shared state

`ResearchState` (`core/state.py`) carries:

- `request` — the validated query/audience/max_sources; the one input every agent reads.
- `iteration` / `route_history` — drives the `max_iterations` guardrail and gives a readable
  audit trail of who ran, in what order.
- `sources` — retrieved `SourceDocument`s, needed so the Writer can cite the same evidence the
  Researcher actually used (not re-derive it from prose).
- `research_notes` / `analysis_notes` / `final_answer` — the three handoff artifacts; each
  worker's precondition is "the previous field is set," which is what the Supervisor's routing
  policy checks.
- `agent_results` — per-agent `AgentResult` with token/cost metadata, used by the benchmark to
  compute `estimated_cost_usd` without re-instrumenting the LLM client.
- `trace` — lightweight `{name, payload}` events (`supervisor.route`, `researcher.run`, ...) for
  debugging without needing a tracing backend.
- `errors` — accumulated failure messages; both the stopping condition and the benchmark's
  `failure_rate` read this.

## Routing policy

Deterministic finite-state router in `agents/supervisor.py` (no LLM call, so it's cheap and
reproducible):

```text
final_answer set?              -> done
iteration >= max_iterations?   -> done (+ log error)
research_notes is None?        -> researcher
analysis_notes is None?        -> analyst
else                            -> writer
```

Workers always hand control back to the supervisor (`graph/workflow.py`), so a worker that fails
without setting its output field causes the supervisor to route to the *same* step again on the
next pass — a natural bounded retry, without any separate retry-tracking state.

## Guardrails

- **Max iterations**: `Settings.max_iterations` (default 6, from `.env`/`MAX_ITERATIONS`), enforced
  in `SupervisorAgent.run`.
- **Timeout**: `Settings.timeout_seconds` passed as the OpenAI client's per-request timeout in
  `LLMClient`.
- **Retry**: `tenacity` retries `LLMClient.complete` up to 3 times with exponential backoff on
  transient API errors (connection/timeout/status); at the workflow level, the supervisor's loop
  gives each missing step another attempt on the next iteration, bounded by max_iterations above.
- **Fallback**: `MultiAgentWorkflow._worker_node` catches `AgentExecutionError` around every
  worker, appends it to `state.errors`, and returns control to the supervisor instead of crashing
  the whole graph run.
- **Validation**: every boundary (`ResearchQuery`, `SourceDocument`, `AgentResult`,
  `BenchmarkMetrics`) is a Pydantic model, so malformed input/output fails fast with a clear error
  instead of propagating silently.

## Benchmark plan

`malab benchmark --query "..."` runs the same query through `run_baseline` (single LLM call, no
tools) and `MultiAgentWorkflow` (full pipeline), scores both with `evaluation/benchmark.py`, and
writes `reports/benchmark_report.md` + per-run trace JSON. Metrics: wall-clock latency, estimated
USD cost (from token usage), a 0-10 quality heuristic (answer length + fraction of retrieved
sources actually cited), citation coverage, and failure rate (errors or missing final answer).

Example run (`"When does a multi-agent architecture beat a single agent for research tasks?"`):

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate |
|---|---:|---:|---:|---:|---:|
| baseline | 6.87 | 0.0003 | 7.5 | — | 0% |
| multi-agent | 16.16 | 0.0009 | 10.0 | 100% | 0% |

Expected outcome, confirmed by this run: multi-agent costs more latency and tokens, but produces a
fully source-grounded answer (100% citation coverage vs. none for the baseline, which has no
retrieval step to cite from) — the trade-off the benchmark exists to make explicit rather than
asserted from a demo.

Trace evidence (LangSmith, project `multi-agent-research-lab`): `reports/tracing.png` shows four
real `LangGraph` runs from this pipeline, 16–27s latency each, matching the benchmark numbers
above.

## Failure mode and fix

**Failure mode**: a worker agent's LLM call can fail (timeout, rate limit, or the provider
returning an error after `tenacity`'s 3 retries are exhausted). Without a guard, the graph would
either crash the whole run or loop between the supervisor and the failing worker forever, since
the worker's output field would never get set.

**Where it's handled**: `MultiAgentWorkflow._worker_node` (`graph/workflow.py`) wraps every worker
call in a `try/except AgentExecutionError`. On failure it appends the error to `state.errors`,
logs a `<agent>.error` trace event, and returns control to the Supervisor instead of raising out of
the graph. The Supervisor sees the missing output field and routes back to the same worker — a
bounded retry — until either the worker succeeds or `state.iteration` reaches
`Settings.max_iterations` (default 6), at which point the Supervisor routes to `done` and records
"Stopped: max_iterations reached without a final answer." `final_answer` stays `None`, so the
benchmark's `failure_rate` correctly reports the run as failed instead of silently returning a
partial or fabricated answer.

**Verified by**: `tests/test_workflow.py::test_workflow_stops_at_max_iterations_on_repeated_failure`,
which wires in a Researcher stub that always raises and asserts the workflow still terminates
within `max_iterations + 1` steps with the failure recorded in `state.errors`.
