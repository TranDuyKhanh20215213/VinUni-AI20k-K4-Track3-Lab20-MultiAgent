# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline | 6.87 | 0.0003 | 7.5 |  | 0% |  |
| multi-agent | 16.16 | 0.0009 | 10.0 | 100% | 0% |  |

## Single vs multi-agent
- Multi-agent was slower by 9.29s (6.87s -> 16.16s).
- Quality score changed by +2.5 (7.5 -> 10.0).
- Cost: $0.0003 (baseline) vs $0.0009 (multi-agent).

## Traces
- baseline: `reports\trace_baseline.json`
- multi-agent: `reports\trace_multi_agent.json`
