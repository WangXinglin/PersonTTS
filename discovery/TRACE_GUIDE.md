# Sanitized Search Trace Guide

The discovery workspace exposes complete sanitized executed traces from the baseline and completed search rounds. They contain public controller observations only, never question text, gold answers, raw reasoning, raw feedback, backing paths, or unexecuted future checkpoints.

Files:

- `SEARCH_TRACES.jsonl`: latest completed-round trace, or empty before round 0.
- `TRACE_SUMMARY.json`: aggregate statistics for `SEARCH_TRACES.jsonl`.
- `TRACE_HISTORY/round_XX.jsonl`: complete traces for completed rounds.
- `TRACE_HISTORY/round_XX.summary.json`: their aggregate statistics.
- `BASELINE_TRACES.jsonl` and `BASELINE_TRACE_SUMMARY.json`: complete template-baseline evidence when available.
- `CANDIDATE_HISTORY/round_XX.py`: proposal code corresponding to historical round traces.
- `TRACE_ANALYZER.py`: bounded statistics and targeted case-query CLI.

Start from aggregates and then choose useful directions to inspect. Do not directly load a complete JSONL into context:

```bash
python3 TRACE_ANALYZER.py summary
python3 TRACE_ANALYZER.py --source BASELINE_TRACES.jsonl summary
python3 TRACE_ANALYZER.py --source TRACE_HISTORY/round_00.jsonl summary
```

Targeted queries return at most 50 records per page and retain at most 40 representative steps per trace. Use `--offset` to inspect later matching records; this is bounded pagination, not a full JSONL context load:

```bash
python3 TRACE_ANALYZER.py cases --profile profile_0000 --correct false --limit 10
python3 TRACE_ANALYZER.py cases --dataset AIME_24 --action REFINE --limit 10
python3 TRACE_ANALYZER.py cases --model Qwen3-14B --sort weighted-desc --limit 10
python3 TRACE_ANALYZER.py --source TRACE_HISTORY/round_00.jsonl cases --sort latency-desc --limit 10
```

Use the commands autonomously to investigate failure groups, resource tradeoffs, routing, pruning, and refinement. There is no requirement to inspect every record individually or prove full coverage. Models seen in traces are executed-policy behavior, not the available model set; `POLICY_API.md` and `env.models` are authoritative.
