You are improving `candidate.py`, a controller for an offline mathematical-reasoning replay environment. Use one continuous Agent session to inspect the supplied search evidence and then implement the policy improvement directly in `candidate.py`.

Work only inside the current workspace. Read `POLICY_API.md`, `SEED_POLICY.py`, `POLICY_BASE.py`, `feedback.json`, `history.json`, `TRACE_GUIDE.md`, and `TRACE_SUMMARY.json`; in personalized runs also read `REQUIREMENT_PROFILES.json`. `CANDIDATE_HISTORY/` contains proposal code from completed rounds, and `TRACE_HISTORY/` contains their complete sanitized executed traces. `SEARCH_TRACES.jsonl` is the latest completed-round trace alias. When present, `BASELINE_TRACES.jsonl` is the complete sanitized template-baseline trace. Do not spend tool calls searching outside these documented inputs.

Hard data boundary:

- Never inspect parent directories, backing outputs, private replay caches, held-out datasets, or any filesystem path outside the supplied workspace.
- Never search for question text, gold answers, raw reasoning, raw feedback, source paths, or unexecuted future checkpoints.
- The trace JSONL files contain only anonymized observations and actions actually executed by evaluated policies. They remain read-only.
- Do not load or `Read` a complete trace JSONL into model context. Use `TRACE_ANALYZER.py summary` and bounded, paginated `cases --offset N --limit M` queries for statistical analysis and targeted inspection; use aggregate summaries first.
- Do not add file, network, subprocess, reflection, dynamic import, or external private-attribute access to the candidate. Deterministic helpers defined on `CandidatePolicy` are allowed.

Trace inspection is evidence gathering, not a coverage proof. Inspect the aggregate results, then autonomously choose useful anomalous groups and representative traces. You are not required to claim that every record entered context, report `records_analyzed`, or create an analysis report. After sufficient analysis, edit `candidate.py` in this same session. Finish only after it contains one valid `CandidatePolicy`; do not merely describe a plan.

Model availability rule:

- `POLICY_API.md` and runtime `env.models` are authoritative for the complete model pool available to a policy.
- Models observed in traces only show what the incumbent or historical candidate actually spawned. Never infer that an observed subset, including only Qwen3-4B, is the available model set.

Algorithm contract:

- A branch chooses its model in `spawn(model)` and can never switch models.
- `refine(branch_id)` is legal only when the current stage is terminal.
- `env.requirements` is the current user's explicit accuracy, latency, and weighted-compute hard-requirement profile. Use all three thresholds directly to choose model, width, checkpoints, and refine behavior.
- Personalized discovery has no `beta`. Do not invent or infer a replacement scalar from the three explicit requirements.
- Public observations and action return values are dictionaries. Use bracket access such as `state["branches"]` and `branch["branch_id"]`.
- Maximize the explicit utility in `feedback.json`: the profile-weighted fraction of complete seed batches that simultaneously satisfy all three hard constraints. A proposal is accepted only when this utility strictly exceeds the incumbent utility.
- Latency follows replay cohort semantics: sequential actions add; each `*_many` phase adds the maximum cohort latency. Weighted compute includes think, probe, and feedback tokens weighted by model size.
- Base decisions only on public observations already revealed by executed actions, and keep the controller deterministic for a fixed replay seed.

Only `candidate.py` is writable. Do not edit the API, template/baseline copies, profiles, feedback, history, candidate history, analyzer, summaries, or traces. Do not claim performance absent from the supplied evidence.
