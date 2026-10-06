# Public policy API

`CandidatePolicy(config={"requirements": profile})` must expose `solve(self, env)` and return an observed answer or `None`. Personalized runs expose only explicit requirements and resources; legacy beta-grid runs separately receive `beta`.

The policy may use:

- `env.models`: the authoritative complete set of model names available in the current episode. Models observed in baseline or historical traces only describe what those policies spawned and must never be used to narrow this set.
- `env.beta`: available only in legacy beta-grid evaluation. Personalized discovery has no beta field in the environment, observation, or policy config.
- `env.requirements`: an isolated copy of the current profile, containing exactly one hard constraint for accuracy, latency, and weighted compute. The canonical `constraints` list is preserved, and each constraint is also available directly by metric name (for example, `env.requirements["latency_s"]["value"]`). It is `None` only in legacy beta evaluation.
- `env.observe()`: current public state with model pool counts, opened branches, and charged cost.
- `env.spawn(model)`: create one model-bound branch and reveal checkpoint 0.
- `env.spawn_many(model, count)`: atomically open a fixed-model cohort.
- `env.continue_branch(branch_id)`: reveal the next checkpoint in the current stage.
- `env.continue_many(branch_ids)`: atomically advance one synchronous checkpoint round.
- `env.refine(branch_id)`: after a terminal stage, charge feedback tokens and enter the next stage without revealing its first checkpoint.
- `env.refine_many(branch_ids)`: atomically refine a survivor cohort after every member is terminal.
- `env.can_afford_complete_candidates(...)`: report only whether a complete configured candidate/window fits the remaining pool and weighted-compute budget.
- `env.prune(branch_id)`: deactivate a branch.
- `env.finish(answer=None)`: terminate using an observed answer, or use deterministic plurality when omitted.

`env.observe()` and every state-changing action return Python dictionaries. Use bracket access for returned fields.

```python
state = env.observe()
for branch in state["branches"]:
    if branch["can_continue"]:
        env.continue_branch(branch["branch_id"])

opened = env.spawn_many(env.models[0], 4)
env.continue_many(
    [branch["branch_id"] for branch in opened if branch["can_continue"]]
)
```

Each public branch dictionary contains `branch_id`, `model`, `stage_index`, `stage_kind`, `checkpoint_index`, `latest_answer`, `answer_history`, `stage_terminal`, `can_continue`, `can_refine`, and `pruned`. No question, gold answer, raw generation, raw feedback, source path, future answer, or future token delta is public.

In personalized discovery, `REQUIREMENT_PROFILES.json` is the immutable profile population. Each seed evaluates one complete configured discovery-problem batch for every profile. A batch scores 1 only when all three constraints are satisfied; the discovery utility is the profile-weighted batch satisfaction rate. There is no compatibility beta in this mode: use the three thresholds directly.

`SEED_POLICY.py` contains the adapted AutoTTS baselines `ASCConfigured4D`, `ESCConfigured4D`, and `ParallelProbeSR`. Their standard evaluation uses an explicit fixed `ResourceProfile`; it does not map `beta` to a hand-authored model/width/depth/stage schedule.

Allowed imports are `collections`, `math`, `statistics`, `typing`, `policies.base`, and `policies.seeds`. Candidate validation rejects I/O, subprocesses, networking, dynamic imports, reflection, external private attributes, and forbidden data references. A constructor may call `super().__init__(...)`; deterministic private helpers defined on `CandidatePolicy` are also allowed.
