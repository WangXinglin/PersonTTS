# Strategy Guide — Controller Design from Workspace Evidence

Design `candidate.py` by calibrating parameters from workspace data.

Read this **before** analyzing traces. Workflow: (1) compute calibration values during
analysis, (2) write them as literals into `candidate.py` (no file I/O allowed in the candidate).

---

## Part A — Environment Contracts

### A1. Parallel vs Sequential Latency

Both `spawn_many` and `continue_many` charge latency as
`max(think_elapsed) + max(probe_elapsed)` across the cohort — two separate
max-reductions per call. Sequential `spawn()` / `continue_branch()` each charge
`think_elapsed + probe_elapsed` per branch, so N sequential calls cost
N × (think + probe), which is much more than one parallel call.

Weighted compute is always `model_weight × raw_tokens`, **summed** across all
branches regardless of parallel or sequential execution.

**Rule**: Use `spawn_many` for initial branches and `continue_many` for extensions.
Never loop over `spawn` or `continue_branch` unless you need exactly 1 branch.

### A2. Branch Completion and Voting

A branch with `can_continue=False` has finished its reasoning chain. Its answer
is typically more reliable than an intermediate answer from a branch still computing.

**Default approach**: Continue active branches via `continue_many` until none remain,
then vote on final answers.

```python
for _ in range(loop_limit):
    state = env.observe()
    active = [b["branch_id"] for b in state["branches"]
              if b["can_continue"] and not b["pruned"]]
    if not active:
        break
    env.continue_many(active)
```

**When to stop before all branches finish**:
- Latency budget cannot support further rounds (see C7).
- Most branches have terminated with consistent answers and the remaining
  branches are unlikely to change the plurality vote.
- In either case, vote on whatever answers are available. Verify by checking
  whether joint satisfaction_rate improves compared to running to completion.

### A3. Deterministic Plurality for Voting

```python
from policies.base import deterministic_plurality, usable_answers
answers = usable_answers(state["branches"])
return env.finish(deterministic_plurality(answers) if answers else None)
```

### A4. Read All Constraint Values from env.requirements

```python
reqs = env.requirements
acc_target = reqs["accuracy"]["value"]       # >= threshold
lat_budget = reqs["latency_s"]["value"]      # <= threshold
comp_budget = reqs["weighted_compute"]["value"]  # <= threshold
```

**Aggregation per constraint** (how each constraint is computed across seed batches):
- accuracy: **mean** of per-batch accuracy. A single batch with low accuracy drags down the mean.
- latency_s: **batch** of per-batch latency. Only the slowest ~5% of batches need to exceed budget for this to fail.
- weighted_compute: **mean** of per-batch compute.

**Aggregation of joint satisfaction**: a seed batch is "satisfied" when all three
constraints pass simultaneously. Satisfaction_rate = fraction of satisfied batches.

### A5. Satisfaction is All-or-Nothing per Seed Batch

A seed batch counts as "satisfied" only when ALL THREE constraints pass simultaneously.
Joint satisfaction_rate is the optimization target.

Joint satisfaction_rate is generally **less than or equal to** the minimum of
per-constraint satisfaction rates, because different batches can fail on different
constraints. For example: accuracy_sat=0.877, compute_sat=0.993, but
joint_sat=0.874 — some batches fail accuracy while others fail compute, and the
union of failed batches is larger than either alone. (Verified: 34/1000 rounds show
this gap in data.)

**Acceptance**: strict `>` (proposal must beat incumbent; ties rejected).
When incumbent is already at satisfaction_rate = 1.0, any proposal with the same
score is rejected. This is the normal plateau behavior after finding a satisfying
policy, not a policy failure.

### A6. Candidate Restrictions

`candidate.py` cannot do file I/O, subprocess calls, reflection, dynamic imports,
or access private attributes. Allowed imports: `collections`, `math`, `statistics`,
`typing`, `policies.base`, `policies.seeds`. All data must be baked as literals.

---

## Part B — Calibration

### B1. Latency Estimate

From `BASELINE_TRACE_SUMMARY.json` (or `TRACE_SUMMARY.json` after a round):

```python
mean_phase_lat = summary["all"]["avg_latency_s"] / summary["all"]["avg_trace_steps_total"]
```

This is the mean per-action latency, averaged over all spawn and continue actions
in the baseline policy. Two things this does NOT tell you:

1. **Not a batch estimate**: The scorer checks `batch(latency) <= threshold`. The mean
   underestimates the batch. Applying a safety factor (e.g., using 70-80% of the budget)
   is a reasonable heuristic to account for this, but does not guarantee the batch will
   be within budget.

2. **Not the cost of your policy**: The baseline policy (ASCConfigured4D, width=40)
   produces different per-action latencies than your policy. The mean_phase_lat is
   a starting point; after your first round, recalibrate from your own TRACE_SUMMARY.

**Per-action cost structure** (same for spawn_many and continue_many):
Each parallel call charges `max(think_elapsed) + max(probe_elapsed)` across the
cohort. The max of N branches is greater than the per-branch mean; how much greater
depends on variance. There is no simple formula to predict this from the baseline
mean — treat it as an underestimate.

### B2. Token and Compute Estimates

```python
avg_continues = summary["all"]["action_counts"]["CONTINUE"] / summary["all"]["episodes"]
tokens_per_continue = summary["all"]["avg_raw_tokens"] / max(1, avg_continues)
```

`tokens_per_continue` is a rough average. Actual token counts vary by checkpoint
position and model. Spawn checkpoints and continue checkpoints may have different
token counts. Treat this as an order-of-magnitude estimate.

### B3. Model Weights

| Model | Weight |
|---|---|
| Qwen3-0.6B | 1.0 |
| Qwen3-1.7B | 2.833 |
| Qwen3-4B | 6.667 |
| Qwen3-8B | 13.333 |
| Qwen3-14B | 23.333 |
| Qwen3-32B | 53.333 |

### B4. Budget Feasibility Check

Given a model choice and the rough estimates above, check whether a
`(width, loop_limit)` pair plausibly fits within both budgets.

`loop_limit` is the maximum number of `continue_many` calls your code allows.
Actual execution depth may be less if branches reach terminal early.

```python
# Rough estimates — underestimates likely, treat as lower bounds on cost:
est_actions = 1 + loop_limit  # 1 spawn_many + loop_limit continue_many calls
est_latency = est_actions * mean_phase_lat
est_compute = width * est_actions * tokens_per_continue * model_weight
```

**Check a few combos at the current model before switching**:
```python
combos = []
for w in [1, 3, 5]:
    for ll in [max_loop, max_loop // 2, max_loop // 4]:
        if ll < 0:
            continue
        lat = (1 + ll) * mean_phase_lat
        comp = w * (1 + ll) * tokens_per_continue * model_weight
        if lat < lat_budget and comp < comp_budget:
            combos.append((w, ll, lat, comp))
```
If no combo fits at this model, try a lighter model.

**Edge cases**:
- If `tokens_per_continue` or `mean_phase_lat` is zero or missing (no baseline
  data), you cannot estimate costs. Use conservative defaults: width=1,
  loop_limit=5, default model (Qwen3-4B). Recalibrate after the first round.
- `loop_limit = 0` means spawn-only (no continue calls). This is valid — you get
  one checkpoint per branch. This costs 1 × width × tokens_per_continue ×
  model_weight in compute and 1 × mean_phase_lat in latency.
- Do not use `max(1, ...)` to force a feasible result. If `loop_limit` computes to
  negative, the budget is too tight for this model. Report infeasibility and try
  a lighter model.

**Rough estimates, not guarantees**: These calculations use means where the scorer
uses batch (for latency), and ignore per-action variance. They are useful for
narrowing the design space and catching obviously infeasible designs, not for
certifying that a design will pass evaluation.

### B5. After a Failed Round: Read Feedback

Read `feedback.json` for per-constraint diagnostics:
```python
# feedback.json -> summary -> profiles -> <profile_id> -> constraints
# Each constraint has: satisfaction_rate, mean_margin
# satisfaction_rate: fraction of seed batches where THIS constraint passed
# mean_margin: average headroom/shortfall across batches
#   margin > 0: on average, constraint is satisfied with room to spare
#   margin < 0: on average, constraint is violated
```

**mean_margin > 0 does NOT mean all batches passed.** The mean can be positive
while some batches have negative margin (and thus fail). Check satisfaction_rate,
not just mean_margin.

Recalibrate estimates (B1-B2) when your actual costs from `TRACE_SUMMARY.json`
significantly deviate from the baseline-derived estimates. If actual costs matched
estimates, the issue is in parameter choices, not calibration — no need to
recalibrate everything.

---

## Part C — Conditional Search Heuristics

Format: **trigger** → **evidence** → **action** → **verify joint satisfaction_rate**
→ **exceptions/when this doesn't apply**.

Evidence strength:
- **MECHANISM**: from cost model / scorer physics. Always applies.
- **OBSERVATION**: from data (N >= 30), consistent direction, but confounded.
- **HYPOTHESIS**: small samples (N < 30), mixed direction, or heavily confounded.
  Treat as diagnostic suggestion only.
- **INCONCLUSIVE**: data shows no clear direction.

### C1. Check Accuracy Before Resource Tuning [HEURISTIC]

**Trigger**: Planning initial strategy.

**Evidence**: The supplied feedback reports a satisfaction rate and mean margin
for each constraint. These current-run diagnostics are the authoritative evidence
for deciding which constraint is binding.

**Action**: Unless the latency or compute budget is very tight, ensure branches have
enough depth to produce valid answers and enough width for vote diversity.

**Verify**: After a round, read per-constraint satisfaction rates. If accuracy_sat
is the lowest, this diagnosis holds. If latency_sat or compute_sat is lower, the
bottleneck is resource-side.

**Exception**: Profiles with very tight latency or compute budgets may be
resource-constrained. In those cases, follow the corresponding resource-side
diagnostics instead of assuming accuracy is the bottleneck.

### C2. Depth Calibration is More Often Productive Than Width [HYPOTHESIS]

**Trigger**: Deciding whether to adjust depth or width after a failure.

**Evidence (E4.6, E7.3)**: Among 44 transitions where only the extracted depth
feature changed:
- 14 depth increased: 5 improved, 2 regressed (mean Dsat = +0.212)
- 6 depth decreased: 4 improved, 1 regressed (mean Dsat = +0.217)
- 24 unclear direction (AST extraction gap — actual depth unknown)
- Width changes: 78 transitions, 20 improve vs 14 regress (mean Dsat = +0.015)

Both increasing and decreasing depth can improve satisfaction.

**Traceable cases**:
- aime/profile_0017 S0 r0→r1: depth [3]→[3,12], Dsat=+1.000 — branches too shallow
  at depth=3, increasing to 12 allowed terminal answers.
- aime/profile_0039 S1 r1→r2: depth [25]→[15], Dsat=+1.000 — depth 25 over-allocated,
  reducing to 15 freed resources while maintaining answer quality.
- aime/profile_0008 S0 r2→r3: depth [36]→[48], Dsat=-0.441 — increasing depth consumed
  more budget without improving accuracy.
- aime/profile_0029 S1 r1→r2: depth [5]→[4], Dsat=-1.000 — one fewer round broke
  answer quality entirely.

**Action**: When a round fails, consider adjusting depth:
1. **First check**: Did branches actually reach the loop_limit? Read TRACE_SUMMARY
   or observe whether branches had `can_continue=True` when the loop ended.
   - If most branches terminated well before loop_limit: reducing loop_limit will
     NOT save latency or compute. The unused headroom costs nothing.
   - If branches were still running at loop_limit: they may need more depth for
     answer quality. Consider increasing loop_limit if latency allows.
2. **Second check**: If loop_limit was reached and accuracy is the binding constraint,
   increasing depth may help — but verify the increased cost fits the budget (B4).
3. **If resources are the bottleneck**: reducing depth is a valid option, but it
   changes reasoning behavior. Verify joint satisfaction_rate after the change.

**This is not a universal "prefer deeper" rule.** The benefit comes from matching
depth to the problem and budget, not from maximizing depth.

### C3. Width Scales with Accuracy Target — Weakly [HYPOTHESIS]

**Trigger**: Deciding how many branches to spawn.

**Evidence**: Width 11-20 shows highest satisfaction (0.751, 26 profiles), but width
21+ drops to 0.624 (15 profiles). Width single-feature-change transitions show
near-zero net effect (Dsat = +0.015, N=78). Width matters less than depth for
accuracy in this dataset.

**Action**: Start with the width from B4 (budget-derived). Width primarily helps
through vote diversity; 3-8 branches are typically sufficient. If accuracy target
is very high (>= 0.7) and compute budget allows, consider more width.

### C4. Early Stopping: No Default Recommendation [INCONCLUSIVE]

**Trigger**: Considering whether to add or remove confidence-based early exit.

**Evidence (E4.3, E7.3)**: Both adding and removing early stopping show mixed
results. N=8 and N=10 for single-feature-change transitions — too small to
conclude anything. Paired comparison (96 profiles): 34 improved vs 47 regressed,
near even.

**Action**: Decide based on your specific situation:
- **Tight latency, branches running long**: Adding early stopping may reduce latency
  at the cost of answer quality. Test it.
- **Failed on accuracy, branches terminating early**: Removing early stopping or
  raising its threshold may let more branches reach terminal.
- **Default**: Omit on the first round. Adjust based on feedback.
- Early stopping within an episode (stopping branches early) is a different concept
  from stopping the outer discovery loop early. Only the former is relevant here.

**Verify**: Check **joint** satisfaction_rate after any change. A change that lowers
one per-constraint sat but raises another can still improve joint sat — individual
per-constraint drops do not automatically make the change wrong.

### C5. Refine: Budget-Dependent [INCONCLUSIVE]

**Trigger**: Considering `env.refine()` / `env.refine_many()`.

**Evidence (E4.4, E7.3)**: Single-feature-change data is sparse (refine:add N=12,
refine:remove N=5). Both directions show positive mean Dsat, which means neither
direction has a reliable signal. Refine proposals have higher mean satisfaction
(0.700 vs 0.655), but this is confounded by incumbent level.

**Action**: Refine charges additional tokens and latency. If both budgets have
headroom after the main spawn-continue cycle, refine may help branches that
reached terminal with wrong answers. If latency or compute is tight, skip it.

**Verify**: Check **joint** satisfaction_rate. A per-constraint satisfaction drop
on latency or compute does not by itself mean refine was harmful — the joint
rate may still improve if accuracy gains outweigh resource costs.

### C6. Tight Compute Budget -> Smaller Model or Width=1 [MECHANISM]

**Trigger**: Compute budget < 100,000.

**Evidence**: At compute < 10K with Qwen3-4B (weight=6.667), a single branch with
1 checkpoint costs ~6,667. Two branches cost ~13K. Width=1 may be forced by
arithmetic. At 10K-100K, width <= 5 with lighter models (E4.8).

**Action**: Try a few (width, loop_limit) combos at the current model (B4). If
none fit, try a lighter model. At very tight budgets, width=1 with maximum
affordable depth is the fallback.

### C7. Tight Latency Budget -> Fewer Continue Rounds [MECHANISM]

**Trigger**: Latency budget < 50s.

**Evidence**: At ~8-12s per parallel action, only ~3-5 actions fit in 50s.
Accepted policies with lat < 50s have median depth=3 (E4.7).

**Action**: With very tight latency, branches will likely not all reach terminal.
Vote on whatever answers are available at the loop limit. Width can be higher
since compute budget is freed by low depth.

### C8. Diagnose Before Adjusting [MECHANISM]

**Trigger**: A round failed.

**Evidence**: Among 800 transitions, 53.4% show flat satisfaction, 31.8% improve,
14.9% regress (E7.1). Over-correction is a real risk.

**Action**:
1. Read `feedback.json` per-constraint satisfaction rates. Identify which
   constraint(s) have sat < 1.0. The constraint with the lowest sat is a useful
   **starting point** for diagnosis — but different seed batches may fail on
   different constraints, and joint_sat can be lower than any individual
   constraint's sat (A5).
2. Read margins. Small negative margin: fine-tune loop_limit or width by 1.
   Large negative margin: recalibrate estimates from TRACE_SUMMARY (B5).
3. Do NOT assume fixing the lowest-sat constraint will fix joint satisfaction.
   Verify that the adjustment doesn't worsen another constraint enough to
   offset the gain. Re-run budget feasibility (B4) after adjusting.
4. Avoid structural overhaul for small gaps. Incremental changes are the most
   common pattern in improving transitions (E7.2).

**Verify**: Compare joint satisfaction_rate before and after, not individual
constraint rates in isolation.

### C9. When One Constraint Has Headroom [MECHANISM]

**Trigger**: One constraint passes comfortably while another doesn't.

**Action**: Headroom on one constraint suggests resources **might** be
reallocatable, but verify before committing:
- **Accuracy headroom**: Reducing depth or width to free latency/compute is risky.
  The relationship between depth and accuracy is non-linear — a branch that was
  just barely reaching terminal may produce no valid answer if depth is reduced
  by even 1-2 rounds.
- **Latency headroom**: Increasing loop_limit by 1-2 to improve accuracy is
  relatively safe — latency cost increases roughly linearly.
- **Compute headroom**: Adding 1-2 branches for vote diversity has a predictable
  cost increase.

**Verify**: After any reallocation, check joint satisfaction_rate. If the
constraint that had headroom now fails, the reallocation was too aggressive.

---

## Decision Flowchart

```
1. READ BASELINE_TRACE_SUMMARY.json (or TRACE_SUMMARY.json after round 1+)
   -> mean_phase_lat, tokens_per_continue, model_weight
   (If missing or zero: use defaults width=1, loop_limit=5, Qwen3-4B)

2. READ env.requirements
   -> acc_target, lat_budget (batch), comp_budget (mean)

3. TRY a few (width, loop_limit) combos at the current model:
   for each combo:
     est_latency = (1 + loop_limit) * mean_phase_lat
     est_compute = width * (1 + loop_limit) * tokens_per_continue * model_weight
     IF est_latency < lat_budget AND est_compute < comp_budget:
       → candidate combo
   (These are rough estimates. Passing this check does not guarantee
    the batch latency will be within budget.)

4. IF no combo fits at current model:
   → Try lighter model (lower weight), repeat step 3

5. CHOOSE a combo. Build policy:
   spawn_many + continue_many loop up to loop_limit
   Vote with deterministic_plurality on available answers

6. AFTER FAILURE: read feedback.json
   → Check per-constraint satisfaction_rate and margin
   → Diagnose: which constraint(s) violated? (C8)
   → Small gap: adjust width or loop_limit by 1-2
   → Large gap: recalibrate from TRACE_SUMMARY (B5)
   → Check whether loop_limit was actually reached (C2)
   → Verify: does adjustment plausibly improve JOINT sat? (A5)
```

## Failure Diagnosis

| Symptom | Diagnosis | Suggested Adjustment |
|---|---|---|
| accuracy sat near 0, many branches still had can_continue=True | Branches did not run deep enough to produce valid answers | Increase loop_limit if latency allows |
| accuracy sat low, width=1-2, branches reached terminal | Not enough votes for reliable plurality | Increase width if compute allows |
| accuracy sat low, width>=5, loop_limit reached, branches at terminal | Model capacity insufficient | Consider larger model if budget allows; check for parse errors |
| latency sat < 1.0 | Total execution time exceeds batch budget | Reduce loop_limit; recalibrate mean_phase_lat from TRACE_SUMMARY |
| compute sat < 1.0 | width × depth × model_weight too large | Reduce width or loop_limit; consider lighter model |
| accuracy + latency both low | Depth needed for accuracy but latency budget too tight | Try lighter model (more depth per latency unit) |
| All three barely violated | Near feasibility boundary | Try width=1 with max depth at lightest feasible model |
| Joint sat unchanged despite adjustment | Changed a non-binding parameter, or improvement on one constraint offset by regression on another | Check all three constraints; verify loop_limit was actually reached before reducing it |
| Proposal sat = incumbent sat, rejected | Normal plateau (strict > acceptance) | Not a policy failure; try a different design direction if improvement is needed |
