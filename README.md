# PersonTTS — AIME reproduction package

Code and data to reproduce the main AIME experiment: the 2x2 ablation over the
strategy guide and warm-start retrieval, plus the three algorithmic baselines
and the AutoTTS baseline scored on the held-out split.

This package is scoped to that experiment. Other benchmarks, the bank-scaling
sweep, and the round-wise analyses are not included.

## What the experiment does

A user is described by a **requirement profile**: three hard constraints on
accuracy, latency and weighted compute. Given a profile, an LLM agent searches
for an **inference-resource-allocation controller** that satisfies all three at
once as often as possible.

The metric is the **joint satisfaction rate (JSR)**: over 1000 replay seeds,
the fraction in which mean accuracy, batch latency and mean weighted compute
simultaneously meet the profile's thresholds. Profiles are weighted equally.

Evaluation is a deterministic offline replay over pre-sampled rollouts. It
never calls a reasoning model, so it needs no GPU and no API key, and repeated
runs give bit-identical numbers. Only the discovery loop calls an LLM.

| | Problems | Visible to the agent |
|---|--:|---|
| discovery split (AIME_24 + AIME_25) | 60 | yes, re-evaluated every round |
| held-out split (AIME_26) | 30 | no, used only for final scoring |

The four ablation arms:

| Arm | Strategy guide | Warm start |
|---|:--:|:--:|
| PersonTTS (w/o Guide, w/o warm-start) | no | no |
| PersonTTS (w/o Guide) | no | yes |
| PersonTTS (w/o warm-start) | yes | no |
| PersonTTS | yes | yes |

## Layout

```
config/experiment_aime.json         experiment configuration
config/baseline_specs.json          the three algorithmic baselines
private_cache/AIME_24|25|26/        replay cache, one file per problem
profiles/aime_only/                 100 profiles the experience bank was built from
profiles/pilot_aime_20/             20 target profiles, the evaluation set
artifacts/experience_banks/         the experience bank used for warm start
artifacts/bank_source_policies/     the 100 controllers the bank entries point at
discovery/                          the discovery loop, prompts and strategy guide
evaluation/  replay/  policies/     scoring, replay engine, policy interfaces
eval/                               answer extraction and mathematical grading
scripts/                            entry points
```

The 20 target profiles and the 100 bank-source profiles have **no threshold
triple in common**. That separation is what makes warm-start retrieval a test
of transfer to an unseen user rather than a lookup of the user itself.

## Setup

Use Python 3.11 for the pinned grading dependencies.

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing without an API key

Everything below is pure CPU and deterministic.

**Check the replay cache is intact** (90 files, six models each):

```bash
python scripts/validate_cache.py --config config/experiment_aime.json \
  --datasets AIME_24,AIME_25,AIME_26 \
  --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
```

**Check the experience bank resolves.** The bank stores a path and a SHA-256
for each entry, and entries whose file is missing or altered are dropped
silently, which would quietly turn the warm-start arms into no-warm-start arms.
This prints one matched source per target profile:

```bash
python scripts/dry_run_retrieval.py \
  --bank artifacts/experience_banks/aime_bank_v1.json \
  --target-profiles profiles/pilot_aime_20/profiles.json \
  --target-config config/experiment_aime.json \
  --target-benchmark AIME
```

All 20 lines must show a matched source. Any `(no match: ...)` means the bank
did not load and the numbers that follow would be wrong.

**Score the three algorithmic baselines.** `asc`, `esc` and `parallel_probe`
are configured in `config/baseline_specs.json`. These need no discovery run,
so this works straight out of the archive:

```bash
python scripts/evaluate_heldout.py run \
  --campaign baselines_aime_20 --all-profiles --policy-type asc \
  --dataset AIME_26 --split heldout \
  --config config/experiment_aime.json \
  --profiles profiles/pilot_aime_20/profiles.json \
  --workers 8 --seed-count 1000
```

Repeat for `esc` and `parallel_probe`. Lower
`--seed-count` for a quick smoke test; the full figure uses 1000.

**Score the AutoTTS baseline.** Run the profile-free AutoTTS discovery and then
evaluate its selected controller on `AIME_26`:

```bash
BETA=0.5 bash scripts/run_autotts_baseline.sh
python scripts/eval_autotts_baseline.py \
  --run-id autotts_beta05 --campaign autotts_aime --dataset AIME_26 \
  --workers 8 --seed-count 1000
```

**Score a discovered controller.** This needs a campaign, which only exists
after a discovery run (next section), because the archive ships the inputs to
the search rather than its outputs:

```bash
python scripts/evaluate_heldout.py run \
  --campaign <campaign-id> --profile <profile-id> \
  --policy-type optimized --dataset AIME_26 --split heldout \
  --config config/experiment_aime.json \
  --profiles profiles/pilot_aime_20/profiles.json \
  --workers 8 --seed-count 1000
```

`--config` and `--profiles` default to the AIME configuration and target
profiles shipped in this package. Passing them explicitly is recommended when
running from another working directory. `--all-profiles` is rejected with
`--policy-type optimized`, which is per-profile by construction.

`--workers` only affects speed. Results are aggregated by a record key derived
from (dataset, problem, profile, seed), so any worker count yields identical
output.

## Reproducing the discovery loop (needs an API key)

Discovery is the only stage that calls an LLM. Set `ANTHROPIC_AUTH_TOKEN`, and
`ANTHROPIC_BASE_URL` if you route through a gateway.

```bash
python scripts/profile_campaign.py init \
  --campaign-id <campaign-id> \
  --config config/experiment_aime.json \
  --profiles profiles/pilot_aime_20/profiles.json \
  --round-limit 5 --seed-count 1000 --round-start-mode autotts_reset \
  --evaluation-workers 8 --agent-timeout-seconds 3600.0 \
  --agent-max-thinking-tokens 4000 --bucket-count 256 --max-attempts 3 \
  --model <model-id> \
  [--warm-start-bank artifacts/experience_banks/aime_bank_v1.json] \
  [--no-strategy-guide --prompt discovery/prompts/discovery_no_guide.md]
```

The two bracketed lines select the arm. Turning the guide off takes **both**
flags: `--no-strategy-guide` stops the guide being injected, and the alternate
prompt removes the wording that refers to it. Passing only one leaves the arm
contaminated.

`--round-limit`, `--seed-count`, `--round-start-mode` and `--model` define the
search and change the results if altered. `--evaluation-workers` is a speed
knob in `[1, 60]`, but it is frozen into the campaign manifest at `init`.

`round_start_mode=autotts_reset` redrafts from the warm-start seed at the start
of every round rather than refining the previous incumbent, so the bank and the
guide keep acting across all five rounds instead of only at initialisation.

Then run workers against the campaign:

```bash
CAMPAIGN_ID=<campaign-id> bash scripts/run_profile_campaign_worker.sh   # honours WORKER_ONCE=1
python scripts/profile_campaign.py status --campaign-id <campaign-id>
```

One worker handles one profile end to end, so useful concurrency is capped at
the number of profiles.

## Notes on the data

Each replay cache file holds, per problem and model, 128 pre-sampled rollouts.
A rollout is one initial pass plus up to five self-refinement rounds; within a
pass, checkpoints every 500 tokens each carry a ~30-token probe that reads out
the current answer. Sampling used `T=0.6, top_p=0.95, top_k=20`.

**Cost units.** `weighted_compute` counts tokens weighted by model size,
relative to the 0.6B model, so its unit is 0.6B-equivalent tokens:

| 0.6B | 1.7B | 4B | 8B | 14B | 32B |
|--:|--:|--:|--:|--:|--:|
| 1.00 | 2.83 | 6.67 | 13.33 | 23.33 | 53.33 |

This is a compute proxy, not a price. Restating it in currency is not a change
of units: the discovered controllers estimate their own costs internally, and a
third of them hardcode this weighting, so they would have to be re-discovered
under any other cost model rather than rescaled.

A replay seed permutes the order in which the 128 rollouts are consumed; it
does not change their content.

**Timing precision.** `think_elapsed_s`, `probe_elapsed_s` and
`feedback_elapsed_s` are stored to six decimal places rather than full float64,
which keeps the package inside its size budget. Microsecond resolution is far
below the measurement precision of a wall-clock latency in seconds. Scoring a
controller against the rounded and the unrounded cache gives the same
satisfaction rate.

`config/experiment_aime.json` also contains the deny-list used to constrain the
agent's file access; it only protects paths outside the released experiment
inputs.

## License

Original PersonTTS code is licensed under [Apache License 2.0](LICENSE),
Copyright 2026 PersonTTS contributors. Third-party code retains its original
licenses and notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The code license does not relicense third-party benchmark questions or other
third-party data.
