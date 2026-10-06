from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn with bounded depth to satisfy tight batch latency constraints.

    The baseline sequential ASC spawns branches one at a time, running each to
    terminal before starting the next.  This makes total latency the SUM of all
    per-step latencies across every branch.  By switching to spawn_many /
    continue_many, each round charges only the MAX cohort latency, cutting
    wall-clock time dramatically while majority-voting preserves accuracy.

    Design parameters (for profile_0050: accuracy>=0.4265, batch_latency<=638s,
    mean_wc<=21.3M):
      width = 5          parallel branches for robust majority voting
      max_continues = 15  depth cap keeping 16 rounds * ~25s/round  < 638s
      consensus = 4       early-stop when 4/5 agree to save easy-problem latency
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def solve(self, env) -> Optional[str]:
        # --- model selection: smallest available for lowest per-token latency ---
        model = "Qwen3-4B"
        if model not in env.models:
            model = env.models[0]

        width = 5
        max_continues = 15
        consensus_min = 4  # early-stop threshold (out of width)

        # Parallel spawn: one latency step (max of *width* spawn latencies)
        env.spawn_many(model, width)

        for round_idx in range(max_continues):
            state = env.observe()
            continuable = [
                b["branch_id"]
                for b in state["branches"]
                if b["can_continue"] and not b["pruned"]
            ]
            if not continuable:
                break
            env.continue_many(continuable)

            # --- early consensus check (skip round 0 to let reasoning develop) ---
            if round_idx >= 1:
                state = env.observe()
                answers = []
                for b in state["branches"]:
                    if not b["pruned"]:
                        ans = valid_answer(b.get("latest_answer"))
                        if ans is not None:
                            answers.append(ans)
                if len(answers) >= consensus_min:
                    counts = Counter(answers)
                    top_count = max(counts.values())
                    if top_count >= consensus_min:
                        # deterministic winner: first appearance with top count
                        winner = next(
                            a for a in answers if counts[a] == top_count
                        )
                        return env.finish(winner)

        # Fall back to deterministic plurality across all branches
        return env.finish()
