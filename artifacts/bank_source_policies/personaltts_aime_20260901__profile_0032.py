from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Three-branch parallel spawn with first-valid-answer selection.

    Profile 0032 constraints:
      accuracy  >= 0.00167 (mean)   -- BINDING
      latency   <= 5.471 s (batch)    -- comfortable with 3 branches
      weighted  <= 4 383 490 (mean) -- large headroom

    The incumbent (round 2) spawns 2 parallel Qwen3-0.6B branches and
    uses deterministic_plurality, yielding 97/60000 correct (utility 0.095).

    This policy adds a third parallel branch to improve null recovery:
    when branches 0 and 1 both return null, branch 2 may still produce
    the correct answer.  The first-valid-answer strategy (pick the
    earliest branch with a non-null answer) avoids the risk that two
    agreeing wrong answers outvote a lone correct answer, which standard
    plurality voting with 3 branches would introduce.

    Latency impact of max-of-3 vs max-of-2 is minimal: exponential tail
    analysis of AIME_25 single-branch latency (p90=4.98, p99=5.51)
    gives P(max-of-3 > 5.47) ~ 3.5 % per AIME_25 problem, translating
    to ~2 % batch-level batch failure -- well below the accuracy-driven
    batch pass ceiling.

    Weighted compute: 3 x ~517 = ~1 551 per problem, negligible against
    the 73 000 per-problem budget.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    @staticmethod
    def _model_size_estimate(name: str) -> float:
        """Heuristic model-size extractor (billions) from the model name."""
        for part in name.replace("-", " ").replace("_", " ").split():
            upper = part.upper()
            if upper.endswith("B"):
                try:
                    return float(upper[:-1])
                except ValueError:
                    continue
        return 1000.0

    def solve(self, env) -> Optional[str]:
        # 1. Pick the smallest (fastest) available model to minimise latency.
        model = min(env.models, key=self._model_size_estimate)

        # 2. Spawn three branches in parallel.
        #    spawn_many adds only the MAX cohort latency (one parallel step),
        #    so three branches barely increase latency over two.
        branches = env.spawn_many(model, 3)

        # 3. First-valid-answer strategy: iterate branches in order and
        #    pick the earliest non-null answer.  This maximises null
        #    recovery (branch 2 catches double-null from branches 0-1)
        #    without the majority-voting risk that plurality would add.
        answer = None
        for b in branches:
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                answer = ans
                break

        # 4. Finish with the selected answer (or None if all branches null).
        return env.finish(answer)
