from __future__ import annotations

from typing import Optional

from policies.seeds import ASCConfigured4D


class CandidatePolicy(ASCConfigured4D):
    """Valid discovery seed using AutoTTS ASC without a beta-derived heuristic."""

    NAME = "candidate"

    def solve(self, env) -> Optional[str]:
        return super().solve(env)
