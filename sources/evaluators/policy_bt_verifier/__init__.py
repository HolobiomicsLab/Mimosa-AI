"""Policy-BT verifier (E44-E48 architecture) — a selectable Mimosa
verification channel.

Re-executes each generation's deliverables in a verifier-owned sandbox,
digests the workspace deterministically, compares generations with an
LLM-written screened policy (plus a kimi per-pair visual channel for
figure-dominant tasks) and rewards arrival-moment win-rate vs
predecessors. See ``PLAN.md`` in this package for the measured
provenance of every design decision.
"""

from .digest import build_digest, digest_summary, is_figure_dominant
from .evaluator import PolicyBTVerifierEvaluator, find_run_snapshot
from .gradient import fold_gradient, refine_gradient
from .logging_utils import STAGES, StageLogger
from .policy import (
    MAIN_STANZA,
    POLICY_SHA,
    ensure_policy,
    run_policy,
    smoke_test,
    static_screen,
)
from .recon import RECON_PROMPT_VERSION
from .recon import build_prompt as build_recon_prompt
from .reexec import ReexecRunner
from .registry import TaskRegistry, task_key
from .verdicts import bradley_terry_reward, bt_fit_weighted, win_rate_reward
from .visual import VisualChannel

__all__ = [
    "MAIN_STANZA",
    "POLICY_SHA",
    "RECON_PROMPT_VERSION",
    "STAGES",
    "PolicyBTVerifierEvaluator",
    "ReexecRunner",
    "StageLogger",
    "TaskRegistry",
    "VisualChannel",
    "bradley_terry_reward",
    "bt_fit_weighted",
    "build_digest",
    "build_recon_prompt",
    "digest_summary",
    "ensure_policy",
    "find_run_snapshot",
    "fold_gradient",
    "is_figure_dominant",
    "refine_gradient",
    "run_policy",
    "smoke_test",
    "static_screen",
    "task_key",
    "win_rate_reward",
]
