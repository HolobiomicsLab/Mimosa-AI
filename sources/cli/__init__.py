"""
Interactive onboarding CLI for Mimosa-AI.
Guides users through setup and launches the appropriate execution mode.
"""

from .onboard_cli import OnboardCLI
from .evaluation_cli import EvaluationCLI
from .pretty_print import (
    print_ok,
    print_warn,
    print_err,
    print_info,
    print_step,
    print_phase,
    print_section,
    print_rule,
    print_iteration_header,
    print_box,
    print_kv_row,
    print_summary,
    print_agent_answers,
    CYAN, GREEN, YELLOW, RED, BLUE, MAGENTA, BOLD, DIM, RESET,
)

__all__ = [
    "OnboardCLI",
    "EvaluationCLI",
    "print_ok",
    "print_warn",
    "print_err",
    "print_info",
    "print_step",
    "print_phase",
    "print_section",
    "print_rule",
    "print_iteration_header",
    "print_box",
    "print_kv_row",
    "print_summary",
    "print_agent_answers",
    "CYAN", "GREEN", "YELLOW", "RED", "BLUE", "MAGENTA", "BOLD", "DIM", "RESET",
]
