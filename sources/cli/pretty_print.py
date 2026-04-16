"""
Pretty-print utilities for Mimosa-AI CLI output.
Modern Claude-code–style terminal output with consistent, clean styling.

Usage::

    from sources.cli.pretty_print import (
        print_ok, print_warn, print_err, print_info,
        print_phase, print_section, print_rule,
        print_iteration_header,
        print_box,
        print_kv_row, print_summary,
        print_agent_answers,
        print_step,
        CYAN, GREEN, YELLOW, RED, BLUE, MAGENTA, BOLD, DIM, RESET,
    )
"""

from __future__ import annotations

import textwrap

# ── ANSI colour constants ──────────────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

# Default column width used for banners/summaries
_W = 80


# ── Status lines ──────────────────────────────────────────────────────────────

def print_ok(msg: str) -> None:
    """Print a green success line  ✓  <msg>"""
    print(f"{GREEN}  ✓  {msg}{RESET}")


def print_warn(msg: str) -> None:
    """Print a yellow warning line  ⚠  <msg>"""
    print(f"{YELLOW}  ⚠  {msg}{RESET}")


def print_err(msg: str) -> None:
    """Print a red error line  ✗  <msg>"""
    print(f"{RED}  ✗  {msg}{RESET}")


def print_info(msg: str) -> None:
    """Print a dim informational line  ·  <msg>"""
    print(f"{DIM}  ·  {msg}{RESET}")


# ── Step header (used by onboarding CLI) ──────────────────────────────────────

def print_step(step: int, total: int, title: str, width: int = 60) -> None:
    """
    Numbered step header used by the onboarding wizard.

    Example::

        ────────────────────────────────────────────────────────────
          Step 1/8  ·  API Key Check
        ────────────────────────────────────────────────────────────
    """
    bar = "─" * width
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{CYAN}  Step {step}/{total}  ·  {title}{RESET}")
    print(f"{CYAN}{bar}{RESET}")


# ── Phase / section banners ──────────────────────────────────────────────────

def print_phase(
    title: str,
    icon: str = "",
    width: int = _W,
    color: str = CYAN,
) -> None:
    """
    Full-width phase banner with centred title and horizontal rules.
    """
    label = f"{icon}  {title}" if icon else title
    bar = "─" * width
    print(f"\n{color}{bar}{RESET}")
    print(f"{color}{label:^{width}}{RESET}")
    print(f"{color}{bar}{RESET}")


def print_section(
    title: str,
    color: str = GREEN,
    width: int = 80,
) -> None:
    """
    Compact inline section label.
    """
    label = f"  {title}  "
    remaining = max(0, width - len(label) - 2)
    print(f"\n{color}──{label}{'─' * remaining}{RESET}")


def print_rule(width: int = _W, color: str = CYAN) -> None:
    """Print a plain horizontal rule."""
    print(f"{color}{'─' * width}{RESET}")


# ── Iteration banner ─────────────────────────────────────────────────────────

def print_iteration_header(
    current: int,
    total: int,
    subtitle: str = "Self-Improvement Loop",
    width: int = _W,
) -> None:
    """
    Prominent iteration counter banner shown at the start of each DGM loop.
    """
    bar = "═" * width
    print(f"\n{CYAN}{bar}{RESET}")
    if total <= 1:
        print(f"{CYAN}{BOLD} Starting task... {RESET}")
    else:
        print(f"{CYAN}{BOLD}  ITERATION {current}/{total}  ·  {subtitle}{RESET}")
        print(f"{DIM} Mimosa will now learn how to build the workflow for the task.{RESET}")
    print(f"{CYAN}{bar}{RESET}")


# ── Content box ──────────────────────────────────────────────────────────────

def print_box(
    content: str,
    title: str = "",
    color: str = CYAN,
    width: int = 80,
    truncate: int = 512,
) -> None:
    """
    Render content inside a Unicode-bordered box.

    Long lines are word-wrapped to fit inside the box.  Lines longer than
    *truncate* characters are hard-truncated and annotated with a
    ``…(N chars not shown)`` note.

    Example::

        ╭─  CURRENT TASK  ──────────────────────────────────────────╮
        │  Create a simple test workflow that demonstrates basic        │
        │  functionality by outputting the text 'Hello World'.          │
        ╰──────────────────────────────────────────────────────────────╯
    """
    inner_w = width - 4  # 2 border chars + 2 spaces padding on each side

    # Top border
    if title:
        top_label = f"─  {title}  "
        top_fill = max(0, width - len(top_label) - 2)
        top = f"╭{top_label}{'─' * top_fill}╮"
    else:
        top = f"╭{'─' * (width - 2)}╮"

    print(f"{color}{top}{RESET}")

    for raw_line in content.splitlines():
        if len(raw_line) <= inner_w:
            print(f"{color}│  {raw_line.ljust(inner_w)}  │{RESET}")
        elif len(raw_line) <= truncate:
            # word-wrap
            wrapped_lines = textwrap.wrap(raw_line, inner_w) or [""]
            for wrapped in wrapped_lines:
                print(f"{color}│  {wrapped.ljust(inner_w)}  │{RESET}")
        else:
            # hard-truncate + annotate
            shown = raw_line[:truncate]
            note = f"…({len(raw_line) - truncate} chars not shown)"
            for wrapped in textwrap.wrap(shown, inner_w) or [""]:
                print(f"{color}│  {wrapped.ljust(inner_w)}  │{RESET}")
            print(f"{DIM}│  {note.ljust(inner_w)}  │{RESET}")

    print(f"{color}╰{'─' * (width - 2)}╯{RESET}")


# ── Key-value rows / summary ──────────────────────────────────────────────────

def print_kv_row(
    key: str,
    value: str,
    color: str = YELLOW,
    key_width: int = 20,
) -> None:
    """Print a single ``key → value`` row with aligned columns."""
    print(f"  {BOLD}{key:<{key_width}}{RESET}  {color}{value}{RESET}")


def print_summary(
    title: str,
    items: list[tuple[str, str]],
    color: str = GREEN,
    width: int = _W,
) -> None:
    """
    Print a titled summary block with aligned key-value pairs.

    Example::

        ────────────────────────────────────────────────────────────────────────────────
          ✨ WORKFLOW COMPLETION SUMMARY
        ────────────────────────────────────────────────────────────────────────────────
          UUID              20260408_145530_9a908c10
          Total time        12.340s
          Generation        3.120s
          Dependencies      1.200s
          Execution         8.020s
        ────────────────────────────────────────────────────────────────────────────────
    """
    bar = "─" * width
    key_width = max((len(k) for k, _ in items), default=12) + 2
    print(f"\n{color}{bar}{RESET}")
    print(f"{color}  {title}{RESET}")
    print(f"{color}{bar}{RESET}")
    for key, value in items:
        print(f"  {BOLD}{key:<{key_width}}{RESET}  {color}{value}{RESET}")
    print(f"{color}{bar}{RESET}\n")


# ── Agent answers ─────────────────────────────────────────────────────────────

def print_agent_answers(
    answers_text: str,
    color: str = CYAN,
    width: int = _W,
) -> None:
    """
    Display workflow agent answers inside a labelled section.

    Example::

        ───────────────────────  WORKFLOW AGENTS ANSWERS  ───────────────────────────────
          agent 0: Task completed successfully.
          agent 1: All assertions passed.
        ─────────────────────────────────────────────────────────────────────────────────
    """
    if not answers_text or not answers_text.strip():
        return

    label = "  WORKFLOW AGENTS ANSWERS  "
    side = max(0, width - len(label))
    left = side // 2
    right = side - left
    print(f"\n{color}{'─' * left}{label}{'─' * right}{RESET}")
    for line in answers_text.splitlines():
        print(f"{color}  {line}{RESET}")
    print(f"{color}{'─' * width}{RESET}\n")
