"""
Targeting-computer terminal theme for the Mimosa wizard CLIs.

Amber-on-black "flight console" design language shared by the
onboarding and evaluation wizards: fixed-width status tags, dot-leader
readout lines, progress-tracked step headers and open corner-tick
frames (no closed boxes, so content can wrap freely at any width).

Styling is disabled automatically when stdout is not a TTY or when
``NO_COLOR`` is set, and can be forced back on with ``FORCE_COLOR``.
Constants resolve to empty strings when styling is off, so f-strings
built from them degrade to plain text. Color support is decided once,
at import time — set ``NO_COLOR``/``FORCE_COLOR`` before importing.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap

MAX_WIDTH = 62


def ansi_enabled() -> bool:
    """Return whether ANSI styling should be emitted on stdout."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


_ENABLED = ansi_enabled()


def _sgr(params: str) -> str:
    """Return an SGR escape sequence, or '' when styling is disabled."""
    return f"\033[{params}m" if _ENABLED else ""


# ── Palette ────────────────────────────────────────────────────────────
AMBER = _sgr("38;5;214")            # primary chrome, tags, prompts
EMBER = _sgr("38;5;130")            # dim chrome: leaders, frames, rules
WHITE = _sgr("97")                  # labels and key content
GREY  = _sgr("38;5;245")            # secondary text
BOLD  = _sgr("1")
DIM   = _sgr("2")
RESET = _sgr("0")
_WARN_TAG = _sgr("48;5;214") + _sgr("38;5;232")   # black on amber
_FAIL_TAG = _sgr("48;5;124") + _sgr("97")          # white on red

LOCKED = f"{AMBER}◄ LOCKED{RESET}"


def term_width() -> int:
    """Return the usable chrome width for the current terminal."""
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    return max(40, min(cols - 4, MAX_WIDTH))


_FIGLET = """\
  ███╗   ███╗██╗███╗   ███╗ ██████╗ ███████╗ █████╗
  ████╗ ████║██║████╗ ████║██╔═══██╗██╔════╝██╔══██╗
  ██╔████╔██║██║██╔████╔██║██║   ██║███████╗███████║
  ██║╚██╔╝██║██║██║╚██╔╝██║██║   ██║╚════██║██╔══██║
  ██║ ╚═╝ ██║██║██║ ╚═╝ ██║╚██████╔╝███████║██║  ██║
  ╚═╝     ╚═╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝"""

_BANNER_TICK = " " * 54


def banner(console: str, tagline: str) -> str:
    """Return the framed MIMOSA figlet banner for a wizard console.

    Args:
        console: Console name shown under the figlet (e.g. "Flight console").
        tagline: Short grey tagline shown after the console name.
    """
    return (
        f"\n{EMBER}  ┌╌{_BANNER_TICK}╌┐{RESET}\n"
        f"{AMBER}{BOLD}\n{_FIGLET}\n{RESET}\n"
        f"  {WHITE}{BOLD}{console.upper()}{RESET} {EMBER}▏{RESET} "
        f"{GREY}{tagline}{RESET}\n"
        f"{EMBER}  └╌{_BANNER_TICK}╌┘{RESET}\n"
    )


# ── Status lines ───────────────────────────────────────────────────────

def ok(msg: str) -> None:
    """Print an amber ``[ OK ]`` status line."""
    print(f"  {AMBER}[ OK ]{RESET}  {msg}")


def warn(msg: str) -> None:
    """Print a black-on-amber ``[WARN]`` status line."""
    print(f"  {_WARN_TAG}[WARN]{RESET}  {msg}")


def fail(msg: str) -> None:
    """Print a white-on-red ``[FAIL]`` status line."""
    print(f"  {_FAIL_TAG}[FAIL]{RESET}  {msg}")


def info(msg: str) -> None:
    """Print a dim ``[ ·· ]`` informational line."""
    print(f"  {EMBER}[ ·· ]{RESET}  {GREY}{msg}{RESET}")


def leader(name: str, value: str, width: int = 30) -> str:
    """Return a dot-leader readout segment: ``name ····· value``."""
    dots = "·" * max(2, width - len(name))
    return f"{WHITE}{name}{RESET} {EMBER}{dots}{RESET} {GREY}{value}{RESET}"


# ── Chrome ─────────────────────────────────────────────────────────────

def rule() -> None:
    """Print a dim horizontal rule spanning the chrome width."""
    print(f"  {EMBER}{'─' * term_width()}{RESET}")


def step_header(step: int, total: int, title: str,
                show_progress: bool = True) -> None:
    """Print a step header with a ``▰▰▱`` progress track and rule.

    Args:
        step: 1-based current step.
        total: Total number of steps.
        title: Step title (rendered uppercase).
        show_progress: Hide the track and counter on repeat passes.
    """
    print()
    label = f"{WHITE}{BOLD}{title.upper()}{RESET}"
    if show_progress:
        track = f"{AMBER}{'▰' * step}{EMBER}{'▱' * (total - step)}{RESET}"
        counter = f"{AMBER}STEP {step} OF {total}{RESET}"
        print(f"  {track}  {counter} {EMBER}▏{RESET}{label}")
    else:
        print(f"  {EMBER}▏{RESET}{label}")
    rule()


def substep(label: str, title: str, note: str = "") -> None:
    """Print a sub-step heading like ``3a ▏ORCHESTRATION MODEL``."""
    print(f"\n  {WHITE}{label}{RESET} {EMBER}▏{RESET}{AMBER}{title.upper()}{RESET}")
    if note:
        print(f"       {GREY}{note}{RESET}")


def section(label: str) -> None:
    """Print a ``── LABEL ────`` section rule."""
    tail = "─" * max(0, term_width() - len(label) - 4)
    print(f"\n  {EMBER}── {RESET}{AMBER}{BOLD}{label}{RESET} {EMBER}{tail}{RESET}")


def frame_top(label: str) -> None:
    """Print the opening corner-tick of a HUD frame: ``┌─ LABEL ───``."""
    tail = "─" * max(0, term_width() - len(label) - 4)
    print(f"  {EMBER}┌─ {RESET}{AMBER}{BOLD}{label}{RESET}{EMBER} {tail}{RESET}")


def frame_bottom() -> None:
    """Print the closing corner-tick of a HUD frame: ``────┘``."""
    print(f"  {EMBER}{'─' * (term_width() - 1)}┘{RESET}")


def kv(label: str, value: str, accent: bool = False,
       label_width: int = 14) -> None:
    """Print a dot-leader key/value row inside a frame, wrapping long values.

    Args:
        label: Row label (rendered uppercase).
        value: Row value; wraps onto aligned continuation lines.
        accent: Render the value in amber instead of grey.
        label_width: Column the dot leaders pad the label to.
    """
    label = label.upper()
    dots = "·" * max(2, label_width - len(label))
    colour = AMBER if accent else GREY
    avail = max(20, term_width() - label_width - 6)
    lines = textwrap.wrap(str(value), width=avail) or [""]
    print(f"    {WHITE}{label}{RESET} {EMBER}{dots}{RESET} {colour}{lines[0]}{RESET}")
    pad = " " * (label_width + 6)
    for line in lines[1:]:
        print(f"{pad}{colour}{line}{RESET}")


def wrap(text: str, width: int | None = None, indent: int = 2) -> str:
    """Fill *text* to the chrome width with a uniform left indent."""
    width = width or (term_width() + 6)
    pad = " " * indent
    return textwrap.fill(text, width=width, initial_indent=pad,
                         subsequent_indent=pad)


# ── Prompts ────────────────────────────────────────────────────────────

def ask(prompt: str, default: str = "") -> str:
    """Print an amber ``»`` prompt and return stripped input (empty → default).

    Exits cleanly on Ctrl-C / EOF.
    """
    suffix = f" {EMBER}[{default}]{RESET}" if default else ""
    try:
        answer = input(f"\n  {AMBER}»{RESET} {WHITE}{prompt}{RESET}{suffix}{AMBER}:{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return answer if answer else default


def ask_yn(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question and return the answer as a boolean."""
    hint = "Y/n" if default else "y/N"
    raw = ask(f"{prompt} ({hint})", default="y" if default else "n").lower()
    return raw in ("y", "yes", "1", "true")


if __name__ == "__main__":
    print("\n  theme smoke check — every element renders:\n")
    ok(leader("ANTHROPIC_API_KEY", "found"))
    warn(leader("workspace mount", "outside toolomics tree"))
    fail(leader("toolomics uplink", "no servers"))
    info("scanning discovery addresses")
    step_header(3, 9, "LLM model selection")
    substep("3a", "Orchestration model", "planning and workflow generation")
    print(f"    {AMBER}[1]{RESET}  {WHITE}Anthropic · recommended{RESET}  {LOCKED}")
    print(f"         {GREY}anthropic/claude-opus-4-8{RESET}")
    section("IGNITION")
    frame_top("PRE-FLIGHT")
    print()
    kv("mode", "GOAL", accent=True)
    kv("objective", "Reproduce Figure 3 of Smith et al. 2024 from the "
                    "public dataset and report RMSE against the paper.")
    print()
    frame_bottom()
    print(wrap("Corner-tick frames stay open on the right, so wrapped "
               "content never breaks the chrome."))
    print("\n  smoke check done\n")
