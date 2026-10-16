#!/usr/bin/env python3
"""Tests for the targeting-computer terminal theme (sources/cli/theme.py)."""

import io
import os
import re
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.cli import theme

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI escape sequences so assertions see visible text only."""
    return _ANSI_RE.sub("", text)


def test_ansi_enabled_respects_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert theme.ansi_enabled() is False


def test_ansi_enabled_force_color_overrides_non_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert theme.ansi_enabled() is True


def test_ansi_enabled_off_when_not_a_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert theme.ansi_enabled() is False


def test_status_tags_align_messages(capsys):
    theme.ok("msg")
    theme.warn("msg")
    theme.fail("msg")
    theme.info("msg")
    lines = [_plain(ln) for ln in capsys.readouterr().out.splitlines()]
    assert len(lines) == 4
    assert len({line.index("msg") for line in lines}) == 1


def test_leader_aligns_values_across_names():
    short = _plain(theme.leader("SHORT", "found"))
    longer = _plain(theme.leader("A_MUCH_LONGER_NAME", "found"))
    assert short.index("found") == longer.index("found")


def test_step_header_shows_progress_track(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    theme.step_header(3, 9, "LLM model selection")
    out = _plain(capsys.readouterr().out)
    assert "STEP 3 OF 9" in out
    assert "LLM MODEL SELECTION" in out
    assert "▰▰▰" in out
    assert "▱" * 6 in out


def test_step_header_repeat_pass_hides_progress(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    theme.step_header(6, 9, "Your objective", show_progress=False)
    out = _plain(capsys.readouterr().out)
    assert "STEP" not in out
    assert "YOUR OBJECTIVE" in out


def test_frame_edges_share_width(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    theme.frame_top("PRE-FLIGHT")
    theme.frame_bottom()
    top, bottom = [_plain(ln) for ln in capsys.readouterr().out.splitlines()]
    assert top.startswith("  ┌─ PRE-FLIGHT")
    assert bottom.endswith("┘")
    assert len(top) == len(bottom)


def test_kv_wraps_long_values_with_aligned_continuation(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "66")
    theme.kv("objective", "word " * 40)
    lines = [_plain(ln) for ln in capsys.readouterr().out.splitlines()]
    assert len(lines) > 1
    value_col = lines[0].index("word")
    assert all(line.index("word") == value_col for line in lines[1:])
    max_len = theme.term_width() + 6
    assert all(len(line) <= max_len for line in lines)


def test_kv_uppercases_label_and_pads_leaders(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    theme.kv("mode", "GOAL")
    theme.kv("astra export", "OFF")
    lines = [_plain(ln) for ln in capsys.readouterr().out.splitlines()]
    assert lines[0].startswith("    MODE ")
    assert lines[1].startswith("    ASTRA EXPORT ")
    assert lines[0].index("GOAL") == lines[1].index("OFF")


def test_banner_frames_console_name_and_tagline():
    text = _plain(theme.banner("Flight console", "autonomous science"))
    assert "FLIGHT CONSOLE" in text
    assert "autonomous science" in text
    assert "███╗" in text
    lines = text.splitlines()
    top = next(ln for ln in lines if "┌╌" in ln)
    bottom = next(ln for ln in lines if "└╌" in ln)
    assert len(top) == len(bottom)


def test_term_width_is_capped_and_floored(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    assert theme.term_width() == theme.MAX_WIDTH
    monkeypatch.setenv("COLUMNS", "30")
    assert theme.term_width() == 40


if __name__ == "__main__":
    test_leader_aligns_values_across_names()
    print("theme smoke checks passed")
