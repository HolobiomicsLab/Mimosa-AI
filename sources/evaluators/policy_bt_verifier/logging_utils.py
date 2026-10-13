"""Stage logging for the policy_bt verifier (plan §6b).

One clear stdout line per pipeline stage, prefixed
``[policy_bt][<uuid8>]`` and stage-tagged, plus the same facts as one
JSON object per line to a file sink (``verifier_policy_bt.log``) so the
Observatory webui can tail the run without parsing prose.
:meth:`StageLogger.payload` additionally logs FULL payloads (policy
source, sandbox listing, per-generation digests, per-pair verdicts, recon
config, raw + refined gradients) — concise on stdout, verbatim in the
sink; every stage/payload record carries ``stage_cost_usd`` (per-stage
LLM spend delta) and ``spend_usd`` (cumulative). Verbose mode adds
per-pair and per-repair detail lines. :meth:`preview` shows a bounded
numbered console preview of a multi-line payload (policy source) while
the sink keeps the full text; :meth:`table` prints aligned operator
tables (live Bradley-Terry standings). ANSI colors paint the stage tags
on an interactive TTY (``NO_COLOR`` or a non-TTY stdout disables them;
``color=`` forces a side), so captured/CI output stays plain and
grep-safe. No module in this package prints anywhere else — everything
goes through :class:`StageLogger`.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

from sources.cli.pretty_print import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RESET,
    YELLOW,
)

PREFIX = "policy_bt"
#: Stage tags, in evaluation-flow order (tests assert this order).
STAGES = (
    "REEXEC",
    "DIGEST",
    "RECON",
    "POLICY",
    "VISUAL",
    "PAIRS",
    "REWARD",
    "GRADIENT",
    "DONE",
)


def _color_output() -> bool:
    """ANSI colors only on an interactive TTY (NO_COLOR disables)."""
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, OSError, ValueError):
        return False


#: Stage tag -> ANSI style(s); unlisted stages stay plain.
_STAGE_TAG_STYLES: dict[str, tuple[str, ...]] = {
    "REEXEC": (CYAN,),
    "DIGEST": (CYAN,),
    "RECON": (CYAN,),
    "POLICY": (CYAN, BOLD),
    "VISUAL": (MAGENTA,),
    "PAIR": (MAGENTA,),
    "PAIRS": (MAGENTA,),
    "REWARD": (GREEN, BOLD),
    "GRADIENT": (MAGENTA,),
    "DONE": (GREEN, BOLD),
    "FILES": (CYAN,),
}


class StageLogger:
    """Stage-tagged stdout lines + JSON-lines file sink for one evaluation.

    Attributes:
        uuid: Full workflow uuid being evaluated.
        uuid8: LAST 8 characters of *uuid* (the log prefix tag — Mimosa
            uuids start with the date, so the first 8 collide across a
            whole day's generations; P5.1).
        sink: Optional JSON-lines file sink path (created lazily).
        verbose: When True, :meth:`detail` lines are printed too.
        spend_usd: Cumulative LLM spend recorded via :meth:`add_cost`.
    """

    def __init__(
        self,
        uuid: str,
        sink: Path | None = None,
        verbose: bool = False,
        color: bool | None = None,
    ) -> None:
        """Bind the logger to one evaluation run.

        Args:
            uuid: Workflow uuid (used for the ``[<uuid8>]`` tag).
            sink: Optional path of the JSON-lines log file; the parent
                directory is created when the first record is written.
            verbose: Enable per-pair / per-repair detail lines.
            color: Force ANSI colors on/off; None auto-detects a TTY
                (``NO_COLOR`` disables, as customary).
        """
        self.uuid = uuid
        self.uuid8 = uuid[-8:] if uuid else "--------"
        self.sink = sink
        self.verbose = verbose
        self.color = _color_output() if color is None else bool(color)
        self.spend_usd = 0.0
        self.n_llm_calls = 0
        self._last_spend = 0.0

    def paint(self, text: str, *styles: str) -> str:
        """Wrap *text* in ANSI *styles* (no-op unless colors are on)."""
        active = [s for s in styles if s]
        if not self.color or not active:
            return text
        return "".join(active) + text + RESET

    def _line(
        self,
        tag: str,
        message: str,
        tag_styles: tuple[str, ...],
        pad: int = 8,
    ) -> None:
        """One prefixed stdout line with a painted tag."""
        tag = self.paint(tag.ljust(pad), *tag_styles)
        print(f"[{PREFIX}][{self.uuid8}] {tag} {message}", flush=True)

    # ------------------------------------------------------------------
    # Public emitters
    # ------------------------------------------------------------------

    def stage(self, stage: str, message: str = "", **fields: Any) -> None:
        """Emit one stage line to stdout and one JSON record to the sink.

        Args:
            stage: Stage tag (``REEXEC``, ``DIGEST``, ...).
            message: Free-form outcome summary (short).
            **fields: Machine-readable facts appended as ``key=value``
                pairs and stored verbatim in the JSON record.
        """
        extras = "".join(f" {k}={v}" for k, v in fields.items())
        self._line(stage, f"{message}{extras}", _STAGE_TAG_STYLES.get(stage, ()))
        record: dict[str, Any] = {
            "stage": stage,
            "uuid": self.uuid,
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
        }
        if message:
            record["message"] = message
        record.update(fields)
        record["spend_usd"] = round(self.spend_usd, 6)
        record["stage_cost_usd"] = self._stage_cost()
        self._write_record(record)

    def payload(
        self, stage: str, message: str = "", data: Any = None, **fields: Any
    ) -> None:
        """Concise stdout line + one JSON record carrying a FULL payload.

        The stdout side stays one short line (same shape as :meth:`stage`);
        the sink side stores *data* verbatim — never truncated — so the
        log file holds complete policy sources, sandbox listings, digest
        summaries, per-pair verdicts and both gradients (raw + refined).

        Args:
            stage: Stage tag (``REEXEC``, ``POLICY``, ``PAIR``, ...).
            message: Short outcome summary for stdout.
            data: Arbitrary JSON-serialisable payload for the sink.
            **fields: Machine-readable facts printed as ``key=value``
                pairs and stored verbatim in the JSON record.
        """
        extras = "".join(f" {k}={v}" for k, v in fields.items())
        self._line(stage, f"{message}{extras}", (MAGENTA,))
        record: dict[str, Any] = {
            "stage": stage,
            "uuid": self.uuid,
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
        }
        if message:
            record["message"] = message
        record["payload"] = data
        record.update(fields)
        record["spend_usd"] = round(self.spend_usd, 6)
        record["stage_cost_usd"] = self._stage_cost()
        self._write_record(record)

    def _stage_cost(self) -> float:
        """LLM spend since the previous emitted record (per-stage delta)."""
        delta = round(self.spend_usd - self._last_spend, 6)
        self._last_spend = self.spend_usd
        return delta

    def detail(self, text: str) -> None:
        """Verbose-only detail line (per pair, per repair round)."""
        if not self.verbose:
            return
        self._line("DETAIL", text, (DIM,))
        self._write_record({"stage": "DETAIL", "uuid": self.uuid, "text": text})

    def warn(self, text: str) -> None:
        """Loud warning line (failures the operator must see)."""
        self._line("WARNING", self.paint(text, YELLOW), (YELLOW, BOLD), pad=0)
        self._write_record(
            {
                "stage": "WARNING",
                "uuid": self.uuid,
                "ts": dt.datetime.now().isoformat(timespec="seconds"),
                "text": text,
            }
        )

    def add_cost(self, cost_usd: float | None) -> None:
        """Accumulate LLM spend (``None`` prices are skipped)."""
        self.n_llm_calls += 1
        if cost_usd is not None:
            self.spend_usd += float(cost_usd)

    def spend_line(self) -> str:
        """Format the cumulative-spend suffix used on stage lines."""
        return f"${self.spend_usd:.4f}"

    def preview(
        self,
        label: str,
        text: str,
        max_lines: int = 12,
        max_chars: int = 100,
        **fields: Any,
    ) -> None:
        """Bounded console preview of a multi-line payload; FULL in sink.

        Console shows the first *max_lines* lines (each capped at
        *max_chars*) with line numbers and a ``+N more lines`` marker;
        the sink record stores *text* verbatim (never truncated).
        """
        lines = text.splitlines() if text else []
        self._preview_console(label, lines, max_lines, max_chars)
        record: dict[str, Any] = {
            "stage": "PREVIEW",
            "uuid": self.uuid,
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "label": label,
            "n_lines": len(lines),
            "text": text,
        }
        record.update(fields)
        self._write_record(record)

    def _preview_console(
        self, label: str, lines: list[str], max_lines: int, max_chars: int
    ) -> None:
        """Numbered, bounded preview block on the console."""
        self._line(
            "PREVIEW",
            self.paint(label, BOLD)
            + f" ({len(lines)} lines, showing {min(len(lines), max_lines)})",
            (CYAN,),
        )
        for i, ln in enumerate(lines[:max_lines], 1):
            print(f"      {self.paint(f'{i:>3} │', DIM)} {ln[:max_chars]}", flush=True)
        hidden = len(lines) - min(len(lines), max_lines)
        if hidden > 0:
            print(
                self.paint(
                    f"      … +{hidden} more lines (full text in the JSON sink)",
                    DIM,
                ),
                flush=True,
            )

    def table(
        self,
        label: str,
        header: list[str],
        rows: list[list[Any]],
        row_styles: list[str | None] | None = None,
        **fields: Any,
    ) -> None:
        """Aligned console table + one TABLE record in the sink.

        *row_styles* optionally paints whole rows (one ANSI style per
        row, ``None`` for plain) — alignment is computed on the plain
        cell text, so colors never skew the columns.
        """
        plain = [[str(c) for c in row] for row in rows]
        widths = [
            max([len(str(h))] + [len(row[i]) for row in plain])
            for i, h in enumerate(header)
        ]
        self._line("TABLE", self.paint(label, BOLD), (CYAN,))
        head = "  ".join(
            self.paint(str(h).ljust(w), DIM)
            for h, w in zip(header, widths, strict=False)
        )
        print(f"      {head}{RESET if self.color else ''}", flush=True)
        styles = row_styles or [None] * len(plain)
        for row, style in zip(plain, styles, strict=False):
            line = "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=False))
            print(f"      {self.paint(line, style)}", flush=True)
        record: dict[str, Any] = {
            "stage": "TABLE",
            "uuid": self.uuid,
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "label": label,
            "header": header,
            "rows": plain,
        }
        record.update(fields)
        self._write_record(record)

    # ------------------------------------------------------------------
    # Sink
    # ------------------------------------------------------------------

    def _write_record(self, record: dict[str, Any]) -> None:
        """Append one JSON record to the file sink (best effort)."""
        if self.sink is None:
            return
        try:
            self.sink.parent.mkdir(parents=True, exist_ok=True)
            with open(self.sink, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            # The sink is observability only; never fail the run for it.
            pass
