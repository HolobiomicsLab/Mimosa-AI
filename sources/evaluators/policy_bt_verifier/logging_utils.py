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
per-pair and per-repair detail lines. No module in this package prints
anywhere else — everything goes through :class:`StageLogger`.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

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
    ) -> None:
        """Bind the logger to one evaluation run.

        Args:
            uuid: Workflow uuid (used for the ``[<uuid8>]`` tag).
            sink: Optional path of the JSON-lines log file; the parent
                directory is created when the first record is written.
            verbose: Enable per-pair / per-repair detail lines.
        """
        self.uuid = uuid
        self.uuid8 = uuid[-8:] if uuid else "--------"
        self.sink = sink
        self.verbose = verbose
        self.spend_usd = 0.0
        self.n_llm_calls = 0
        self._last_spend = 0.0

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
        print(
            f"[{PREFIX}][{self.uuid8}] {stage:<8} {message}{extras}",
            flush=True,
        )
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

    def payload(self, stage: str, message: str = "", data: Any = None,
                **fields: Any) -> None:
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
        print(
            f"[{PREFIX}][{self.uuid8}] {stage:<8} {message}{extras}",
            flush=True,
        )
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
        print(f"[{PREFIX}][{self.uuid8}] DETAIL  {text}", flush=True)
        self._write_record({"stage": "DETAIL", "uuid": self.uuid, "text": text})

    def warn(self, text: str) -> None:
        """Loud warning line (failures the operator must see)."""
        print(f"[{PREFIX}][{self.uuid8}] WARNING {text}", flush=True)
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
