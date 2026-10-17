"""Themed operator console for the Policy-BT verifier, in the Mimosa CLI style.

Used by PolicyBTVerifierEvaluator: every stage event also lands in the JSON
sink ``verifier_policy_bt.log`` (see evaluator._log); this module renders the
human-facing view — reproduction outcome, which files each pair compared, the
generated policy code, Bradley-Terry standings, and the gradient preview.
Styling follows the amber/ember CLI theme and turns itself off when stdout is
not a terminal or NO_COLOR is set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TextIO

WIDTH = 110
BAR_CELLS = 10
PREVIEW_LINES = 10
PREVIEW_FILES = 6
GRADIENT_LINES = 8

__all__ = ["Reporter", "pair_file_diff"]


def _sgr(params: str, on: bool) -> str:
    """Return one SGR escape, or the empty string when styling is off.

    Args: params (str): SGR parameter string; on (bool): Styling enabled flag.
    Returns: str: The escape sequence, or "".
    """
    return f"\033[{params}m" if on else ""


def _short(identifier: str) -> str:
    """Shorten a generation identifier to its last dash-separated segment.

    Args: identifier (str): Full workflow identifier.
    Returns: str: At most the final 12 characters, still unique per session.
    """
    tail = identifier.rsplit("_", 1)[-1]
    return tail if len(tail) <= 12 else tail[:8] + "…" + tail[-4:]


def _side_files(root: Path, side: str) -> set[str]:
    """Collect one pair side's relative file paths without following links.

    Args: root (Path): Pair directory; side (str): "a" or "b".
    Returns: set[str]: Relative paths of every regular file on that side.
    """
    base = root / side
    found = set()
    for directory, _dirs, files in os.walk(base):
        found |= {os.path.relpath(Path(directory) / name, base) for name in files}
    return found


def pair_file_diff(root: Path) -> dict:
    """Measure which files a pair comparison actually had on each side.

    Called by the evaluator inside the disposable pair directory. Splits both
    sides' relative paths into common, a-only, and b-only sets, each capped
    to PREVIEW_FILES entries with exact per-set sizes.

    Args:
        root (Path): Pair directory containing the ``a`` and ``b`` copies.

    Returns:
        dict: Capped ``common``/``only_a``/``only_b`` name lists plus exact
        per-set sizes under ``counts``.
    """
    files_a, files_b = _side_files(root, "a"), _side_files(root, "b")
    common, only_a, only_b = files_a & files_b, files_a - files_b, files_b - files_a
    return {"common": sorted(common)[:PREVIEW_FILES],
            "only_a": sorted(only_a)[:PREVIEW_FILES],
            "only_b": sorted(only_b)[:PREVIEW_FILES],
            "counts": {"common": len(common), "only_a": len(only_a),
                       "only_b": len(only_b)}}


def records(pairs: list[dict]) -> dict:
    """Tally wins, losses, and ties per generation from pair verdicts.

    Args: pairs (list[dict]): Committed pair verdicts in a/b/winner form.
    Returns: dict: UUID to (wins, losses, ties) tuple.
    """
    tallies: dict[str, list[int]] = {}
    for pair in pairs:
        for uuid in (pair["a"], pair["b"]):
            tallies.setdefault(uuid, [0, 0, 0])
        if pair["winner"] == "tie":
            tallies[pair["a"]][2] += 1
            tallies[pair["b"]][2] += 1
        else:
            won, lost = (pair["a"], pair["b"]) if pair["winner"] == "a" \
                else (pair["b"], pair["a"])
            tallies[won][0] += 1
            tallies[lost][1] += 1
    return {uuid: tuple(counts) for uuid, counts in tallies.items()}


class Reporter:
    """Render the verifier's operator view in the amber Mimosa CLI theme.

    Created once per evaluate() call; every emitter prints immediately. Color
    is forced off when stdout is not a TTY or NO_COLOR is set, so captured and
    piped output stays clean plain text with the same layout.
    """

    def __init__(self, uuid: str, color: bool | None = None,
                 stream: TextIO | None = None):
        """Bind the generation, palette, and output stream.

        Args:
            uuid (str): Workflow identifier of the generation being evaluated.
            color (bool | None): Force styling on/off; None auto-detects TTY.
            stream (TextIO | None): Output stream; None uses sys.stdout.

        Returns:
            None.
        """
        enabled = color if color is not None else (
            sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
        self.out = stream or sys.stdout
        self.amber = _sgr("38;5;214", enabled)
        self.ember = _sgr("38;5;130", enabled)
        self.white = _sgr("97", enabled)
        self.grey = _sgr("38;5;245", enabled)
        self.bold = _sgr("1", enabled)
        self.green = _sgr("92", enabled)
        self.red = _sgr("91", enabled)
        self.reset = _sgr("0", enabled)
        self._emit(f"\n  {self.ember}┌─{self.reset}{self.amber}{self.bold}"
                   f" POLICY-BT ▏{_short(uuid)}{self.reset}")

    def _emit(self, text: str) -> None:
        """Write one rendered line to the bound stream.

        Args: text (str): Fully styled line.
        Returns: None.
        """
        print(text, file=self.out)

    def stage(self, tag: str, message: str) -> None:
        """Print one minor event line, dim, with a bracketed stage tag.

        Args: tag (str): Short stage label, for example SNAPSHOT; message
            (str): One-line fact.
        Returns: None.
        """
        self._emit(f"  {self.ember}[{tag:<7}]{self.reset} "
                   f"{self.grey}{message}{self.reset}")

    def kv(self, label: str, value: str, accent: bool = False) -> None:
        """Print a dot-leader key/value row inside the frame.

        Args: label (str): Row key; value (str): Row value; accent (bool):
            Render the value in bright white instead of grey.
        Returns: None.
        """
        dots = "·" * max(2, 26 - len(label))
        color = self.white if accent else self.grey
        self._emit(f"         {self.white}{label:<10}{self.reset} {self.ember}"
                   f"{dots}{self.reset} {color}{value}{self.reset}")

    def execution(self, record: dict) -> None:
        """Print the reproduction outcome block for one generation.

        Args: record (dict): The persistable execution record from reexec.
        Returns: None.
        """
        good = record.get("eligible")
        mark = f"{self.green}✓ REPRODUCIBLE{self.reset}" if good \
            else f"{self.red}✗ {record.get('status', 'unknown')}{self.reset}"
        self.kv("reproduce", mark, accent=True)
        self.kv("script", str(record.get("script")))
    def bundle(self, bundle: dict, new_adapters: list[str]) -> None:
        """Print the frozen criteria table and previews of generated code.

        Tolerates partial bundles (tests inject minimal fakes): missing or
        non-list criteria and missing sources simply skip their blocks.

        Args: bundle (dict): Prepared task bundle; new_adapters (list[str]):
            Adapter SHA ids generated by this evaluation.
        Returns: None.
        """
        self._emit(f"  {self.ember}├─{self.reset} {self.amber}{self.bold}"
                   f"TASK POLICY{self.reset} {self.grey}criteria are frozen "
                   f"for this task session{self.reset}")
        criteria = bundle.get("criteria")
        for criterion in sorted([c for c in criteria if isinstance(c, dict)], key=lambda item: item["stage"]) if isinstance(criteria, list) else []:
            arrow = {"higher": "↑ higher-is-better",
                     "lower": "↓ lower-is-better",
                     "satisfy": "= pass/fail"}.get(criterion["direction"], "?")
            self.kv(f"s{criterion['stage']} {criterion['id']}", arrow)
        adapters = bundle.get("adapters") or {}
        for adapter_id in [i for i in new_adapters if i in adapters][:2]:
            self.preview("schema adapter", adapter_id[:12], adapters[adapter_id])
        if bundle.get("core_source"):
            self.preview("comparison policy", "core", bundle["core_source"])

    def preview(self, kind: str, name: str, source: str) -> None:
        """Print a numbered, capped preview of one generated source.

        Args: kind (str): "comparison policy" or "schema adapter";
            name (str): Display name; source (str): Generated Python source.
        Returns: None.
        """
        lines = source.strip().splitlines()
        shown, hidden = lines[:PREVIEW_LINES], len(lines) - PREVIEW_LINES
        self._emit(f"         {self.grey}┄ {kind} {name} — "
                   f"{len(lines)} lines{self.reset}")
        for number, line in enumerate(shown, 1):
            self._emit(f"         {self.ember}{number:>3}│{self.reset} "
                       f"{self.white}{line[:WIDTH - 18]}{self.reset}")
        if hidden > 0:
            self._emit(f"         {self.grey}┄ … {hidden} more lines"
                       f"{self.reset}")

    def pair(self, left: str, right: str, verdict: dict, files: dict) -> None:
        """Print one pair comparison: files on both sides and the verdict.

        Args: left (str), right (str): Generation identifiers at a and b;
            verdict (dict): Validated pair verdict; files (dict): Output of
            pair_file_diff for this pair.
        Returns: None.
        """
        winner = verdict["winner"]
        label = {"a": f"{self.green}A WINS{self.reset}",
                 "b": f"{self.green}B WINS{self.reset}",
                 "tie": f"{self.grey}TIE{self.reset}"}[winner]
        channel = verdict.get("channel", "code")
        self._emit(f"  {self.ember}├─{self.reset} {self.bold}{_short(left)}"
                   f"{self.reset} {self.grey}vs{self.reset} {self.bold}"
                   f"{_short(right)}{self.reset}  →  {label}  "
                   f"{self.grey}[{channel}·stage {verdict['stage']}]"
                   f"{self.reset}")
        counts = files["counts"]
        self._file_list("both", files["common"], counts["common"])
        self._file_list("a only", files["only_a"], counts["only_a"])
        self._file_list("b only", files["only_b"], counts["only_b"])

    def _file_list(self, label: str, shown: list[str], total: int) -> None:
        """Print one capped file-set row for a pair side.

        Args: label (str): Row key; shown (list[str]): Preview paths;
            total (int): Full set size for the "+N more" marker.
        Returns: None.
        """
        extra = f" +{total - len(shown)} more" if total > len(shown) else ""
        names = ", ".join(shown) if shown else "—"
        self._emit(f"         {self.grey}{label:<7} {total:>3} · {names}"
                   f"{extra}{self.reset}")

    def visual(self, paths: dict, verdict: dict | None) -> None:
        """Print the vision tie-break attempt and its outcome.

        Args: paths (dict): Side to selected relative image paths;
            verdict (dict | None): Visual result; a tie (orders disagreed or
            both tied) keeps the code tie, ``None`` means the channel failed.
        Returns: None.
        """
        decisive = bool(verdict) and verdict.get("winner") != "tie"
        outcome = "kept" if decisive else "no agreed winner → tie stands"
        self.kv("vision", f"{len(paths['a'])}×{len(paths['b'])} figures · "
                f"{outcome}")

    def standings(self, ratings: dict, pairs: list[dict], current: str,
                  arrivals: dict) -> None:
        """Print the Bradley-Terry table with bars, records, and deltas.

        Args: ratings (dict): UUID to current rating; pairs (list[dict]): All
            committed pair verdicts; current (str): This evaluation's UUID;
            arrivals (dict): UUID to arrival_reward for the delta column.
        Returns: None.
        """
        pair_word = "pair" if len(pairs) == 1 else "pairs"
        self._emit(f"  {self.ember}├─{self.reset} {self.amber}{self.bold}"
                   f"BRADLEY-TERRY STANDINGS{self.reset} {self.grey}"
                   f"{len(ratings)} generations · {len(pairs)} {pair_word}"
                   f"{self.reset}")
        tallies = records(pairs)
        order = sorted(ratings, key=lambda uuid: -ratings[uuid])
        for rank, uuid in enumerate(order, 1):
            mark = f" {self.amber}◀ this{self.reset}" if uuid == current else ""
            self._row(rank, uuid, ratings[uuid], tallies.get(uuid, (0, 0, 0)),
                      arrivals.get(uuid), mark)

    def _row(self, rank: int, uuid: str, rating: float, record: tuple,
             arrival: float | None, mark: str) -> None:
        """Print one standings row: rank, bar, rating, record, delta.

        Args: rank (int): 1-based rank; uuid (str): Generation; rating
            (float): Current BT rating; record (tuple): (wins, losses, ties);
            arrival (float | None): Reward at arrival; mark (str): Row accent.
        Returns: None.
        """
        filled = round(rating * BAR_CELLS)
        bar = f"{self.amber}{'▰' * filled}{self.reset}" \
              f"{self.ember}{'▱' * (BAR_CELLS - filled)}{self.reset}"
        wins, losses, ties = record
        delta = "" if arrival is None else f"  Δ{rating - arrival:+.3f}"
        self._emit(f"  {self.grey}{rank:>2}{self.reset}  {bar}  "
                   f"{self.white}{rating:.3f}{self.reset}  "
                   f"{self.green}{wins}W{self.reset} "
                   f"{self.red}{losses}L{self.reset} "
                   f"{self.grey}{ties}T{self.reset}  "
                   f"{self.bold}{_short(uuid)}{self.reset}"
                   f"{self.grey}{delta}{self.reset}{mark}")

    def gradient(self, text: str) -> None:
        """Print a capped preview of the gradient that steers the next mutation.

        Args: text (str): Full abstracted textual gradient.
        Returns: None.
        """
        self._emit(f"  {self.ember}├─{self.reset} {self.amber}{self.bold}"
                   f"GRADIENT → next mutation{self.reset}")
        lines = text.strip().splitlines()[:GRADIENT_LINES]
        for number, line in enumerate(lines, 1):
            self._emit(f"         {self.ember}{number:>3}│{self.reset} "
                       f"{self.white}{line[:WIDTH - 16]}{self.reset}")

    def error(self, message: str) -> None:
        """Print the instrument-failure block for this generation.

        Args: message (str): Exception text recorded as unscored.
        Returns: None.
        """
        self._emit(f"  {self.ember}├─{self.reset} {self.red}✗ VERIFIER "
                   f"INSTRUMENT ERROR — generation left unscored{self.reset}")
        self._emit(f"         {self.grey}{message[:WIDTH - 12]}{self.reset}")

    def close(self) -> None:
        """Print the frame's closing tick.

        Args: None. Returns: None.
        """
        self._emit(f"  {self.ember}{'─' * (WIDTH - 1)}┘{self.reset}")


def _demo() -> None:
    """Render every reporter block with fixture data, no LLM and no files.

    Args: None. The demo forces styling on to show the intended look.
    Returns: None.
    """
    report = Reporter("20261012_181428_83188bee", color=True)
    report.stage("SNAPSHOT", "10 files · colors forced for demo")
    report.execution({"eligible": True, "status": "succeeded",
                      "script": "run.sh", "duration_s": 3.2,
                      "files": {"regenerated": ["results.txt"], "missing_outputs": []}})
    report.bundle({"criteria": [
        {"id": "valid_conformation", "stage": 1, "direction": "satisfy"},
        {"id": "min_energy", "stage": 2, "direction": "lower"}],
        "core_source": 'def compare(a, b):\n    ea = a["values"]["min_energy"]\n    if ea != b["values"]["min_energy"]:\n        return {"winner": "a", "stage": 2}\n    return {"winner": "tie", "stage": 0}\n',
        "adapters": {"a1b2c3d4e5f6": 'def normalize():\n    text = read_text("results.txt")\n    return {"values": {"min_energy": -9}, "evidence": []}\n'}},
        ["a1b2c3d4e5f6"])
    report.pair("20261012_175447_9551c99a", "20261012_181428_83188bee",
                {"winner": "b", "stage": 2, "channel": "code"},
                {"common": ["requirements.txt", "run.sh"], "only_a": ["hp_solver.py"],
                 "only_b": ["mc_pull.py"], "counts": {"common": 2, "only_a": 3, "only_b": 3}})
    report.visual({"a": ["fold_a.png"], "b": ["fold_b.png"]}, None)
    report.standings({"9551c99a": 0.357, "83188bee": 0.573},
                     [{"a": "9551c99a", "b": "83188bee", "winner": "b"}],
                     "83188bee", {"9551c99a": 0.584, "83188bee": 0.5})
    report.gradient("Your run reports energy -9, which is lower (better) than\nthe other run's -8.")
    report.error("demo error block")
    report.close()


if __name__ == "__main__":
    _demo()
