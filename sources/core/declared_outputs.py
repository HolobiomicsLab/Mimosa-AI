"""The plan's declared outputs, carried to the verifier without threading them.

A plan step declares ``expected_outputs`` before it runs. The verifier never
sees them: it is handed a uuid and reads the workflow folder, and neither
``state_result.json`` nor any file in that folder carries the declaration. So a
step can be scored a success on 26 claims about the document the agent chose to
write, and then kill the run at the next step's dependency gate because the file
the plan asked for was never produced. That is issue #196, observed on a real
run: 0.799, accepted, then

    Cannot execute step 'dataset_and_tool_acquisition' — missing dependencies:
      ['reproduction_spec_analysis[missing_outputs:workspace/analysis/iimn_reproduction_plan.md]']

Threading ``expected_outputs`` from the planner through
``start_workflow_evolution``, the generation loop, ``IndividualRun`` and the
factory would touch five shared signatures in a repo someone else also works in.
It is not needed. The planner already passes ``original_task=step_task``, and
that same string is what the verifier keys its rubric cache on — so the
declaration can be written under that key and read back under it, with one write
in the planner and one read in the verifier and no signature changed.

The record is advisory to the rest of the system: a missing or unreadable file
yields an empty list and the verifier scores exactly as it did before.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)

FILENAME_FMT = "declared_outputs_{task_key}.json"

#: Suffixes whose content should be records rather than prose. A stub
#: here is disqualifying; a report describing stubs is not.
_DATA_SUFFIXES = {".csv", ".tsv", ".mgf", ".json", ".parquet", ".mzml", ".mztab"}


def task_key(task_text: str) -> str:
    """Stable 16-hex-char key derived from a step's task text.

    The single definition of the key, so the planner's write and the verifier's
    read cannot drift apart. ``VerifierEvaluator._task_cache_key`` delegates
    here; ``tests/declared_outputs_test.py`` pins that they agree.
    """
    return hashlib.sha256((task_text or "").encode("utf-8")).hexdigest()[:16]


def path_for(temp_root: Path | str, task_text: str) -> Path:
    return Path(temp_root) / FILENAME_FMT.format(task_key=task_key(task_text))


def record(temp_root: Path | str, task_text: str, outputs: list[str]) -> Path | None:
    """Write the declaration for one step. Returns the path, or None on failure.

    Overwrites: unlike the rubric cache, which freezes the first extraction, the
    plan is authoritative every time it is regenerated.
    """
    if not task_text or not outputs:
        return None
    path = path_for(temp_root, task_text)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"task_key": task_key(task_text),
                        "expected_outputs": list(outputs)}, indent=2),
            encoding="utf-8",
        )
        return path
    except OSError as e:
        _logger.warning("Could not record declared outputs at %s: %s", path, e)
        return None


def load(temp_root: Path | str, task_text: str) -> list[str]:
    """Declared outputs for this step, or [] when none were recorded."""
    path = path_for(temp_root, task_text)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _logger.warning("Could not read declared outputs %s: %s", path, e)
        return []
    outputs = data.get("expected_outputs") if isinstance(data, dict) else None
    if not isinstance(outputs, list):
        return []
    return [str(o) for o in outputs if str(o).strip()]


def as_claims(outputs: list[str]) -> list[dict]:
    """One mandatory claim per declared output, at maximum importance.

    Deliberately not persisted into the rubric cache: the cache freezes what was
    *extracted* from a run, and these are not extracted from anything — they are
    the plan's requirement, and they must follow the plan when it changes.
    """
    claims = []
    for output in outputs:
        raw = str(output)
        # A plan may declare a directory ("…/data/") as an output. Asserting a
        # file exists at that path would fail a maximum-importance claim on a
        # step that did exactly what was asked. Planner._verify_expected_outputs
        # already makes this distinction; the claim must make it too.
        is_directory = raw.rstrip().endswith(("/", "\\"))
        stem = Path(raw.rstrip("/\\")).name or raw
        slug = "".join(c if c.isalnum() else "_" for c in stem.lower()).strip("_")
        # "Non-empty" is satisfied by a stub. Observed live: under five of
        # these claims a run that could not obtain its inputs wrote
        # feature_table_qtof.csv whose second line reads
        # "# PLACEHOLDER: ... NO REAL DATA AVAILABLE", and all five passed at
        # importance 10 on "File exists and is non-empty (1462 bytes)". A claim
        # that a placeholder satisfies applies pressure to create the file
        # without applying any to fill it, so it must ask for content.
        #
        # But the disqualifier has to distinguish a file that *is* a placeholder
        # from a file that *reports on* one. A first version said "a file whose
        # body announces missing or unavailable data does not satisfy this
        # claim", and it failed a 292-line processing log — the most substantive
        # artefact in the workspace — because the log honestly recorded that its
        # sibling outputs were placeholders. Penalising that is exactly backwards.
        is_data = Path(raw.rstrip("/\\")).suffix.lower() in _DATA_SUFFIXES
        substance = (
            "It holds actual records — data rows, spectra, or entries — and not "
            "merely comments, headers, or placeholder text standing in for data "
            "that could not be obtained."
            if is_data else
            "It is substantive content produced by this step, not an empty stub "
            "or an unfilled template. A report that documents what was attempted "
            "and honestly records missing inputs or limitations does satisfy "
            "this claim; a file with no content of its own does not."
        )
        expectation = (
            f"A directory exists at that path in the workspace and holds at "
            f"least one file. {substance}"
            if is_directory else
            f"A file exists at that path in the workspace. {substance}"
        )
        claims.append({
            "id": f"declared_output_{slug}",
            "description": (
                f"The plan declared `{output}` as an output of this step. "
                f"{expectation}"
            ),
            "importance": 10,
            "importance_rationale": "declared by the plan before the step ran",
            "likely_relevant_files": [str(output)],
            "source": "plan_expected_outputs",
        })
    return claims
