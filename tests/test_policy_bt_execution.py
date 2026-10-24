"""Behavior checks for complete evidence copies and honest reproduction gates."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources.evaluators.policy_bt_verifier.artifacts import (
    PREVIEW_SAMPLE_BYTES,
    inventory,
    pair_workspace,
    snapshot,
)
from sources.evaluators.policy_bt_verifier.reexec import (
    FAILURE_PROMPT,
    LOG_LIMIT_BYTES,
    ReproductionInstrumentError,
    _run,
    reproduce,
)


def _answer(script: str = "reproduce.sh", outputs: list[str] | None = None) -> str:
    """Supply a semantic contract in tests of execution, not LLM judgment.

    Args: script (str), outputs (list[str] or None): Fixture entrypoint and outputs.
    Returns: str: JSON matching the public semantic-assessment contract.
    """
    return json.dumps({"script": script, "installs_requirements": True,
                       "regenerates_outputs": True,
                       "generated_outputs": ["result.data"] if outputs is None else outputs,
                       "reason": "The fixture installs its empty requirements and runs the task."})


def _workspace(root: Path, body: str) -> Path:
    """Create a real executable fixture using only an empty requirements file.

    Args: root (Path), body (str): Fixture location and commands after installation.
    Returns: Path: Workspace with a script and preexisting output evidence.
    """
    root.mkdir()
    (root / "requirements.txt").write_text("", encoding="utf-8")
    (root / "reproduce.sh").write_text(
        "set -eu\npython -m pip install --no-index -r requirements.txt\n" + body,
        encoding="utf-8")
    (root / "result.data").write_text("old result", encoding="utf-8")
    return root


def test_inventory_keeps_unknown_binary_files_and_symlink_gaps(tmp_path):
    """Unknown formats and external links remain visible without link traversal."""
    root = tmp_path / "evidence"
    root.mkdir()
    payload = b"\x00\xff\x80" * (PREVIEW_SAMPLE_BYTES + 1)
    (root / "unknown.instrument-format").write_bytes(payload)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not read", encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (root / "records").mkdir()
    (root / "records" / "run.data").write_text("repeat\n" * 400 + "last line\n", encoding="utf-8")

    entries = {item["path"]: item for item in inventory(root)}

    assert set(entries) == {"unknown.instrument-format", "linked", "records/run.data"}
    unknown = entries["unknown.instrument-format"]
    assert unknown["sha256"] == hashlib.sha256(payload).hexdigest()
    assert unknown["binary"] and unknown["preview_truncated"]
    assert unknown["bytes_read"] == len(payload)
    assert entries["linked"]["kind"] == "symlink"
    assert entries["linked"]["gap"] == "not_followed"
    text = entries["records/run.data"]
    assert text["repeated_lines_removed"] == 399
    assert "last line" in text["preview"]
    assert not text["preview_truncated"]


def test_inventory_lists_fifo_without_opening_it(tmp_path):
    """A non-regular filesystem object is a visible gap, not a blocked read."""
    os.mkfifo(tmp_path / "channel")
    entry, = inventory(tmp_path)
    assert entry["path"] == "channel"
    assert entry["kind"] == "special"
    assert entry["gap"] == "unsupported_filesystem_object"


def test_snapshot_skips_fifo_instead_of_failing_the_session(tmp_path):
    """A socket or FIFO in a workspace must not unscore every later generation."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.txt").write_text("energy -9", encoding="utf-8")
    os.mkfifo(source / "ray.sock")
    copy = snapshot(source, tmp_path / "copy")
    assert (copy / "result.txt").read_text() == "energy -9"
    assert not (copy / "ray.sock").exists()


@pytest.mark.parametrize("answer", ['{"k": 1}', '```{"k": 1}```', '```json\n{"k": 1}\n```'])
def test_decode_object_accepts_every_fence_shape(answer):
    """A one-line fenced answer used to raise IndexError and abort evaluation."""
    from sources.evaluators.policy_bt_verifier.artifacts import decode_object

    assert decode_object(answer) == {"k": 1}


def test_snapshot_rejects_self_copy_and_pair_copies_are_disposable(tmp_path):
    """Nested copies cannot recurse, and pair mutations never reach source evidence."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "artifact.bin").write_bytes(b"original")
    with pytest.raises(ValueError, match="overlap"):
        snapshot(source, source / "inside")
    assert not (source / "inside").exists()

    with pair_workspace(source, source) as first, pair_workspace(source, source) as second:
        assert first != second
        (first / "a" / "artifact.bin").write_bytes(b"mutated")
        assert (first / "b" / "artifact.bin").read_bytes() == b"original"
        assert (second / "a" / "artifact.bin").read_bytes() == b"original"
        assert (source / "artifact.bin").read_bytes() == b"original"
    assert not first.exists() and not second.exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_missing_shell_script_is_ineligible_without_llm_or_execution(tmp_path, enabled):
    """Disabling reruns cannot remove the mandatory reproduction-script gate."""
    (tmp_path / "analysis.py").write_text("raise RuntimeError('not a shell script')", encoding="utf-8")
    result = reproduce(tmp_path, {"policy_bt_reexecute": enabled}, None)
    assert not result["eligible"]
    assert result["status"] == "missing_script"
    assert "installs requirements" in result["reason"]
    assert not result["attempted"] and result["execution_dir"] is None


def test_disabled_reexecution_leaves_script_and_outputs_untouched(tmp_path):
    """A valid disabled run checks the contract but never runs shell commands."""
    root = _workspace(tmp_path / "source", "printf changed > result.data\nprintf executed > marker\n")
    before = inventory(root)
    result = reproduce(root, {"policy_bt_reexecute": False}, lambda _: _answer())
    assert result["eligible"] and result["status"] == "disabled"
    assert not result["attempted"] and result["run"] is None
    assert inventory(root) == before
    assert not (root / "marker").exists()


def test_install_only_script_contract_is_ineligible_when_execution_disabled(tmp_path):
    """A semantic failure remains a hard gate even without an actual rerun."""
    (tmp_path / "setup.sh").write_text("python -m pip install -r requirements.txt", encoding="utf-8")
    answer = json.loads(_answer("setup.sh", []))
    answer.update(regenerates_outputs=False, reason="Only installs dependencies; never runs the task.")
    result = reproduce(tmp_path, {"policy_bt_reexecute": False}, lambda _: json.dumps(answer))
    assert not result["eligible"] and result["status"] == "invalid_contract"
    assert "never runs the task" in result["reason"]


def test_unsafe_llm_output_path_is_an_instrument_error_not_a_task_loss(tmp_path):
    """A bad LLM contract must raise, never delete outside evidence or score zero."""
    outside = tmp_path / "preserve.txt"
    outside.write_text("untouched", encoding="utf-8")
    root = _workspace(tmp_path / "owned", "true\n")
    with pytest.raises(ReproductionInstrumentError, match="Unsafe generated-output path"):
        reproduce(root, {"policy_bt_reexecute": True}, lambda _: _answer(outputs=["../preserve.txt"]))
    assert outside.read_text() == "untouched"


def test_malformed_contract_answer_raises_instead_of_scoring_zero(tmp_path):
    """Provider garbage is a verifier failure, not evidence against the workflow."""
    root = _workspace(tmp_path / "owned", "true\n")
    with pytest.raises(ReproductionInstrumentError, match="Unreadable"):
        reproduce(root, {"policy_bt_reexecute": False}, lambda _: "not json")


def _evidence(prompt: str) -> dict:
    """Decode the EVIDENCE object of one contract prompt.

    Args: prompt (str): Prompt the contract stage passed to the LLM callback.
    Returns: dict: The JSON evidence after the ``EVIDENCE:`` marker.
    """
    return json.loads(prompt.split("\nEVIDENCE:\n", 1)[1])


def test_contract_prompt_does_not_grow_with_file_contents(tmp_path, monkeypatch):
    """2026-10-21 conformer run: 1,384 files with previews made a 6.3 M-character
    contract prompt (1.3 M tokens, over the model limit). Rows carry no previews,
    and rows past the cap are counted per directory, shallow paths first."""
    from sources.evaluators.policy_bt_verifier import reexec

    root = _workspace(tmp_path / "owned", "python pipeline.py\n")
    (root / "conformers" / "raw").mkdir(parents=True)
    for index in range(200):
        (root / "conformers" / "raw" / f"c{index:03d}.xyz").write_text("C 0.1 0.2 0.3\n" * 1500)
    monkeypatch.setattr(reexec, "MAX_CONTRACT_ROWS", 50)
    prompts = []
    result = reproduce(root, {"policy_bt_reexecute": False}, lambda prompt: prompts.append(prompt) or _answer())
    evidence = _evidence(prompts[0])
    assert result["status"] == "disabled"
    assert len(prompts[0]) < 30_000
    assert {"reproduce.sh", "requirements.txt", "result.data"} <= {row["path"] for row in evidence["inventory"]}
    assert all("preview" not in row for row in evidence["inventory"])
    assert evidence["unlisted_entries"] == 153
    assert evidence["unlisted_directories"] == {"conformers/raw": 153}


def test_contract_reads_referenced_files_step_by_step(tmp_path):
    """The contract no longer gets every preview, so it must be able to read
    the script run.sh calls, and list a directory, before it answers."""
    root = _workspace(tmp_path / "owned", "python pipeline.py\n")
    (root / "pipeline.py").write_text("open('result.data', 'w').write('fresh')\n")
    prompts = []

    def ask(prompt):
        prompts.append(_evidence(prompt))
        if len(prompts) == 1:
            return json.dumps({"action": "inspect", "requests": [{"path": "pipeline.py"}, {"path": "."}, {"path": "../x"}]})
        return _answer()

    result = reproduce(root, {"policy_bt_reexecute": False}, ask)
    file_page, directory_page, refused = prompts[1]["inspections"]
    assert result["status"] == "disabled"
    assert "result.data" in file_page["content"] and file_page["remaining_bytes"] == 0
    assert "pipeline.py" in [entry["name"] for entry in directory_page["entries"]]
    assert "parent" in refused["error"]
    assert prompts[1]["rounds_left"] == prompts[0]["rounds_left"] - 1


def test_contract_that_never_stops_inspecting_is_an_instrument_error(tmp_path):
    """An endless inspect loop must end, as a verifier fault, not a task loss."""
    from sources.evaluators.policy_bt_verifier.artifacts import INSPECT_ROUNDS

    root = _workspace(tmp_path / "owned", "true\n")
    calls = []
    answer = json.dumps({"action": "inspect", "requests": [{"path": "reproduce.sh"}]})
    with pytest.raises(ReproductionInstrumentError, match="kept inspecting"):
        reproduce(root, {"policy_bt_reexecute": False}, lambda _: calls.append(1) or answer)
    assert len(calls) == INSPECT_ROUNDS + 1


def _container_workspace(tmp_path: Path) -> tuple[Path, dict]:
    """Build a snapshot whose outputs came from a tool the host lacks.

    The workflow ran xtb in a container; the host rerun of reproduce.sh
    writes a partial output, empties the delivered xtb log, then fails with
    "xtb: command not found".

    Args: tmp_path (Path): Test directory.
    Returns: tuple[Path, dict]: The owned snapshot and its provenance origin.
    """
    source = tmp_path / "source"
    for folder in ("out", "crest_output", "logs"):
        (source / folder).mkdir(parents=True)
    (source / "requirements.txt").write_text("")
    (source / "input.xyz").write_text("C 0 0 0\n")
    (source / "crest_output" / "old.xyz").write_text("C 5 5 5\n")
    (source / "reproduce.sh").write_text("set -eu\npython -m pip install --no-index -r requirements.txt\n"
                                         "mkdir -p out\nprintf partial > out/partial.txt\n"
                                         "xtb input.xyz --opt > logs/xtb.log\n")
    (source / "logs" / "xtb.log").write_text("normal termination of xtb\n")
    (source / "out" / "c01.xyz").write_text("C 0.1 0 0\n")
    (source / "out" / "summary.csv").write_text("conformer,energy\nc01,-9.1\n")
    (source / "out" / "stale.xyz").write_text("C 9 9 9\n")
    (source / "out" / "kept.xyz").write_text("C 4 4 4\n")
    os.utime(source / "out" / "stale.xyz", (1e9, 1e9))
    inputs = {"input.xyz", "reproduce.sh", "requirements.txt", "crest_output/old.xyz", "out/kept.xyz"}
    baseline = {item["path"]: item["sha256"] for item in inventory(source) if item["path"] in inputs}
    root = snapshot(source, tmp_path / "owned")
    return root, {"baseline": baseline, "window": [time.time() - 60, time.time() + 1], "source": str(source)}


def _container_ask(cause="missing_host_dependency", line="xtb: command not found", named=("out",), seen=None):
    """Answer the contract, failure and provenance prompts of one rerun.

    Args: cause (str), line (str): Diagnosis cause and quoted log line;
        named (tuple): Paths the provenance answer accepts; seen (list|None):
        Receives the prompt kinds asked.
    Returns: Callable[[str], str]: The LLM callback.
    """
    from sources.evaluators.policy_bt_verifier.provenance import PROVENANCE_PROMPT

    def ask(prompt):
        if prompt.startswith(FAILURE_PROMPT):
            return json.dumps({"diagnosis": "xtb is absent on this host.", "cause": cause,
                               "missing": ["xtb"], "evidence_line": line})
        if prompt.startswith(PROVENANCE_PROMPT):
            (seen if seen is not None else []).append("provenance")
            return json.dumps({"dependency_used": True, "regenerated": list(named), "rejected": [],
                               "reason": "each structure has its xtb log"})
        return _answer(outputs=["out", "crest_output"])
    return ask


def test_missing_host_tool_trusts_outputs_the_run_wrote(tmp_path):
    """2026-10-21 conformer task: xtb ran in the tool container but cannot be
    installed on the verifier host, so every generation lost the rerun gate.
    Files the run wrote become regenerated outputs. The rerun's partial file
    and its emptied log are undone; a file older than the run is set aside;
    an output unchanged since before the run (an earlier session's
    crest_output/) is neither required nor trusted."""
    root, origin = _container_workspace(tmp_path)
    result = reproduce(root, {"policy_bt_reexec_timeout_s": 60}, _container_ask(), origin)
    facts = json.loads((root / ".policy_bt_reexec" / "execution.json").read_text())
    provenance = result["provenance"]
    assert result["eligible"] and result["status"] == "provenance_verified", result["reason"]
    assert result["files"]["regenerated"] == facts["regenerated"] == ["out/c01.xyz", "out/summary.csv"]
    assert facts["status"] == "provenance_verified" and facts["eligible"]
    assert (root / "logs" / "xtb.log").read_text() == "normal termination of xtb\n"
    assert (root / "crest_output" / "old.xyz").read_text() == "C 5 5 5\n"
    assert not (root / "out" / "partial.txt").exists() and not (root / "out" / "stale.xyz").exists()
    assert provenance["set_aside"] == ["out/stale.xyz"] and provenance["outside_window"] == 1
    assert (root / "out" / "kept.xyz").read_text() == "C 4 4 4\n"  # equal to its pre-run copy
    assert provenance["unchanged_input_outputs"] == ["crest_output"] and "xtb" in result["reason"]


@pytest.mark.parametrize(("cause", "line", "origin"), [
    ("code_error", "xtb: command not found", True),
    ("missing_requirement", "xtb: command not found", True),
    ("missing_host_dependency", "xtb was not found on any mirror", True),
    ("missing_host_dependency", "xtb: command not found", False),
])
def test_provenance_fallback_needs_a_proven_missing_host_dependency(tmp_path, cause, line, origin):
    """A code error, a fixable missing requirement, a quote absent from the
    logs, or no run window keeps the failed rerun ineligible without asking
    the provenance model."""
    root, window = _container_workspace(tmp_path)
    seen = []
    result = reproduce(root, {"policy_bt_reexec_timeout_s": 60}, _container_ask(cause, line, seen=seen),
                       window if origin else None)
    assert not result["eligible"] and result["status"] == "execution_failed"
    assert seen == []


def test_provenance_model_cannot_promote_inputs_or_old_files(tmp_path):
    """Only files written during the run window can become regenerated."""
    root, origin = _container_workspace(tmp_path)
    ask = _container_ask(named=("input.xyz", "out/stale.xyz"))
    result = reproduce(root, {"policy_bt_reexec_timeout_s": 60}, ask, origin)
    assert not result["eligible"] and result["status"] == "execution_failed"
    assert result["provenance"]["refused_names"] == ["input.xyz", "out/stale.xyz"]
    assert result["provenance"]["missing_outputs"] == ["out"]


def test_workspace_changed_after_snapshot_disables_the_fallback(tmp_path):
    """The restore copy must equal the snapshot taken before the rerun."""
    root, origin = _container_workspace(tmp_path)
    (Path(origin["source"]) / "out" / "c01.xyz").write_text("C 7 7 7\n")
    seen = []
    result = reproduce(root, {"policy_bt_reexec_timeout_s": 60}, _container_ask(seen=seen), origin)
    assert not result["eligible"] and result["provenance"]["attempted"] is False
    assert "changed after the snapshot" in result["provenance"]["reason"] and seen == []


def test_execution_facts_published_for_adapters(tmp_path):
    """Reproduction truth must reach adapters as measured fact, not log text."""
    source = _workspace(tmp_path / "source", "printf regenerated > result.data\n")
    owned = snapshot(source, tmp_path / "owned")
    record = reproduce(owned, {"policy_bt_reexec_timeout_s": 60}, lambda _: _answer())
    facts = json.loads((owned / ".policy_bt_reexec" / "execution.json").read_text())
    assert facts["status"] == record["status"] == "succeeded"
    assert facts["eligible"] is True and facts["script"] == "reproduce.sh"
    assert facts["regenerated"] == ["result.data"] and facts["missing_outputs"] == []



def test_reexecution_uses_owned_venv_and_cannot_count_stale_output(tmp_path, monkeypatch):
    """Successful reproduction must create output in a clean, credential-free venv."""
    monkeypatch.setenv("POLICY_BT_TEST_CREDENTIAL", "must-not-leak")
    command = """python - <<'PY'
import json, os, sys
from pathlib import Path
Path('result.data').write_text(json.dumps({
    'venv': sys.prefix != sys.base_prefix,
    'credential': os.environ.get('POLICY_BT_TEST_CREDENTIAL'),
    'pip_requires_venv': os.environ.get('PIP_REQUIRE_VIRTUALENV'),
    'home': os.environ.get('HOME'),
}))
PY
"""
    source = _workspace(tmp_path / "source", command)
    before = inventory(source)
    owned = snapshot(source, tmp_path / "owned")
    result = reproduce(owned, {"policy_bt_reexec_timeout_s": 60}, lambda _: _answer())
    output = json.loads((owned / "result.data").read_text())
    assert result["eligible"], result["reason"]
    assert output["venv"] and output["credential"] is None
    assert output["pip_requires_venv"] == "true"
    assert output["home"] != os.environ.get("HOME")
    assert result["files"]["regenerated"] == ["result.data"]
    assert result["preserved_outputs"] == ["result.data"]
    assert (Path(result["execution_dir"]) / "original_outputs" / "result.data").read_text() == "old result"
    assert inventory(source) == before


def test_zero_exit_without_regeneration_does_not_pass_on_old_outputs(tmp_path):
    """A zero exit cannot reuse the output that existed before reproduction."""
    source = _workspace(tmp_path / "source", "true\n")
    owned = snapshot(source, tmp_path / "owned")
    result = reproduce(owned, {"policy_bt_reexec_timeout_s": 60}, lambda _: _answer())
    assert result["returncode"] == 0
    assert not result["eligible"] and result["status"] == "outputs_missing"
    assert result["files"]["missing_outputs"] == ["result.data"]
    assert not (owned / "result.data").exists()
    assert (source / "result.data").read_text() == "old result"


def test_recreated_empty_output_folder_counts_as_regenerated(tmp_path):
    """2026-10-15: an empty ``model_checkpoints/`` (training with
    checkpoint_interval=0) made a correct ClinTox rerun "outputs_missing"."""
    body = "mkdir -p model_checkpoints\nprintf regenerated > result.data\n"
    source = _workspace(tmp_path / "source", body)
    (source / "model_checkpoints").mkdir()
    owned = snapshot(source, tmp_path / "owned")
    answer = _answer(outputs=["model_checkpoints", "result.data"])
    result = reproduce(owned, {"policy_bt_reexec_timeout_s": 60}, lambda _: answer)
    assert result["eligible"], result["reason"]
    assert result["files"]["missing_outputs"] == []


def test_output_folder_not_recreated_is_still_missing(tmp_path):
    """The gate stays strict: a declared folder the rerun never makes fails."""
    source = _workspace(tmp_path / "source", "printf regenerated > result.data\n")
    (source / "model_checkpoints").mkdir()
    owned = snapshot(source, tmp_path / "owned")
    answer = _answer(outputs=["model_checkpoints", "result.data"])
    result = reproduce(owned, {"policy_bt_reexec_timeout_s": 60}, lambda _: answer)
    assert result["status"] == "outputs_missing"
    assert result["files"]["missing_outputs"] == ["model_checkpoints"]



def test_rerun_applies_configured_runtime_without_breaking_isolation(tmp_path):
    """SAB runtime: caps reach every pip call, variables reach the script, and a
    configured variable can never replace the venv-first PATH."""
    command = 'printf "%s|%s" "$(cat "$PIP_CONSTRAINT")" "$POLICY_BT_GRADING_FLAG" > result.data\n'
    source = _workspace(tmp_path / "source", command)
    owned = snapshot(source, tmp_path / "owned")
    runtime = {"python": sys.executable, "constraints": ["numpy<2.0"],
               "env": {"POLICY_BT_GRADING_FLAG": "1", "PATH": "/hijacked"}}
    config = {"policy_bt_reexec_timeout_s": 60, "policy_bt_reexec_runtime": runtime}
    result = reproduce(owned, config, lambda _: _answer())
    assert result["eligible"], result["reason"]
    assert (owned / "result.data").read_text() == "numpy<2.0|1"
    major_minor = f"{sys.version_info.major}.{sys.version_info.minor}."
    assert result["runtime"]["python_version"].startswith(major_minor)


def _failing_workspace(tmp_path: Path) -> Path:
    """Snapshot a rerun that fails like the 2026-10-15 deepchem install."""
    body = 'echo "ERROR: 2.7.1 Requires-Python >=3.7,<3.11" >&2\necho "ERROR: No matching distribution" >&2\nexit 3\n'
    return snapshot(_workspace(tmp_path / "source", body), tmp_path / "owned")


def test_failed_rerun_reason_carries_llm_root_cause(tmp_path):
    """The gradient used to get only pip's version-list tail; the decisive
    Requires-Python line and the venv's Python version must reach the LLM."""
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        if prompt.startswith(FAILURE_PROMPT):
            return json.dumps({"diagnosis": "deepchem 2.7.1 needs Python <3.11."})
        return _answer()

    result = reproduce(_failing_workspace(tmp_path), {"policy_bt_reexec_timeout_s": 60}, ask)
    assert result["status"] == "execution_failed"
    assert "Diagnosis: deepchem 2.7.1 needs Python <3.11." in result["reason"]
    assert "Stderr tail:" in result["reason"]
    assert "Requires-Python >=3.7,<3.11" in prompts[-1]
    assert result["runtime"]["python_version"] in prompts[-1]


def test_failed_diagnosis_call_keeps_stderr_reason_and_records_error(tmp_path):
    """A provider failure on the diagnosis cannot unscore the generation."""
    def ask(prompt):
        if prompt.startswith(FAILURE_PROMPT):
            raise RuntimeError("LLM API error")
        return _answer()

    result = reproduce(_failing_workspace(tmp_path), {"policy_bt_reexec_timeout_s": 60}, ask)
    assert result["status"] == "execution_failed"
    assert "Diagnosis:" not in result["reason"] and "No matching distribution" in result["reason"]
    assert result["diagnosis_error"] == "RuntimeError: LLM API error"


def test_snapshot_keeps_only_the_marker_of_a_workspace_virtualenv(tmp_path):
    """2026-10-15: a 2.2 GB .venv pushed the recon prompt over the 8 MB limit."""
    source = tmp_path / "source"
    (source / ".venv" / "lib").mkdir(parents=True)
    (source / ".venv" / "pyvenv.cfg").write_text("version = 3.10.20\n")
    (source / ".venv" / "lib" / "site.py").write_text("x = 1\n")
    (source / "task.py").write_text("print(1)\n")
    copy = snapshot(source, tmp_path / "copy")
    assert sorted(item["path"] for item in inventory(copy)) == [".venv/pyvenv.cfg", "task.py"]


def _fake_provisioner(monkeypatch, behaviour=None) -> dict:
    """Register an importable provisioner that builds a plain venv.

    Stands in for execution_sandbox.provision_capsule_venv (the real one needs
    the full ScienceAgentBench base stack). Returns the runtime config value.
    """
    import subprocess
    import types

    def provision(capsule, venv):
        if behaviour:
            behaviour()
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        return {"path": [str(venv / "bin")], "per_program": ["numpy"], "versions": ["numpy==1.26.4"]}

    module = types.ModuleType("policy_bt_fake_provisioner")
    module.provision = provision
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return {"provisioner": f"{module.__name__}:provision"}


def test_provisioned_rerun_ignores_pip_like_the_grader(tmp_path, monkeypatch):
    """2026-10-16 drugex gen 0: run.sh's `pip install -r requirements.txt`
    gave the verifier scikit-learn 1.3.2 while the grader, which ignores
    requirements.txt, used 1.7.2. With the grader's package set provisioned,
    every pip spelling in run.sh must be a no-op and no install step is
    required by the contract."""
    body = ("pip install scikit-learn==1.3.2\npip3 install numpy\n"
            "python -m pip install -r requirements.txt\n"
            "python -c \"import os; open('result.data','w').write(os.environ['PIP_NO_INDEX'])\"\n")
    owned = snapshot(_workspace(tmp_path / "source", body), tmp_path / "owned")
    no_install = json.loads(_answer())
    no_install["installs_requirements"] = False
    config = {"policy_bt_reexec_timeout_s": 60, "policy_bt_reexec_runtime": _fake_provisioner(monkeypatch)}
    result = reproduce(owned, config, lambda _: json.dumps(no_install))
    assert result["eligible"], result["reason"]
    assert (owned / "result.data").read_text() == "1"
    stderr = (owned / ".policy_bt_reexec" / "stderr.log").read_text()
    assert stderr.count("[verifier] pip is disabled") == 4  # reproduce.sh line + 3 above
    assert "scikit-learn==1.3.2" in stderr
    assert result["runtime"]["provisioned"]["per_program"] == ["numpy"]


def test_generic_rerun_still_requires_an_install_step(tmp_path):
    """Without a provisioner (non-benchmark tasks) nothing installs packages
    for run.sh, so the install step stays mandatory."""
    owned = snapshot(_workspace(tmp_path / "source", "true\n"), tmp_path / "owned")
    no_install = json.loads(_answer())
    no_install["installs_requirements"] = False
    result = reproduce(owned, {"policy_bt_reexecute": False}, lambda _: json.dumps(no_install))
    assert result["status"] == "invalid_contract"


def test_provisioning_failure_is_an_instrument_error(tmp_path, monkeypatch):
    """The grader excludes a task whose packages cannot be installed; the
    verifier must not score it 0 either."""
    def fail():
        raise RuntimeError("Required package install failed (exit 1) for ['qsprpred']")

    owned = snapshot(_workspace(tmp_path / "source", "true\n"), tmp_path / "owned")
    config = {"policy_bt_reexec_timeout_s": 60, "policy_bt_reexec_runtime": _fake_provisioner(monkeypatch, fail)}
    with pytest.raises(ReproductionInstrumentError, match="qsprpred"):
        reproduce(owned, config, lambda _: _answer())


def test_timeout_kills_descendants_before_they_write(tmp_path):
    """The timeout stops a descendant, not just its waiting parent shell."""
    script = tmp_path / "wait.sh"
    script.write_text("(/bin/sleep 0.8; printf survived > late-output) &\nwait\n", encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    result = _run(["/bin/bash", str(script)], tmp_path,
                  {"env": {"PATH": os.defpath}, "logs": logs, "timeout": 0.15})
    assert result["timed_out"]
    assert result["returncode"] != 0
    time.sleep(0.9)
    assert not (tmp_path / "late-output").exists()


def test_bounded_logs_drain_both_streams_and_keep_diagnostic_tail(tmp_path):
    """Output flooding cannot block execution or grow retained log files unbounded."""
    size = LOG_LIMIT_BYTES + 4096
    source = f"import sys; sys.stdout.buffer.write(b'x'*{size}+b'END'); sys.stderr.buffer.write(b'y'*{size}+b'ERR')"
    result = _run([sys.executable, "-c", source], tmp_path,
                  {"env": {"PATH": os.defpath}, "logs": tmp_path, "timeout": 10})
    assert result["returncode"] == 0 and not result["timed_out"]
    for stream, ending in (("stdout", "END"), ("stderr", "ERR")):
        capture = result[stream]
        assert capture["bytes"] == size + 3
        assert capture["truncated"] and capture["tail"].endswith(ending)
        assert Path(capture["path"]).stat().st_size == LOG_LIMIT_BYTES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
