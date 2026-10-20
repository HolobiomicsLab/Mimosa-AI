"""Transfer-time input manifest that tells the verifier which inputs changed."""

from pathlib import Path

from sources.utils.input_manifest import (
    STATUS_TAMPERED,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    inputs_root,
    record_transfer,
    verify_inputs,
)


def _transfer(tmp_path: Path, folder: str, text: str) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    (workspace / folder).mkdir(parents=True, exist_ok=True)
    (workspace / folder / "test.csv").write_text(text, encoding="utf-8")
    root = inputs_root(tmp_path / "workflows", workspace)
    record_transfer(root, workspace, folder)
    return root, workspace


def test_workspace_without_a_transfer_record_is_unverified(tmp_path):
    result = verify_inputs(inputs_root(tmp_path / "workflows", tmp_path), tmp_path)
    assert result == {"status": STATUS_UNVERIFIED, "tampered": [], "data": None}


def test_untouched_inputs_verify_and_keep_a_pristine_table_copy(tmp_path):
    root, workspace = _transfer(tmp_path, "hiv", "smiles,HIV_active\nC,1\n")
    result = verify_inputs(root, workspace)
    assert result["status"] == STATUS_VERIFIED and result["tampered"] == []
    assert (Path(result["data"]) / "hiv" / "test.csv").read_text(
        encoding="utf-8"
    ) == "smiles,HIV_active\nC,1\n"


def test_rewritten_or_deleted_input_is_named_and_the_copy_keeps_transfer_bytes(
    tmp_path,
):
    """HIV run_oct_18: a workflow overwrote the hidden test labels with real ones."""
    root, workspace = _transfer(tmp_path, "hiv", "smiles,HIV_active\nC,1\n")
    (workspace / "hiv" / "train.csv").write_text(
        "smiles,HIV_active\nO,0\n", encoding="utf-8"
    )
    record_transfer(root, workspace, "hiv")
    (workspace / "hiv" / "test.csv").write_text(
        "smiles,HIV_active\nC,0\n", encoding="utf-8"
    )
    (workspace / "hiv" / "train.csv").unlink()
    result = verify_inputs(root, workspace)
    assert result["status"] == STATUS_TAMPERED
    assert result["tampered"] == ["hiv/test.csv", "hiv/train.csv"]
    assert (Path(result["data"]) / "hiv" / "test.csv").read_text(
        encoding="utf-8"
    ) == "smiles,HIV_active\nC,1\n"


def test_new_transfer_replaces_the_previous_task_record(tmp_path):
    """csv_mode reuses one workspace; task B must never be checked against task A."""
    root, workspace = _transfer(tmp_path, "dkpes", "index,Signal\n1,-999\n")
    (workspace / "dkpes" / "test.csv").unlink()
    (workspace / "dkpes").rmdir()
    _transfer(tmp_path, "clintox", "smiles,CT_TOX\nC,1\n")
    result = verify_inputs(root, workspace)
    assert result["status"] == STATUS_VERIFIED
    assert not (Path(result["data"]) / "dkpes").exists()


def test_record_of_another_task_is_unverified_for_a_snapshot_without_its_folder(
    tmp_path,
):
    """A deleted or never-transferred dataset folder gives no verdict."""
    root, _ = _transfer(tmp_path, "dkpes", "index,Signal\n1,-999\n")
    (tmp_path / "snapshot" / "other").mkdir(parents=True)
    assert verify_inputs(root, tmp_path / "snapshot")["status"] == STATUS_UNVERIFIED


def test_corrupt_record_is_unverified_with_its_error_instead_of_raising(tmp_path):
    """A truncated manifest must never turn a generation into an instrument error."""
    root, workspace = _transfer(tmp_path, "hiv", "smiles,HIV_active\nC,1\n")
    (root / "manifest.json").write_text('{"folder": "hiv", "fil', encoding="utf-8")
    result = verify_inputs(root, workspace)
    assert (result["status"], result["data"]) == (STATUS_UNVERIFIED, None)
    assert result["error"].startswith("JSONDecodeError")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
