#!/usr/bin/env python3
"""
Tests for the E7 F2/F3 provisioning fixes in execution_sandbox.py.

Background (E7 venv divergence audit, experiments_verifiers/results/
E7_venv_divergence_2026-09-21.md): four asb_full_iterative_1 capsules were
EXCLUDED as infra because of unpinned/unmapped provisioning:
  - WaterQuality (task 77): pipreqs silently dropped ``import skgstat``
    (no IMPORT_NAME_REMAP entry; PyPI project is scikit-gstat) ->
    "ModuleNotFoundError: No module named 'skgstat'".
  - cgcnn (task 97): deepchem pulled unpinned torchdata -> 0.11.0, where
    ``torchdata.datapipes`` (imported by deepchem at module load) no longer
    exists -> "ModuleNotFoundError: No module named 'torchdata.datapipes'".
  - modnet x2 (tasks 101/102): unpinned tensorflow-probability resolved to
    >= 0.25, which requires TF >= 2.18 while tensorflow is pinned <= 2.17.0
    -> "This version of TensorFlow Probability requires TensorFlow version
    >= 2.18".
  - (secondary) mountainLion2 reported ``osgeo`` to pipreqs; PyPI only has
    gdal, another missing IMPORT_NAME_REMAP entry.

Fixes under test (all offline; no venv, no network):
  F2: IMPORT_NAME_REMAP gains "skgstat": "scikit-gstat" and "osgeo": "gdal".
  F3: PINNED_CONSTRAINTS gains "torchdata<0.10" and
      "tensorflow-probability<=0.24.0".

These tests intentionally assert the presence of the pins/remaps and the
mapping behaviour, not that pip installs anything.
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation.execution_sandbox import (
    IMPORT_NAME_REMAP,
    PINNED_CONSTRAINTS,
    SPECIAL_CASE_INSTALLS,
    ExecutionSandbox,
)

# --- F2: IMPORT_NAME_REMAP entries -------------------------------------------


def test_remap_has_skgstat_to_scikit_gstat():
    assert IMPORT_NAME_REMAP.get("skgstat") == "scikit-gstat"


def test_remap_has_osgeo_to_gdal():
    assert IMPORT_NAME_REMAP.get("osgeo") == "gdal"


def test_apply_dependency_rules_maps_skgstat():
    packages, present = ExecutionSandbox._apply_dependency_rules("skgstat\ngeopandas\n")
    assert "scikit-gstat" in packages
    assert "skgstat" not in packages
    assert "geopandas" in packages
    assert present == ["skgstat", "geopandas"]


def test_apply_dependency_rules_maps_osgeo():
    packages, present = ExecutionSandbox._apply_dependency_rules("osgeo\n")
    assert packages == ["gdal"]
    assert present == ["osgeo"]


def test_remap_lookup_is_case_insensitive_like_discovery_path():
    # _apply_dependency_rules lowercases before the remap lookup; keep the
    # keys lowercase so pipreqs' reported casing can never bypass them.
    for key in ("skgstat", "osgeo"):
        assert key == key.lower()
        assert key in IMPORT_NAME_REMAP


# --- F3: PINNED_CONSTRAINTS pins ---------------------------------------------


def test_constraints_pin_torchdata_below_0_10():
    # torchdata 0.10 removed torchdata.datapipes, which deepchem imports at
    # module load; without the pin pip resolves 0.11.0 (E7 surviving venvs).
    assert any(c.startswith("torchdata<0.10") for c in PINNED_CONSTRAINTS), PINNED_CONSTRAINTS


def test_constraints_pin_tfp_at_most_0_24():
    # tfp >= 0.25 requires TF >= 2.18, conflicting with tensorflow<=2.17.0.
    assert any(
        c.startswith("tensorflow-probability<=0.24.0") for c in PINNED_CONSTRAINTS
    ), PINNED_CONSTRAINTS


def test_constraints_still_pin_the_base_stack_era():
    # The two new pins must not disturb the authors' era caps they pair with.
    assert any(c.startswith("tensorflow<=2.17.0") for c in PINNED_CONSTRAINTS)
    assert any(c.startswith("torch<=2.3.0") for c in PINNED_CONSTRAINTS)


# --- fingerprint correctness (pins must change the shared-venv key) ----------


def test_fingerprint_changes_with_constraints():
    sb = object.__new__(ExecutionSandbox)
    sb.base_packages = ["numpy<2.0"]
    fp_before = sb._venv_fingerprint()

    import sources.benchmark_evaluation.execution_sandbox as es

    original = es.PINNED_CONSTRAINTS
    try:
        es.PINNED_CONSTRAINTS = original + ["torchdata<0.10"]
        fp_after = sb._venv_fingerprint()
    finally:
        es.PINNED_CONSTRAINTS = original

    assert fp_before != fp_after, "adding a pin must rotate the shared-venv fingerprint"
    assert len(fp_before) == 16 and len(fp_after) == 16


# --- F2 enforcement: pipreqs drops remap-needed imports (AST fallback) -------


def _bare_sandbox(capsule_dir):
    """ExecutionSandbox without the heavy venv/env setup done in __init__."""
    sb = object.__new__(ExecutionSandbox)
    import logging

    sb.logger = logging.getLogger("tests.sandbox_provisioning_pins")
    sb.capsule_path = Path(capsule_dir)
    return sb


def test_ast_fallback_recovers_dropped_skgstat(tmp_path):
    # pipreqs queries PyPI under the IMPORT name and silently drops "skgstat"
    # (reproduced live, E7) — so the remap never sees it. The AST fallback
    # must recover the mapped PyPI name.
    cap = tmp_path / "capsule"
    cap.mkdir()
    (cap / "prog.py").write_text("import os\nimport skgstat as skg\nprint(skg)\n")
    sb = _bare_sandbox(cap)
    assert sb._remapped_imports_pipreqs_missed(["os"]) == ["scikit-gstat"]


def test_ast_fallback_recovers_dropped_osgeo(tmp_path):
    cap = tmp_path / "capsule"
    cap.mkdir()
    (cap / "prog.py").write_text("from osgeo import gdal\ngdal.UseExceptions()\n")
    sb = _bare_sandbox(cap)
    assert sb._remapped_imports_pipreqs_missed([]) == ["gdal"]


def test_ast_fallback_skips_imports_pipreqs_reported(tmp_path):
    cap = tmp_path / "capsule"
    cap.mkdir()
    (cap / "prog.py").write_text("import skgstat\nfrom osgeo import gdal\n")
    sb = _bare_sandbox(cap)
    # both reported (any casing) -> nothing to add
    assert sb._remapped_imports_pipreqs_missed(["SKGSTAT", "osgeo"]) == []


def test_ast_fallback_skips_unmapped_and_dotted_imports(tmp_path):
    cap = tmp_path / "capsule"
    cap.mkdir()
    (cap / "prog.py").write_text(
        "import torchdata.datapipes\nimport numpy\nfrom pathlib import Path\n"
    )
    sb = _bare_sandbox(cap)
    assert sb._remapped_imports_pipreqs_missed([]) == []


def test_ast_fallback_survives_unparsable_file(tmp_path):
    cap = tmp_path / "capsule"
    cap.mkdir()
    (cap / "broken.py").write_text("def broken(:\n")
    (cap / "good.py").write_text("import skgstat\n")
    sb = _bare_sandbox(cap)
    assert sb._remapped_imports_pipreqs_missed([]) == ["scikit-gstat"]


# --- companion fixes surfaced by the live re-grade ----------------------------


def test_extra_deps_give_deepchem_pyyaml():
    # deepchem 2.8.0 imports yaml (deepchem/data/data_loader.py:23) without
    # declaring pyyaml; the cgcnn VER run died on unpickling a dc NumpyDataset.
    packages, present = ExecutionSandbox._apply_dependency_rules("deepchem\n")
    assert "pyyaml" in packages and present == ["deepchem"]


def test_subprocess_env_sets_legacy_keras(monkeypatch):
    # tf_keras<=2.17.0 is only honored with TF_USE_LEGACY_KERAS=1; without it
    # Keras-3 removes tf.keras.optimizers.legacy (modnet 0.4.1) and rejects
    # BatchNormalization kwargs (E7 aquatic/brain_blood).
    monkeypatch.delenv("TF_USE_LEGACY_KERAS", raising=False)
    sb = object.__new__(ExecutionSandbox)
    sb.venv_path = Path("/nonexistent-venv")
    sb.cpu_only = True
    env = sb._subprocess_env()
    assert env["TF_USE_LEGACY_KERAS"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""


# --- E37 R6: deepchem provisioning regression (clintox) ------------------------


def test_constraints_pin_deepchem_at_most_2_7_1():
    # The B2 provisioning fix capped deepchem at the SAB-era 2.7.1 (the
    # release the dgl special-case install and the torchdata<0.10 pin were
    # validated against); the cap went missing before E37 and clintox's
    # grading venv lost a working deepchem.
    assert any(
        c.startswith("deepchem<=2.7.1") for c in PINNED_CONSTRAINTS
    ), PINNED_CONSTRAINTS


def test_pipreqs_analysis_copy_includes_nested_worker_scripts(tmp_path):
    # E37 R6 (clintox): the deepchem import lived in a worker script under
    # a capsule subdirectory; deps_analysis only received top-level *.py,
    # so pipreqs never saw the import and the grading venv shipped without
    # deepchem. The analysis copy must recurse (structure preserved so
    # same-named files cannot collide) while data files stay out.
    cap = tmp_path / "capsule"
    (cap / "workers").mkdir(parents=True)
    (cap / "main.py").write_text("import pandas\n")
    (cap / "workers" / "train.py").write_text("import deepchem\n")
    (cap / "workers" / "util.py").write_text("import os\n")
    (cap / "data.csv").write_text("smiles,tox\nCCO,1\n")
    sb = _bare_sandbox(cap)
    dest = tmp_path / "deps_analysis"
    sb._copy_py_sources_for_analysis(dest)
    assert (dest / "main.py").exists()
    assert (dest / "workers" / "train.py").exists()
    assert (dest / "workers" / "util.py").exists()
    assert not (dest / "data.csv").exists()
    assert sorted(p.name for p in dest.rglob("*.py")) == [
        "main.py",
        "train.py",
        "util.py",
    ]


def test_ast_fallback_scans_nested_worker_scripts(tmp_path):
    # The AST fallback had the same top-level-only blind spot; a worker
    # script's remap-needed import must be recovered too.
    cap = tmp_path / "capsule"
    (cap / "tools").mkdir(parents=True)
    (cap / "main.py").write_text("import os\n")
    (cap / "tools" / "helper.py").write_text("import skgstat\n")
    sb = _bare_sandbox(cap)
    assert sb._remapped_imports_pipreqs_missed([]) == ["scikit-gstat"]


def test_discovery_rules_provision_deepchem_chain_end_to_end():
    # What the grading venv receives for a capsule pipreqs reports deepchem
    # for: the package itself, its undeclared pyyaml runtime dep, and the
    # dgl special-case install — all capped by the deepchem<=2.7.1 pin.
    packages, present = ExecutionSandbox._apply_dependency_rules("deepchem\n")
    assert packages == ["deepchem", "pyyaml"]
    assert present == ["deepchem"]
    assert "deepchem" in SPECIAL_CASE_INSTALLS
