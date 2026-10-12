"""Deterministic workspace digest for the policy_bt verifier (plan §1 P2a).

Port of ``experiments_verifiers/harness/e44_digest.py`` (gold-free, common
schema) adapted to production inputs: instead of reading frozen offline
scratch trees, :func:`build_digest` consumes one workspace directory (the
re-execution sandbox copy, a ``/tmp`` run snapshot, or the live workspace)
plus that generation's re-execution ledger and emits a compact JSON dict
describing WHAT the workspace delivered and WHAT a fresh execution proved:

- ``exec``: ledger facts (exit code, wall time, timeout, error signature,
  produced_pred_results, log tail);
- ``deliverables``: one record per deliverable file with the
  ``produced_this_run`` stale-artifact flag from the ledger's fresh set;
- ``tables`` / ``arrays`` / ``figures``: parsed prediction results;
- ``log_metrics``: metric lines scraped from run logs;
- ``json_metrics``: flattened numeric leaves of workspace JSON files;
- ``inputs``: in-workspace train/test input stats (target distributions);
- ``train_used``: rows of training data actually consumed.

Determinism: sorted iteration, fixed caps, rounded floats, no timestamps.
No gold, no labels, no rewards ever enter the digest (gold firewall).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .recon import is_infra_name

MAX_NUM_COLS = 12
MAX_JSON_KEYS = 80
MAX_METRIC_LINES = 40
MAX_TRAIN_COLS = 10
JSON_FILE_CAP = 64 * 1024
ID_COLS_TRIED = 2

#: JSON files larger than JSON_FILE_CAP still contribute top-level
#: scalars (P5.6) but are never PARSED above this hard bound.
MAX_JSON_SCAN = 8 << 20
# METRIC_MENU removed 2026-10-10: LLM-based extraction replaces regex
# scraping for text metrics (operator directive: no regex, ever).
# The metric resolver (metrics.py) now uses an LLM call to extract
# numeric values from raw text output when json_metrics aliases fail.
CONTEXT_LEN = 34
#: 1-D numeric lists surface as metric leaves ONLY under keys that name
#: a metric — "coordinates"/"orderings"/plain index arrays never do.
_LIST_METRIC_KEY = re.compile(
    r"metric|score|auc|rmse|mae|mse|r2|loss|acc\b|accuracy|f1|energ|"
    r"count|degener|mean|std\b|avg|sum\b|reward|entropy|silhouette",
    re.I,
)

_GOAL_FILE_RE = re.compile(
    r"(?<![\w/.])"
    r"[\w./-]+\.(?:sh|txt|py|json|csv|tsv|md|yml|yaml|xml|npy|npz"
    r"|png|jpg|jpeg|pdf|tex|dat|parquet|h5|hdf5)"
    r"\b(?![\w.])",
    re.I,
)

#: P5.5: "test" as its own filename token — "latest.csv"/"contest.csv"
#: are NOT test inputs; "test_x.csv", "x_test.csv", "my-test-runs.tsv" are.
_TEST_FILE_RE = re.compile(r"(?:^|[-_.])test(?:[-_.]|$)", re.I)
_KIND_BY_SUFFIX = {
    ".csv": "csv",
    ".tsv": "csv",
    ".npy": "npy",
    ".npz": "npz",
    ".png": "png",
    ".jpg": "png",
    ".jpeg": "png",
    ".json": "json",
    ".txt": "txt",
}


def _finite(x: Any) -> float | None:
    """Float value when finite, else None (nan/inf stay visible as None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _col_stats(s: pd.Series) -> dict[str, Any]:
    """Numeric summary of one DataFrame column."""
    num = pd.to_numeric(s, errors="coerce")
    ok = num.notna()
    st: dict[str, Any] = {"n_nan": int((~ok).sum()), "n_unique": int(s.nunique(dropna=True))}
    if ok.sum():
        v = num[ok]
        st.update(
            {
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6),
                "mean": round(float(v.mean()), 6),
                "std": round(float(v.std()), 6),
            }
        )
    return st


def _read_csv_safe(p: Path) -> pd.DataFrame | None:
    """CSV/TSV read that degrades to None instead of raising."""
    try:
        return pd.read_csv(p, nrows=200_000)
    except Exception:  # noqa: BLE001 — unreadable tables are evidence too
        try:
            return pd.read_csv(p, sep="\t", nrows=200_000)
        except Exception:  # noqa: BLE001
            return None


def id_order_match(pred: pd.DataFrame, ws: Path) -> dict[str, Any] | None:
    """Do a pred table's leading text columns match a column of an
    in-workspace *test* input file, in order?

    Reproduces the grader's data-correctness check without gold (same
    workspace, both sides).

    P5.5 honesty rules:

    - the comparison requires EQUAL length — a prefix-vs-full-table
      match of different-length tables is not an order match;
    - "test" must be its own filename token (``test_x.csv``,
      ``x_test.csv``) — ``latest.csv`` / ``contest.csv`` are not test
      inputs;
    - ``None`` when the check is NOT evaluable (no leading text column,
      or no test input table in the workspace) — "unknown" is not
      "mismatch".
    """
    text_cols = [
        c
        for c in pred.columns[:ID_COLS_TRIED]
        if not pd.api.types.is_numeric_dtype(pred[c])
    ]
    if not text_cols:
        return None
    saw_test_table = False
    for c in text_cols[:1] + text_cols[1:]:  # ordered, first preference wins
        vals = pred[c].astype(str).tolist()
        for tf in sorted(ws.rglob("*.csv")) + sorted(ws.rglob("*.tsv")):
            if "pred_results" in tf.parts or tf.stat().st_size > 20 << 20:
                continue
            if not _TEST_FILE_RE.search(tf.stem):
                continue
            tdf = _read_csv_safe(tf)
            if tdf is None or tdf.empty:
                continue
            saw_test_table = True
            for tc in tdf.columns:
                tvals = tdf[tc].astype(str).tolist()
                if len(tvals) == len(vals) and tvals == vals:
                    return {
                        "test_file": tf.relative_to(ws).as_posix(),
                        "test_col": str(tc),
                        "pred_col": str(c),
                        "match": True,
                    }
    return {"match": False} if saw_test_table else None


def parse_table(p: Path, ws: Path) -> dict[str, Any] | None:
    """One parsed prediction table record."""
    df = _read_csv_safe(p)
    if df is None or df.empty:
        return {
            "path": p.relative_to(ws).as_posix(),
            "n_rows": 0 if df is None else len(df),
            "columns": list(df.columns) if df is not None else [],
            "error": "unreadable",
        }
    t: dict[str, Any] = {
        "path": p.relative_to(ws).as_posix(),
        "n_rows": int(len(df)),
        "columns": list(df.columns)[:24],
    }
    num_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ][:MAX_NUM_COLS]
    t["num_stats"] = {c: _col_stats(df[c]) for c in num_cols}
    first = df.columns[0]
    t["first_col_head"] = [str(v)[:60] for v in df[first].head(3).tolist()]
    if "pred_results" in p.parts:
        t["id_order_match"] = id_order_match(df, ws)
    return t


def parse_array(p: Path, ws: Path) -> dict[str, Any]:
    """One parsed .npy array record."""
    try:
        a = np.load(p, allow_pickle=False)
        flat = np.asarray(a, dtype=float).ravel()
        finite = flat[np.isfinite(flat)]
        st: dict[str, Any] = {"n_unique": int(len(np.unique(finite)))}
        if finite.size:
            st.update(
                {
                    "min": round(float(finite.min()), 6),
                    "max": round(float(finite.max()), 6),
                    "mean": round(float(finite.mean()), 6),
                    "std": round(float(finite.std()), 6),
                }
            )
        return {
            "path": p.relative_to(ws).as_posix(),
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "stats": st,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "path": p.relative_to(ws).as_posix(),
            "error": f"{type(e).__name__}",
        }


def parse_figure(p: Path, ws: Path) -> dict[str, Any]:
    """One parsed figure record (dimensions + mode, no pixels)."""
    try:
        with Image.open(p) as im:
            return {
                "path": p.relative_to(ws).as_posix(),
                "width_px": im.width,
                "height_px": im.height,
                "mode": im.mode,
            }
    except Exception as e:  # noqa: BLE001
        return {
            "path": p.relative_to(ws).as_posix(),
            "error": f"{type(e).__name__}",
        }


def metric_lines(txt: str, source: str) -> list[dict[str, Any]]:
    """Deprecated: LLM extraction replaces regex scraping (2026-10-10)."""
    return []


def _list_shape(obj: list) -> list[int]:
    """Outer extents of a (possibly nested) list (first-element recursion)."""
    shape = [len(obj)]
    for v in obj:
        if isinstance(v, list):
            shape += _list_shape(v)
            break
    return shape


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Numeric leaves of a JSON tree under dotted keys.

    Booleans are NEVER metrics (they land in :func:`flatten_flags`) — a
    ``globally_optimal: true`` flattened to ``1.0`` produced spurious
    decisive verdicts (HP gen 4, 2026-10-07). Nested arrays (coordinates,
    conformations, per-sample lists) are recorded as ``{shape, len}``
    markers instead of per-element scalars, so coordinate leaves whose
    key names merely contain words like "optimal" can never masquerade
    as metric evidence; 1-D numeric lists surface as per-element scalars
    only under metric-like keys.
    """
    out: dict[str, Any] = {}
    if isinstance(obj, bool):
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_json(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if any(isinstance(v, list | dict) for v in obj):
            out[prefix] = {"shape": _list_shape(obj), "len": len(obj)}
        elif (
            obj
            and len(obj) <= 10
            and prefix
            and _LIST_METRIC_KEY.search(prefix)
            and all(isinstance(v, int | float) and not isinstance(v, bool) for v in obj)
        ):
            for i, v in enumerate(obj):
                f = _finite(v)
                if f is not None:
                    out[f"{prefix}[{i}]"] = f
        # non-metric 1-D lists, bool/str lists, empty lists: no leaves
    else:
        f = _finite(obj)
        if f is not None:
            out[prefix] = f
    return out


def flatten_flags(obj: Any, prefix: str = "") -> dict[str, bool]:
    """Boolean leaves of a JSON tree under dotted keys (never metrics).

    Presence/value evidence only: flags are comparable as facts
    (``json_flags``) but never enter the numeric metric leaves, so an
    LLM policy cannot arithmetic-compare a boolean against a coordinate.
    """
    out: dict[str, bool] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_flags(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:10]):
            out.update(flatten_flags(v, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        out[prefix] = obj
    return out


def _top_level_scalars(obj: Any) -> dict[str, float]:
    """Finite top-level scalar values of one JSON object (P5.6).

    Oversized files never enter the full flatten, but their headline
    numbers (``auc``, ``n_rows`` ...) stay visible as evidence.
    """
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, bool | dict | list):
                continue
            f = _finite(v)
            if f is not None:
                out[str(k)] = f
    return out


def _cap_metric_first(flat: dict[str, Any]) -> dict[str, Any]:
    """Cap at :data:`MAX_JSON_KEYS`, metric-like keys first (P5.6).

    The old alphabetical cap could drop every metric leaf of a large
    results file while keeping nested config noise. Metric-like keys win
    the budget; ties stay alphabetical for determinism.
    """
    if len(flat) <= MAX_JSON_KEYS:
        return dict(sorted(flat.items()))
    metric_like = sorted(k for k in flat if _LIST_METRIC_KEY.search(k))
    rest = sorted(k for k in flat if not _LIST_METRIC_KEY.search(k))
    keep = (metric_like + rest)[:MAX_JSON_KEYS]
    return {k: flat[k] for k in keep}


def contract_files(goal: str, ws: Path) -> list[dict[str, Any]]:
    """Files the GOAL names verbatim: presence + bytes + sha + head.

    The goal text is the deliverable contract (P1.4): every filename it
    mentions — ``run.sh``, ``requirement.txt``, any suffix — gets a
    record judged from the workspace SNAPSHOT, independent of freshness
    (agent-written static files like ``run.sh`` are legitimately not
    "fresh"). This is the ONLY digest surface where infra-named files
    appear: a file named in the goal is never infra here, however it is
    classified elsewhere.

    N4 token hygiene: URLs (``//``) are skipped, ``a.json/.txt``-style
    double extensions split to the real file, and extension-embedded
    names (``data.jsonl``) never match a shorter suffix — the policy
    treats ``present:false`` as decisive, so a bogus token is a bogus
    contract violation.

    Args:
        goal: Verbatim task goal text.
        ws: Workspace directory to judge presence from.

    Returns:
        One record per distinct goal-named file (capped at 16).
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in _GOAL_FILE_RE.findall(goal or ""):
        if "//" in name:
            continue  # URL, not a workspace file
        # N4: "a.json/.txt" names TWO extensions of one stem — keep the
        # file ("a.json"), never the merged non-path "a.json/.txt".
        name = name.split("/.", 1)[0]
        rel = name.lstrip("./")
        if not rel or rel.lower() in seen:
            continue
        seen.add(rel.lower())
        p = Path(ws) / rel
        rec: dict[str, Any] = {"path": rel, "present": False}
        try:
            if p.is_file():
                rec["present"] = True
                rec["bytes"] = p.stat().st_size
                rec["sha256"] = hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()[:16]
                rec["head"] = p.read_text(errors="replace")[:400]
        except OSError:
            pass  # presence stays False; unreadable files are not evidence
        out.append(rec)
        if len(out) >= 16:
            break
    return out


def _deliverable_records(ws: Path, fresh: set[str]) -> tuple[list, list, list, list]:
    """Scan deliverable files; returns (deliverables, tables, arrays, figures)."""
    deliverables: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    arrays: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    for p in sorted(ws.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ws).as_posix()
        if any(part.startswith(".") for part in Path(rel).parts):
            continue
        if is_infra_name(rel):  # P1.5: one infra rule (recon.is_infra_name)
            continue
        suffix = p.suffix.lower()
        in_pred = "pred_results" in p.parts
        top = len(Path(rel).parts) == 1
        if not (in_pred or (top and suffix in {".json", ".txt", ".csv"})):
            continue
        if suffix in {".py", ".log", ".sh"} and not in_pred:
            continue
        kind = _KIND_BY_SUFFIX.get(suffix, "other")
        deliverables.append(
            {
                "path": rel,
                "bytes": p.stat().st_size,
                "kind": kind,
                "produced_this_run": rel in fresh,
            }
        )
        if in_pred and suffix in {".csv", ".tsv"}:
            t = parse_table(p, ws)
            if t:
                tables.append(t)
        elif in_pred and suffix == ".npy":
            arrays.append(parse_array(p, ws))
        elif in_pred and suffix in {".png", ".jpg", ".jpeg"}:
            figures.append(parse_figure(p, ws))
    return deliverables, tables, arrays, figures


def build_digest(
    uuid: str,
    ws: Path,
    ledger: dict[str, Any] | None,
    goal: str = "",
) -> dict[str, Any]:
    """Build the deterministic digest for one generation.

    Args:
        uuid: Generation identifier (identifier only — never evidence).
        ws: Workspace directory (sandbox copy / snapshot / live workspace).
        ledger: That generation's re-execution ledger; ``None`` for a
            bootstrap digest rebuilt from a snapshot without fresh
            execution facts (the exec block then records the bootstrap).
        goal: Verbatim task goal — when given, ``contract_files`` records
            every goal-named file (P1.4) so the deliverable contract
            (``run.sh``, ``requirement.txt`` ...) is finally visible.

    Returns:
        The digest dict (schema documented in the policy writer prompt).
    """
    ledger = ledger or {}
    fresh = set(ledger.get("fresh_artifacts") or {})
    ws = Path(ws)
    d: dict[str, Any] = {"uuid": uuid}
    d["exec"] = {
        "exit_code": ledger.get("exit_code"),
        "wall_time_s": ledger.get("wall_time_s"),
        "timed_out": ledger.get("timed_out"),
        "error_signature": ledger.get("error_signature") or [],
        "produced_pred_results": ledger.get(
            "produced_pred_results",
            any(k.startswith("pred_results/") for k in fresh),
        ),
        "bootstrap": ledger.get("bootstrap"),
        "log_tail": [
            ln.replace(str(ws), "WS")
            for ln in (ledger.get("log_tail") or "").splitlines()[-20:]
        ],
    }

    d["contract_files"] = contract_files(goal, ws) if goal else []

    deliverables, tables, arrays, figures = [], [], [], []
    if ws.exists():
        deliverables, tables, arrays, figures = _deliverable_records(ws, fresh)
    d["deliverables"] = deliverables
    d["tables"] = tables
    d["arrays"] = arrays
    d["figures"] = figures

    logs: list[dict[str, Any]] = []
    if ws.exists():
        # ALL .txt files, not just those with "log" in the name —
        # results.txt and run_output.txt carry the actual metrics
        # (HP-folding review 2026-10-10)
        log_files = [ws / "output.log"] + sorted(ws.glob("*.txt"))
        for lf in log_files[:8]:
            if lf.exists():
                logs.extend(
                    metric_lines(
                        lf.read_text(errors="replace")[:400_000], lf.name
                    )
                )
    d["log_metrics"] = logs[:MAX_METRIC_LINES]

    jm: dict[str, dict[str, Any]] = {}
    jf: dict[str, dict[str, bool]] = {}
    if ws.exists():
        # P5.6: filter BEFORE the 16-file cap (an infra or unparseable
        # file used to waste a slot), key by RELPATH (top-level and
        # pred_results files can share a name), and keep top-level
        # scalars of oversized files instead of dropping them entirely.
        jcands: list[tuple[Path, int]] = []
        for jp in sorted(ws.glob("*.json")) + sorted(
            ws.glob("pred_results/*.json")
        ):
            if is_infra_name(jp.name):
                continue
            try:
                size = jp.stat().st_size
            except OSError:
                continue
            if size > MAX_JSON_SCAN:
                continue
            jcands.append((jp, size))
        for jp, size in jcands[:16]:
            try:
                parsed = json.loads(jp.read_text(errors="replace"))
            except Exception:  # noqa: BLE001 — unparseable JSON is skipped
                continue
            rel = jp.relative_to(ws).as_posix()
            flat = (
                flatten_json(parsed)
                if size <= JSON_FILE_CAP
                else _top_level_scalars(parsed)
            )
            if flat:
                jm[rel] = _cap_metric_first(flat)
            flags = flatten_flags(parsed)
            if flags:
                jf[rel] = dict(sorted(flags.items())[:MAX_JSON_KEYS])
    d["json_metrics"] = jm
    d["json_flags"] = jf

    inputs: dict[str, Any] = {"train_csvs": {}, "test_csvs": {}}
    if ws.exists():
        for p in sorted(ws.rglob("*train*.csv")) + sorted(ws.rglob("*train*.tsv")):
            if "pred_results" in p.parts or "train_used" in p.name:
                continue
            df = _read_csv_safe(p)
            if df is None or df.empty:
                continue
            cols = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ][:MAX_TRAIN_COLS]
            inputs["train_csvs"][p.relative_to(ws).as_posix()] = {
                "n_rows": int(len(df)),
                **{c: _col_stats(df[c]) for c in cols},
            }
            if len(inputs["train_csvs"]) >= 3:
                break
        for p in sorted(ws.rglob("*test*.csv")):
            if "pred_results" in p.parts:
                continue
            df = _read_csv_safe(p)
            if df is None or df.empty:
                continue
            inputs["test_csvs"][p.relative_to(ws).as_posix()] = {
                "n_rows": int(len(df)),
                "columns": list(df.columns)[:12],
            }
            if len(inputs["test_csvs"]) >= 3:
                break
    d["inputs"] = inputs

    tu: dict[str, Any] | None = None
    if ws.exists():
        for p in sorted(ws.rglob("*train_used*")):
            if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}:
                df = _read_csv_safe(p)
                tu = {
                    "path": p.relative_to(ws).as_posix(),
                    "n_rows": int(len(df)) if df is not None else None,
                }
                break
    d["train_used"] = tu
    return d


def digest_summary(digest: dict[str, Any]) -> dict[str, Any]:
    """Small stage-line summary of one digest (plan §6b DIGEST line)."""
    tables = digest.get("tables") or []
    idm = None
    for t in tables:
        m = (t.get("id_order_match") or {}).get("match")
        if m is not None:
            idm = "match" if m else "mismatch"
            break
    return {
        "tables": len(tables),
        "arrays": len(digest.get("arrays") or []),
        "figures": len(digest.get("figures") or []),
        "metrics": len(digest.get("log_metrics") or []),
        "deliverables": len(digest.get("deliverables") or []),
        "id_order": idm or "n/a",
    }


def is_figure_dominant(digest: dict[str, Any], share: float = 0.5) -> bool:
    """E26 channel-selection rule on one digest: the task's deliverables
    are figure-dominant when >= *share* of the deliverable pool (narrowed
    to results directories when any exist) are images."""
    deliverables = digest.get("deliverables") or []
    if not deliverables:
        return False
    in_results = [
        dv
        for dv in deliverables
        if "pred_results" in dv["path"]
        or "results" in Path(dv["path"]).parent.as_posix().lower()
    ]
    pool = in_results or deliverables
    images = sum(1 for dv in pool if dv.get("kind") == "png")
    return images / len(pool) >= share
