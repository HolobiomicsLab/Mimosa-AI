"""
Render a 2-D PCA projection of the QD archive's genotype embeddings.

Each archive member is one point; distance on the map approximates the
cosine geometry the novelty score is computed in, so a newcomer landing
far from the cloud illustrates *why* it was admitted. The projection is
refit from the current archive on every call: axes may rotate as members
arrive or get evicted, so each PNG is a self-contained snapshot, not a
frame in a fixed coordinate system. Axis labels carry the explained
variance so the reader can judge how faithful the 2-D picture is.

Kept import-light on purpose (matplotlib + numpy only) so callers and
tests never drag in the embedding backend; the ``__main__`` CLI lazily
imports it to rebuild descriptors from on-disk genotypes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

# Minimum members for a meaningful plane: with 2 points PC2 is pure noise.
MIN_MEMBERS = 3

# Red → amber → green colormap for reward ∈ [0, 1] (matches evolution_tree).
_REWARD_CMAP = LinearSegmentedColormap.from_list(
    "reward_rag",
    [
        (0.0, "#d9534f"),  # red
        (0.5, "#f0ad4e"),  # amber
        (1.0, "#5cb85c"),  # green
    ],
)


def _pca_2d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project row vectors onto their top-2 principal components.

    Args:
        vectors: Array of shape ``(n, d)`` with ``n ≥ 2`` and ``d ≥ 2``.

    Returns:
        ``(coords, explained)`` — coords of shape ``(n, 2)`` and the
        fraction of total variance captured by each axis, shape ``(2,)``.
        A zero-variance input yields all-zero coords and explained.
    """
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].copy()
    # Deterministic sign: the largest-|loading| coordinate of each axis is
    # positive, so small archive changes don't mirror-flip the map.
    for i, comp in enumerate(components):
        if comp[np.argmax(np.abs(comp))] < 0:
            components[i] = -comp
    coords = centered @ components.T
    total_variance = float((singular**2).sum())
    if total_variance <= 0.0:
        return np.zeros((len(vectors), 2)), np.zeros(2)
    return coords, (singular[:2] ** 2) / total_variance


def _descriptor_matrix(members: list[Any]) -> tuple[np.ndarray | None, list[Any]]:
    """Stack members' descriptors into equal-dimension rows.

    Members without a descriptor are dropped; when dimensions disagree
    (e.g. the embedding backend changed mid-run) only the majority
    dimension is kept so the matrix stays rectangular.

    Args:
        members: Objects exposing ``behaviour_descriptor: list[float]``.

    Returns:
        ``(matrix, kept)`` — the ``(n, d)`` float array and the members
        contributing its rows, or ``(None, [])`` when nothing is usable.
    """
    valid = [m for m in members if getattr(m, "behaviour_descriptor", None)]
    if not valid:
        return None, []
    dims = [len(m.behaviour_descriptor) for m in valid]
    majority_dim = max(set(dims), key=dims.count)
    kept = [m for m in valid if len(m.behaviour_descriptor) == majority_dim]
    if len(kept) < len(valid):
        logger.warning(
            f"archive_pca: dropped {len(valid) - len(kept)} member(s) "
            f"with descriptor dim ≠ {majority_dim}"
        )
    matrix = np.asarray([m.behaviour_descriptor for m in kept], dtype=np.float64)
    return matrix, kept


def _draw_members(
    ax: plt.Axes,
    coords: np.ndarray,
    members: list[Any],
    highlight_uuid: str | None,
) -> bool:
    """Scatter archive members, starring the highlighted (newest) one.

    Returns:
        ``True`` when the highlight marker was drawn.
    """
    rewards = [max(0.0, min(1.0, float(getattr(m, "reward", 0.0) or 0.0))) for m in members]
    colours = [_REWARD_CMAP(r) for r in rewards]
    ax.scatter(coords[:, 0], coords[:, 1], c=colours, s=160,
               edgecolors="black", linewidths=0.8, alpha=0.9, zorder=3)
    highlighted = False
    for (x, y), member in zip(coords, members):
        uuid = getattr(member, "uuid", None) or ""
        if highlight_uuid and uuid == highlight_uuid:
            ax.scatter([x], [y], marker="*", s=550, facecolors="none",
                       edgecolors="#1c5fbf", linewidths=1.6, zorder=4)
            highlighted = True
        ax.annotate(uuid[-8:] or "?", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7, family="monospace")
    return highlighted


def render_archive_pca(
    members: list[Any],
    workflow_dir: str | Path,
    output_path: str | Path | None = None,
    highlight_uuid: str | None = None,
    title: str | None = None,
) -> Path | None:
    """Render the archive's genotype embeddings as a 2-D PCA scatter PNG.

    Args:
        members: Archive members (e.g. ``PopulationMember``) exposing
            ``behaviour_descriptor``, ``uuid`` and ``reward``.
        workflow_dir: Workflows root; the default output lands at its top
            level so one image summarises the whole archive.
        output_path: Override for the PNG location. Defaults to
            ``<workflow_dir>/archive_pca.png``.
        highlight_uuid: UUID starred as the newest addition; silently
            ignored when it matches no plotted member.
        title: Optional plot title.

    Returns:
        Path to the written PNG, or ``None`` when fewer than
        ``MIN_MEMBERS`` usable descriptors exist.
    """
    matrix, kept = _descriptor_matrix(members)
    if matrix is None or len(kept) < MIN_MEMBERS or matrix.shape[1] < 2:
        logger.info(
            f"archive_pca: need ≥{MIN_MEMBERS} embedded members, "
            f"have {0 if matrix is None else len(kept)} — skipping render"
        )
        return None

    coords, explained = _pca_2d(matrix)
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    highlighted = _draw_members(ax, coords, kept, highlight_uuid)

    ax.set_xlabel(f"PC1 ({explained[0]:.0%} of variance)", fontsize=9)
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} of variance)", fontsize=9)
    ax.set_title(title or f"QD archive — genotype embedding PCA ({len(kept)} members)",
                 fontsize=12, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=_REWARD_CMAP, norm=plt.Normalize(0.0, 1.0)),
        ax=ax, fraction=0.04, pad=0.02,
    ).set_label("reward", fontsize=9)
    if highlighted:
        ax.scatter([], [], marker="*", s=180, facecolors="none",
                   edgecolors="#1c5fbf", label="newest member")
        ax.legend(loc="best", fontsize=8, frameon=False)

    fig.tight_layout()
    output = Path(output_path) if output_path else Path(workflow_dir) / "archive_pca.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"archive_pca: wrote {output} ({len(kept)} members)")
    return output


def _members_from_disk(workflow_dir: Path) -> list[Any]:
    """Rebuild plottable members by embedding each on-disk genotype.

    Best effort for the CLI: workflows whose code cannot be embedded are
    skipped; a missing score renders as reward 0.

    Args:
        workflow_dir: Directory holding ``<uuid>/workflow_genotype_<uuid>.py``.

    Returns:
        List of lightweight member objects for :func:`render_archive_pca`.
    """
    from types import SimpleNamespace

    from sources.core.genotype_embedding import embed_genotype
    from sources.core.workflow_info import WorkflowInfo

    members: list[Any] = []
    for genotype_file in sorted(workflow_dir.glob("*/workflow_genotype_*.py")):
        uuid = genotype_file.parent.name
        vec = embed_genotype(genotype_file.read_text(encoding="utf-8", errors="replace"))
        if vec is None:
            continue
        try:
            reward = float(WorkflowInfo(uuid, genotype_file.parent).overall_score)
        except Exception as exc:
            logger.debug(f"archive_pca: score lookup failed for {uuid}: {exc}")
            reward = 0.0
        members.append(SimpleNamespace(
            uuid=uuid, behaviour_descriptor=[float(x) for x in vec], reward=reward,
        ))
    return members


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render the QD archive PCA projection.")
    parser.add_argument(
        "workflow_dir",
        nargs="?",
        default="sources/workflows",
        help="Directory containing <uuid>/ workflow folders.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output PNG path (default: <workflow_dir>/archive_pca.png).",
    )
    parser.add_argument(
        "--highlight", default=None,
        help="UUID to star as the newest member.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    disk_members = _members_from_disk(Path(args.workflow_dir))
    path = render_archive_pca(
        disk_members, args.workflow_dir,
        output_path=args.output, highlight_uuid=args.highlight,
    )
    if path is None:
        print(f"Not enough embeddable workflows under {args.workflow_dir} "
              f"(need ≥{MIN_MEMBERS}).")
