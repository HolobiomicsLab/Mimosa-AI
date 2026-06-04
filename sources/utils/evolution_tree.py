"""
Render the workflow evolution tree as a PNG.

Reads ``lineage_<uuid>.json`` files written by :mod:`sources.core.lineage`
(see also :class:`sources.core.workflow_info.WorkflowInfo` for the per-run
score). Lays out workflows by depth-from-seed and renders nodes coloured by
:attr:`WorkflowInfo.overall_score`, with edges styled to distinguish
mutation from crossover. Uses only matplotlib so it does not introduce a
networkx dependency.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from sources.core.lineage import scan_all
from sources.core.workflow_info import WorkflowInfo

logger = logging.getLogger(__name__)


# Red → amber → green colormap for score ∈ [0, 1].
_SCORE_CMAP = LinearSegmentedColormap.from_list(
    "score_rag",
    [
        (0.0, "#d9534f"),  # red
        (0.5, "#f0ad4e"),  # amber
        (1.0, "#5cb85c"),  # green
    ],
)


def _build_depths(
    records: dict[str, dict],
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Compute depth-from-seed for every node and the children adjacency.

    Depth is the longest path from a root (a node with no resolvable
    parents) so siblings produced in different generations don't collide.
    Cycles (which shouldn't happen but might if a lineage file is
    hand-edited) are broken with a visited set.
    """
    children: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {u: 0 for u in records}

    for uuid, rec in records.items():
        parents = [p for p in rec.get("parents", []) if p in records]
        for p in parents:
            children[p].append(uuid)
            in_degree[uuid] = in_degree.get(uuid, 0) + 1

    # Iterative DAG depth pass; tolerates dangling parent UUIDs by treating
    # the node as a root.
    depths: dict[str, int] = {}
    queue: deque[str] = deque()
    for uuid in records:
        if in_degree.get(uuid, 0) == 0:
            depths[uuid] = 0
            queue.append(uuid)

    # Process in topological order, taking max depth from any parent.
    seen = set()
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        for child in children.get(node, []):
            cand = depths[node] + 1
            if cand > depths.get(child, -1):
                depths[child] = cand
            queue.append(child)

    # Anything left (cycles or orphans whose parents reference nonexistent
    # nodes outside `records`) gets depth 0 so it still appears.
    for uuid in records:
        depths.setdefault(uuid, 0)

    return depths, children


def _layout(
    records: dict[str, dict],
    depths: dict[str, int],
) -> dict[str, tuple[float, float]]:
    """Assign (x, y) positions: y = depth, x spreads siblings at each level."""
    by_depth: dict[int, list[str]] = defaultdict(list)
    for uuid, d in depths.items():
        by_depth[d].append(uuid)
    # Sort each layer by created_at (then uuid) for stable, chronological x order.
    for d in by_depth:
        by_depth[d].sort(
            key=lambda u: (records[u].get("created_at", ""), u)
        )

    pos: dict[str, tuple[float, float]] = {}
    for d, layer in by_depth.items():
        n = max(len(layer), 1)
        for i, uuid in enumerate(layer):
            # x ∈ [0, 1] centred per layer; y inverted so seeds sit at the top.
            x = (i + 1) / (n + 1)
            y = -d
            pos[uuid] = (x, y)
    return pos


def _node_score(workflow_dir: Path, uuid: str) -> float | None:
    """Best-effort score lookup for a UUID; None when no state_result exists."""
    try:
        wf = WorkflowInfo(uuid, workflow_dir / uuid)
        if not (workflow_dir / uuid / "state_result.json").exists():
            return None
        return float(wf.overall_score)
    except Exception as e:
        logger.debug(f"evolution_tree: score lookup failed for {uuid}: {e}")
        return None


def render_evolution_tree(
    workflow_dir: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path | None:
    """Scan ``workflow_dir`` and render the evolution tree to PNG.

    Args:
        workflow_dir: Directory that holds ``<uuid>/`` workflow folders.
        output_path: Where to write the PNG. Defaults to
            ``<workflow_dir>/evolution_tree.png``.
        title: Optional plot title; defaults to a generic header.

    Returns:
        Path to the written PNG, or None if no workflows were found.
    """
    root = Path(workflow_dir)
    records = scan_all(root)
    if not records:
        logger.info(f"evolution_tree: no workflows under {root}")
        return None

    depths, children = _build_depths(records)
    pos = _layout(records, depths)
    scores = {u: _node_score(root, u) for u in records}

    n_depth = max(depths.values()) + 1
    max_width = max(
        (sum(1 for u in records if depths[u] == d) for d in range(n_depth)),
        default=1,
    )
    fig_w = max(8.0, 1.4 * max_width)
    fig_h = max(5.0, 1.6 * n_depth)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Edges ──────────────────────────────────────────────────────────────
    for uuid, rec in records.items():
        kind = rec.get("evolution_kind", "seed")
        for parent in rec.get("parents", []):
            if parent not in pos:
                continue
            x0, y0 = pos[parent]
            x1, y1 = pos[uuid]
            style = "--" if kind == "crossover" else "-"
            colour = "#7a4fbf" if kind == "crossover" else "#7f8c8d"
            ax.annotate(
                "",
                xy=(x1, y1 + 0.08),
                xytext=(x0, y0 - 0.08),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colour,
                    linestyle=style,
                    lw=1.2,
                    alpha=0.85,
                ),
            )

    # ── Nodes ──────────────────────────────────────────────────────────────
    xs, ys, colours, sizes = [], [], [], []
    for uuid, (x, y) in pos.items():
        xs.append(x)
        ys.append(y)
        s = scores.get(uuid)
        if s is None:
            colours.append("#bdc3c7")  # no score yet / failed
            sizes.append(400)
        else:
            colours.append(_SCORE_CMAP(max(0.0, min(1.0, s))))
            sizes.append(400 + 600 * s)

    ax.scatter(xs, ys, c=colours, s=sizes, edgecolors="black", linewidths=1.0, zorder=3)

    # Labels: short UUID + score
    for uuid, (x, y) in pos.items():
        s = scores.get(uuid)
        short = uuid[-8:]
        score_txt = f"\n{s:.2f}" if s is not None else "\n—"
        ax.text(
            x,
            y - 0.22,
            short + score_txt,
            ha="center",
            va="top",
            fontsize=7,
            family="monospace",
        )

    # ── Axes / styling ─────────────────────────────────────────────────────
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-n_depth - 0.5, 0.6)
    ax.set_xticks([])
    ax.set_yticks([-d for d in range(n_depth)])
    ax.set_yticklabels([f"depth {d}" for d in range(n_depth)], fontsize=8)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_title(title or "Workflow evolution tree", fontsize=12, pad=12)

    legend_handles = [
        Line2D([0], [0], color="#7f8c8d", lw=1.2, label="mutation"),
        Line2D([0], [0], color="#7a4fbf", lw=1.2, ls="--", label="crossover"),
        mpatches.Patch(facecolor=_SCORE_CMAP(1.0), edgecolor="black", label="score 1.0"),
        mpatches.Patch(facecolor=_SCORE_CMAP(0.5), edgecolor="black", label="score 0.5"),
        mpatches.Patch(facecolor=_SCORE_CMAP(0.0), edgecolor="black", label="score 0.0"),
        mpatches.Patch(facecolor="#bdc3c7", edgecolor="black", label="no score"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    output = Path(output_path) if output_path else root / "evolution_tree.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"evolution_tree: wrote {output} ({len(records)} nodes)")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render the workflow evolution tree.")
    parser.add_argument(
        "workflow_dir",
        nargs="?",
        default="sources/workflows",
        help="Directory containing <uuid>/ workflow folders.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output PNG path (default: <workflow_dir>/evolution_tree.png).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = render_evolution_tree(args.workflow_dir, args.output)
    if path is None:
        print("No workflows found.")
