#!/usr/bin/env python3
"""Workflow Evolution Animation — interactive Jarvis-style viewer.

Walks the evolutionary tree of workflows under ``sources/workflows``, plays the
memory timelapse of each generation on the right, highlights the active node
in the lineage tree on the left, shows the workflow PNG underneath, and tracks
rubric pass/fail evolution across iterations along the bottom.

Usage
-----
    uv run workflow_evolution_anim.py
    uv run workflow_evolution_anim.py --workflows-dir sources/workflows \\
                                      --memory-dir sources/memory

Controls
--------
    SPACE   play / pause
    LEFT  RIGHT  prev / next memory step
    UP    DOWN   jump to previous / next workflow (generation)
    B       jump to the best-scoring generation
    [ / ]   slow down / speed up
    H       toggle help overlay
    Click on tree node → jump to that workflow
    Click on bottom timeline → scrub through every (workflow, step) frame
    Esc / Q quit
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
from PIL import Image

# Reuse the trace-parsing helpers from the existing timelapse tool.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_timelapse import StepInfo, load_memory_files, parse_step  # noqa: E402

# ---------------------------------------------------------------------------
# Visual constants — SF / Jarvis vibe: deep navy background, cyan/amber glow.
# ---------------------------------------------------------------------------
BG = (8, 12, 20)
BG_PANEL = (16, 22, 32)
BG_PANEL_LIGHT = (22, 30, 44)
BORDER = (40, 60, 90)
BORDER_BRIGHT = (90, 140, 200)

ACCENT = (0, 210, 255)         # cyan glow (primary highlight)
ACCENT_DIM = (0, 110, 150)
AMBER = (255, 170, 50)
SUCCESS = (60, 220, 130)
ERROR = (255, 80, 100)
WARN = (240, 200, 80)
INFO = (110, 170, 255)
VIOLET = (180, 120, 255)

TEXT = (220, 230, 240)
TEXT_DIM = (140, 165, 195)
TEXT_FAINT = (90, 110, 140)

STAGE_COLORS = {
    "plan_creator": VIOLET,
    "stage2": (236, 100, 180),
    "stage3": (60, 180, 255),
    "stage4": AMBER,
    "workflow_creator": VIOLET,
    "task_builder": INFO,
    "task_grounded_validator": SUCCESS,
    "verifier_abstract_textual_gradient": AMBER,
}

EVO_COLORS = {
    "seed": (130, 130, 160),
    "mutation": INFO,
    "crossover": VIOLET,
}

# Ambient animation tuning. Every effect is a pure function of an
# accumulated animation clock, so GUI and --record output stay identical.
STAR_COUNT = 140          # background starfield particles
STAR_SEED = 7             # fixed seed keeps record mode deterministic
SCAN_PERIOD_S = 9.0       # seconds per background scanline sweep
PULSE_PERIOD_S = 2.2      # breathing period of the active tree node
FLOW_PERIOD_S = 1.6       # lineage flow-dot travel time per edge
TYPE_REVEAL_BOOST = 1.35  # typewriter finishes before the frame ends
RUBRIC_FLASH_FRAMES = 8   # frames a freshly-completed column stays flashed
BURST_PARTICLES = 10      # radial sparks on transition arrival


@dataclass
class Transition:
    """One-shot animation played when the active workflow changes."""
    from_uuid: str       # source node — genetic parent if known, else prev wf
    to_uuid: str         # destination — the newly active workflow
    from_wi: int         # previous workflow index (for the PNG cross-fade)
    elapsed: float = 0.0
    duration: float = 0.55

    @property
    def t(self) -> float:
        return min(1.0, self.elapsed / self.duration)

    @property
    def ease(self) -> float:
        t = self.t
        return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class ClaimResult:
    name: str
    importance: int
    status: str   # pass / fail / error / unsure
    score: float


@dataclass
class Evaluation:
    overall: float
    n_pass: int
    n_fail: int
    n_error: int
    n_unsure: int
    claims: list[ClaimResult]


@dataclass
class Workflow:
    uuid: str
    iteration: int
    kind: str                # seed / mutation / crossover
    parents: list[str]
    steps: list[StepInfo] = field(default_factory=list)
    evaluation: Evaluation | None = None
    png_path: Path | None = None


def _parse_evaluation(path: Path) -> Evaluation | None:
    """Extract overall score and per-claim pass/fail from an evaluation.txt."""
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    overall = 0.0
    n_pass = n_fail = n_error = n_unsure = 0

    m = re.search(r"Overall:\s*([0-9.]+)", text)
    if m:
        overall = float(m.group(1))
    m = re.search(
        r"Claims:\s*\d+\s+pass=(\d+)\s+fail=(\d+)\s+error=(\d+)\s+unsure=(\d+)",
        text,
    )
    if m:
        n_pass, n_fail, n_error, n_unsure = (int(x) for x in m.groups())

    # Claim blocks always start at column 0 with [name] (importance=…).
    # Anchor there to avoid picking up ANSI escape sequences embedded in
    # stderr traces such as ``[35m"…"[0m``.
    claims: list[ClaimResult] = []
    pattern = re.compile(
        r"^\[(?P<name>[A-Za-z0-9_\- ]+)\]\s*\(importance=(?P<imp>\d+);.*?\n"
        r"(?:.*?\n)*?"
        r"\s*kind=\w+\s+status=(?P<status>\w+)\s+score=(?P<score>[0-9.]+)",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        claims.append(
            ClaimResult(
                name=m.group("name").strip(),
                importance=int(m.group("imp")),
                status=m.group("status"),
                score=float(m.group("score")),
            )
        )
    return Evaluation(overall, n_pass, n_fail, n_error, n_unsure, claims)


_CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:py|python)?\s*\n(.*?)```", re.DOTALL)


def _extract_previews(entry: dict) -> tuple[str, str, str]:
    """Pull thought / code / observation strings out of one raw step dict.

    Handles three formats seen in this repo:
      * ``Thought:`` + ```` ```py ```` fences (memory_timelapse default)
      * ``<code>...</code>`` tagged blocks with prose-before as the thought
      * ``code_action`` field already containing the code
    """
    model_output = entry.get("model_output") or ""
    code_action = entry.get("code_action") or ""
    observations = entry.get("observations")

    thought = ""
    code = ""

    # Try Thought: marker first
    if "Thought:" in model_output:
        for i, line in enumerate(model_output.splitlines()):
            if "Thought:" in line:
                tail = line.split("Thought:", 1)[1].strip()
                if tail:
                    thought = tail[:512]
                else:
                    rest = model_output.splitlines()[i + 1:i + 6]
                    thought = " ".join(r.strip() for r in rest)[:512]
                break

    if not thought:
        # Use the prose before the first code marker.
        for marker_re in (_CODE_TAG_RE, _CODE_FENCE_RE):
            m = marker_re.search(model_output)
            if m:
                thought = model_output[:m.start()].strip()[:512]
                break
        if not thought:
            thought = model_output.strip()[:512]

    # Code: prefer code_action when present, else extracted blocks.
    if code_action.strip():
        code = code_action
    else:
        for marker_re in (_CODE_TAG_RE, _CODE_FENCE_RE):
            m = marker_re.search(model_output)
            if m:
                code = m.group(1)
                break

    obs = ""
    if observations:
        if isinstance(observations, list):
            obs = "\n".join(str(o) for o in observations)
        else:
            obs = str(observations)
    return thought, code, obs


def _load_memory_steps(mem_dir: Path) -> list[StepInfo]:
    """Load every agent trace under ``mem_dir`` into a flat StepInfo list.

    Handles both layouts seen in this repo:
      * nested ``mem_dir/stage*/<agent>.json`` (the original timelapse layout)
      * flat ``mem_dir/<agent>.json`` (the current evolutionary runs)
    """
    if not mem_dir.exists():
        return []
    steps: list[StepInfo] = []
    raw_pool: list[tuple[dict, StepInfo]] = []

    nested = load_memory_files(mem_dir)
    for stage_steps in nested.values():
        for s in stage_steps:
            if str(s.get("agent", "")).startswith("verifier_"):
                continue
            raw_pool.append((s, parse_step(s)))

    import json
    for jf in sorted(mem_dir.glob("*.json")):
        agent = jf.stem
        if agent.startswith("verifier_"):
            continue
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            entry = dict(entry)
            entry.setdefault("step", entry.get("step_number", 0))
            entry.setdefault("stage", agent)
            entry.setdefault("agent", agent)
            raw_pool.append((entry, parse_step(entry)))

    # Re-extract previews using the richer rules above when parse_step
    # left fields empty (different memory layouts use different markers).
    for raw, info in raw_pool:
        t, c, o = _extract_previews(raw)
        if not info.thought_preview and t:
            info = StepInfo(**{**info.__dict__, "thought_preview": t[:512]})
        if not info.code_preview and c:
            info = StepInfo(**{**info.__dict__, "code_preview": c[:4096]})
        if not info.observation_preview and o:
            info = StepInfo(**{**info.__dict__, "observation_preview": o[:1024]})
        steps.append(info)

    steps.sort(key=lambda s: s.start_time)
    return steps


def _load_workflow(wf_dir: Path, memory_root: Path) -> Workflow | None:
    uuid = wf_dir.name
    lineage = wf_dir / f"lineage_{uuid}.json"
    if not lineage.exists():
        return None
    import json
    data = json.loads(lineage.read_text())

    steps = _load_memory_steps(memory_root / uuid)

    png = wf_dir / f"workflow_{uuid}.png"
    return Workflow(
        uuid=uuid,
        iteration=int(data.get("iteration", 0)),
        kind=str(data.get("evolution_kind", "seed")),
        parents=list(data.get("parents", [])),
        steps=steps,
        evaluation=_parse_evaluation(wf_dir / "evaluation.txt"),
        png_path=png if png.exists() else None,
    )


def load_all(workflows_dir: Path, memory_dir: Path) -> list[Workflow]:
    """Discover every workflow with a lineage file and load it."""
    out: list[Workflow] = []
    for child in sorted(workflows_dir.iterdir()):
        if not child.is_dir():
            continue
        wf = _load_workflow(child, memory_dir)
        if wf is not None:
            out.append(wf)
    out.sort(key=lambda w: w.iteration)
    return out


# ---------------------------------------------------------------------------
# Tree layout
# ---------------------------------------------------------------------------
@dataclass
class TreeNode:
    uuid: str
    iteration: int
    x: float
    y: float
    radius: float
    color: tuple[int, int, int]


def layout_tree(
    workflows: list[Workflow],
    rect: pygame.Rect,
    pad: int = 26,
) -> tuple[dict[str, TreeNode], list[tuple[str, str]]]:
    """Place each workflow on a horizontal band keyed by iteration."""
    nodes: dict[str, TreeNode] = {}
    if not workflows:
        return nodes, []

    by_iter: dict[int, list[Workflow]] = {}
    for wf in workflows:
        by_iter.setdefault(wf.iteration, []).append(wf)
    iters = sorted(by_iter)
    n_iter = len(iters)

    inner_w = rect.width - 2 * pad
    inner_h = rect.height - 2 * pad
    row_h = inner_h / max(n_iter, 1)
    radius = max(8.0, min(18.0, row_h * 0.28))

    for row, it in enumerate(iters):
        siblings = by_iter[it]
        n = len(siblings)
        for col, wf in enumerate(siblings):
            cx = rect.x + pad + inner_w * (col + 1) / (n + 1)
            cy = rect.y + pad + row_h * (row + 0.5)
            nodes[wf.uuid] = TreeNode(
                uuid=wf.uuid,
                iteration=wf.iteration,
                x=cx,
                y=cy,
                radius=radius,
                color=EVO_COLORS.get(wf.kind, ACCENT),
            )

    edges: list[tuple[str, str]] = []
    for wf in workflows:
        for p in wf.parents:
            if p in nodes:
                edges.append((p, wf.uuid))
    return nodes, edges


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def edge_curve_point(
    x0: float, y0: float, x1: float, y1: float, t: float,
) -> tuple[float, float]:
    """Point at ``t`` ∈ [0, 1] on a vertical S-curve between two tree nodes.

    Cubic bezier with control points directly below the parent and above
    the child, so edges leave and arrive vertically like a subway map.
    """
    u = 1.0 - t
    mid_y = (y0 + y1) / 2.0
    x = (u ** 3 + 3 * u * u * t) * x0 + (3 * u * t * t + t ** 3) * x1
    y = u ** 3 * y0 + 3 * u * t * mid_y + t ** 3 * y1
    return x, y


def edge_curve(
    x0: float, y0: float, x1: float, y1: float, segments: int = 18,
) -> list[tuple[float, float]]:
    """Polyline approximation of the S-curve edge."""
    return [
        edge_curve_point(x0, y0, x1, y1, i / segments)
        for i in range(segments + 1)
    ]


def glow_circle(surf: pygame.Surface, color, center, radius: int,
                layers: int = 3, alpha: int = 28):
    """Soft additive bloom around ``center``.

    BLEND_ADD ignores per-pixel alpha, so the color itself is pre-scaled
    by ``alpha`` — each layer then adds a dim wash that stacks smoothly.
    """
    x, y = int(center[0]), int(center[1])
    dim = tuple(c * alpha // 255 for c in color)
    for i in range(layers, 0, -1):
        r = radius + i * 4
        halo = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(halo, dim, (r, r), r)
        surf.blit(halo, (x - r, y - r), special_flags=pygame.BLEND_ADD)


def mix_color(a, b, t: float) -> tuple[int, int, int]:
    """Linear blend of two RGB colors, ``t`` toward ``b``."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def typewriter_slice(
    lines: list[str], frac: float,
) -> tuple[list[str], tuple[int, str] | None]:
    """Truncate ``lines`` to the first ``frac`` of their characters.

    Returns the visible lines plus (row, visible_text_of_row) for the
    cursor position, or None when everything is already revealed.
    """
    if frac >= 1.0:
        return lines, None
    total = sum(len(ln) + 1 for ln in lines)
    budget = int(total * max(0.0, frac))
    out: list[str] = []
    for i, ln in enumerate(lines):
        if budget >= len(ln) + 1:
            out.append(ln)
            budget -= len(ln) + 1
            continue
        out.append(ln[:budget])
        return out, (i, ln[:budget])
    return out, None


def draw_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    font: pygame.font.Font,
    accent: tuple[int, int, int] = ACCENT,
) -> pygame.Rect:
    """Render a card with title, glowing top accent, return inner area."""
    pygame.draw.rect(surf, BG_PANEL, rect, border_radius=10)
    pygame.draw.rect(surf, BORDER, rect, width=1, border_radius=10)
    bar = pygame.Rect(rect.x, rect.y, rect.width, 28)
    pygame.draw.rect(surf, BG_PANEL_LIGHT, bar, border_top_left_radius=10,
                     border_top_right_radius=10)
    pygame.draw.line(surf, accent, (rect.x + 14, rect.y + 27),
                     (rect.x + 14 + 4, rect.y + 27), 3)
    label = font.render(title, True, accent)
    # accent bar before the label (drawn — block glyphs render as tofu)
    pygame.draw.rect(surf, accent,
                     (rect.x + 14, rect.y + 8, 4, label.get_height() - 4),
                     border_radius=2)
    surf.blit(label, (rect.x + 26, rect.y + 7))
    # HUD corner ticks
    tick = 10
    for cx, cy, dx, dy in (
        (rect.x, rect.y, 1, 1), (rect.right - 1, rect.y, -1, 1),
        (rect.x, rect.bottom - 1, 1, -1),
        (rect.right - 1, rect.bottom - 1, -1, -1),
    ):
        pygame.draw.line(surf, accent, (cx, cy), (cx + dx * tick, cy), 1)
        pygame.draw.line(surf, accent, (cx, cy), (cx, cy + dy * tick), 1)
    return pygame.Rect(rect.x + 12, rect.y + 36, rect.width - 24,
                       rect.height - 46)


def draw_chevron(surf: pygame.Surface, x: int, y: int, color, size: int = 10):
    pts = [(x, y), (x + size, y + size // 2), (x, y + size)]
    pygame.draw.polygon(surf, color, pts)


def text_lines(
    surf: pygame.Surface, font: pygame.font.Font, lines: list[str],
    pos: tuple[int, int], color, line_h: int, max_lines: int | None = None
):
    x, y = pos
    if max_lines is not None:
        lines = lines[:max_lines]
    for line in lines:
        surf.blit(font.render(line, True, color), (x, y))
        y += line_h


def pil_to_surface(im: Image.Image) -> pygame.Surface:
    if im.mode != "RGB":
        im = im.convert("RGB")
    return pygame.image.fromstring(im.tobytes(), im.size, "RGB")


def fit_surface(src: pygame.Surface, max_w: int, max_h: int) -> pygame.Surface:
    sw, sh = src.get_size()
    if sw == 0 or sh == 0 or max_w <= 0 or max_h <= 0:
        return src
    scale = min(max_w / sw, max_h / sh)
    return pygame.transform.smoothscale(src, (int(sw * scale), int(sh * scale)))


def wrap_text(text: str, font: pygame.font.Font, max_w: int) -> list[str]:
    out: list[str] = []
    for para in text.splitlines() or [""]:
        words = para.split(" ")
        line = ""
        for w in words:
            cand = (line + " " + w).strip()
            if font.size(cand)[0] <= max_w:
                line = cand
            else:
                if line:
                    out.append(line)
                line = w
        out.append(line)
    return out


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def _parent_score(wf: Workflow, workflows: list[Workflow]) -> float | None:
    """Overall score of the first evaluated genetic parent, if any."""
    by_uuid = {w.uuid: w for w in workflows}
    for p in wf.parents:
        parent = by_uuid.get(p)
        if parent is not None and parent.evaluation is not None:
            return parent.evaluation.overall
    return None


def _draw_delta_badge(surf, fonts, x: int, y: int, wf: Workflow,
                      workflows: list[Workflow]) -> None:
    """Small ▲/▼ score-vs-parent indicator next to the header meta line."""
    parent = _parent_score(wf, workflows)
    if parent is None or wf.evaluation is None:
        return
    delta = wf.evaluation.overall - parent
    up = delta >= 0
    color = SUCCESS if up else ERROR
    cy = y + 7
    if up:
        pts = [(x, cy + 4), (x + 8, cy + 4), (x + 4, cy - 4)]
    else:
        pts = [(x, cy - 4), (x + 8, cy - 4), (x + 4, cy + 4)]
    pygame.draw.polygon(surf, color, pts)
    label = fonts["small"].render(f"{delta:+.3f} vs parent", True, color)
    surf.blit(label, (x + 14, y))


def _draw_sparkline(surf, rect: pygame.Rect, workflows: list[Workflow],
                    current_uuid: str) -> None:
    """Tiny overall-score history polyline for the header's right side."""
    scores = [w.evaluation.overall if w.evaluation else 0.0
              for w in workflows]
    if len(scores) < 2:
        return
    pts = []
    for i, s in enumerate(scores):
        x = rect.x + rect.width * i / (len(scores) - 1)
        y = rect.bottom - rect.height * max(0.0, min(s, 1.0))
        pts.append((int(x), int(y)))
    pygame.draw.lines(surf, ACCENT_DIM, False, pts, 1)
    for i, (x, y) in enumerate(pts):
        if workflows[i].uuid == current_uuid:
            pygame.draw.circle(surf, AMBER, (x, y), 3)


def draw_header(surf, rect, fonts, wf: Workflow, workflows: list[Workflow],
                step_idx: int, n_steps: int, playing: bool, speed: float):
    pygame.draw.rect(surf, BG_PANEL, rect)
    pygame.draw.line(surf, BORDER_BRIGHT, (rect.x, rect.bottom - 1),
                     (rect.right, rect.bottom - 1), 1)

    title = fonts["title"].render("MIMOSA-AI", True, ACCENT)
    sub = fonts["body"].render("WORKFLOW EVOLUTION", True, TEXT)
    surf.blit(title, (rect.x + 24, rect.y + 10))
    surf.blit(sub, (rect.x + 24 + title.get_width() + 14, rect.y + 16))

    meta = (
        f"uuid {wf.uuid}    iter {wf.iteration + 1}/{len(workflows)}    "
        f"kind {wf.kind}    step {step_idx + 1}/{max(n_steps, 1)}"
    )
    meta_surf = fonts["small"].render(meta, True, TEXT_DIM)
    surf.blit(meta_surf, (rect.x + 24, rect.y + 44))
    _draw_delta_badge(surf, fonts, rect.x + 24 + meta_surf.get_width() + 20,
                      rect.y + 44, wf, workflows)

    color = SUCCESS if playing else AMBER
    text = fonts["body"].render("PLAY" if playing else "PAUSED", True, color)
    bw = text.get_width() + 40
    bx = rect.right - bw - 24
    pygame.draw.rect(surf, BG_PANEL_LIGHT,
                     (bx, rect.y + 14, bw, 30), border_radius=6)
    pygame.draw.rect(surf, color, (bx, rect.y + 14, bw, 30), width=1,
                     border_radius=6)
    # drawn play/pause icon (font glyphs render as tofu on some systems)
    ix, iy = bx + 11, rect.y + 22
    if playing:
        pygame.draw.polygon(surf, color,
                            [(ix, iy), (ix, iy + 13), (ix + 10, iy + 6)])
    else:
        pygame.draw.rect(surf, color, (ix, iy, 4, 13))
        pygame.draw.rect(surf, color, (ix + 7, iy, 4, 13))
    surf.blit(text, (bx + 28, rect.y + 18))

    speed_txt = fonts["small"].render(f"x{speed:.1f}", True, TEXT_DIM)
    surf.blit(speed_txt, (bx - speed_txt.get_width() - 14, rect.y + 22))

    spark = pygame.Rect(bx - 150, rect.y + 48, 126, 22)
    _draw_sparkline(surf, spark, workflows, wf.uuid)


# ---------------------------------------------------------------------------
# Lineage tree panel
# ---------------------------------------------------------------------------
def _ancestor_uuids(workflows: list[Workflow], current_uuid: str) -> set:
    """UUIDs on any lineage path from the roots down to the current node."""
    by_uuid = {w.uuid: w for w in workflows}
    seen: set = set()
    frontier = [current_uuid]
    while frontier:
        u = frontier.pop()
        if u in seen or u not in by_uuid:
            continue
        seen.add(u)
        frontier.extend(by_uuid[u].parents)
    return seen


def _draw_tree_edges(surf, nodes, edges, by_uuid, ancestry: set,
                     anim_t: float) -> None:
    """Curved edges; the ancestry path glows and carries flow dots."""
    for ei, (parent, child) in enumerate(edges):
        a, b = nodes[parent], nodes[child]
        color = EVO_COLORS.get(by_uuid[child].kind, ACCENT_DIM)
        on_path = parent in ancestry and child in ancestry
        pts = edge_curve(a.x, a.y, b.x, b.y)
        if on_path:
            pygame.draw.lines(surf, color, False, pts, 2)
            # two flow dots per edge, phase-shifted, drifting parent → child
            for k in range(2):
                t = (anim_t / FLOW_PERIOD_S + ei * 0.37 + k * 0.5) % 1.0
                px, py = edge_curve_point(a.x, a.y, b.x, b.y, t)
                glow_circle(surf, color, (px, py), 2, layers=1, alpha=60)
                pygame.draw.circle(surf, TEXT, (int(px), int(py)), 2)
        else:
            pygame.draw.lines(surf, mix_color(color, BG, 0.55), False, pts, 1)


def _draw_tree_node(surf, fonts, n: TreeNode, wf: Workflow,
                    is_current: bool, is_best: bool, anim_t: float) -> None:
    score = wf.evaluation.overall if wf.evaluation else 0.0
    score = max(0.0, min(1.0, score))
    r = n.radius + (4 if is_current else 0)
    if is_current:
        breath = 0.5 + 0.5 * math.sin(anim_t * 2 * math.pi / PULSE_PERIOD_S)
        glow_circle(surf, ACCENT, (n.x, n.y), int(r + 2 + 3 * breath))
    # outer ring tinted by score
    ring = (
        int(60 + 160 * score),
        int(80 + 140 * score),
        int(120 + 60 * (1 - score)),
    )
    pygame.draw.circle(surf, ring, (int(n.x), int(n.y)), int(r) + 2, 2)
    pygame.draw.circle(surf, n.color, (int(n.x), int(n.y)), int(r))
    if is_current:
        pygame.draw.circle(surf, TEXT, (int(n.x), int(n.y)), int(r), 2)
    if is_best:
        dx, dy = int(n.x + r + 7), int(n.y - r - 3)
        pygame.draw.polygon(surf, AMBER, [
            (dx, dy - 4), (dx + 4, dy), (dx, dy + 4), (dx - 4, dy)])
    lbl = fonts["tiny"].render(str(n.iteration), True, BG)
    surf.blit(lbl, (n.x - lbl.get_width() / 2, n.y - lbl.get_height() / 2))


def draw_tree(
    surf, area: pygame.Rect, fonts,
    workflows: list[Workflow], current_uuid: str, anim_t: float,
) -> dict[str, TreeNode]:
    nodes, edges = layout_tree(workflows, area)
    by_uuid = {w.uuid: w for w in workflows}
    ancestry = _ancestor_uuids(workflows, current_uuid)
    scored = [w for w in workflows if w.evaluation is not None]
    best_uuid = (max(scored, key=lambda w: w.evaluation.overall).uuid
                 if scored else None)

    # iteration grid lines
    iters = sorted({n.iteration for n in nodes.values()})
    for it in iters:
        ys = [n.y for n in nodes.values() if n.iteration == it]
        if not ys:
            continue
        y = int(sum(ys) / len(ys))
        pygame.draw.line(surf, (24, 32, 46), (area.x + 8, y),
                         (area.right - 8, y), 1)
        label = fonts["tiny"].render(f"#{it:02d}", True, TEXT_FAINT)
        surf.blit(label, (area.x + 4, y - label.get_height() // 2))

    _draw_tree_edges(surf, nodes, edges, by_uuid, ancestry, anim_t)
    for n in nodes.values():
        _draw_tree_node(surf, fonts, n, by_uuid[n.uuid],
                        n.uuid == current_uuid, n.uuid == best_uuid, anim_t)

    # legend
    lx = area.x + 12
    ly = area.bottom - 22
    for kind, color in EVO_COLORS.items():
        pygame.draw.circle(surf, color, (lx + 6, ly + 8), 5)
        t = fonts["tiny"].render(kind, True, TEXT_DIM)
        surf.blit(t, (lx + 16, ly + 2))
        lx += 18 + t.get_width() + 16
    if best_uuid is not None:
        pygame.draw.polygon(surf, AMBER, [
            (lx + 6, ly + 4), (lx + 10, ly + 8), (lx + 6, ly + 12),
            (lx + 2, ly + 8)])
        t = fonts["tiny"].render("best", True, TEXT_DIM)
        surf.blit(t, (lx + 16, ly + 2))
    return nodes


# ---------------------------------------------------------------------------
# Memory timelapse panel
# ---------------------------------------------------------------------------
SUB_PANEL_SPECS = (
    ("THOUGHT", ACCENT, "thought_preview", TEXT, False),
    ("CODE", SUCCESS, "code_preview", (180, 220, 250), True),
    ("OBSERVATION", AMBER, "observation_preview", (200, 230, 200), False),
)


def draw_step_panel(surf, area: pygame.Rect, fonts, wf: Workflow,
                    step: StepInfo | None, step_idx: int, sub_idx: int,
                    reveal: float, anim_t: float):
    """Render the memory-timelapse panel for one (step, sub_idx) frame.

    Minimal: body of the active sub-panel only. A thin left-edge accent
    stripe colors which sub-panel is showing (cyan=thought, green=code,
    amber=observation). The agent name sits as a thin line at the top.
    While playing, the body types itself out (``reveal`` ∈ [0, 1]) with a
    blinking cursor; paused/scrubbed frames show the full text at once.
    """
    inner = area

    if step is None:
        msg = fonts["body"].render("No memory trace for this workflow.",
                                   True, TEXT_DIM)
        surf.blit(msg, (inner.x + 12, inner.y + 12))
        return

    sub_idx = max(0, min(len(SUB_PANEL_SPECS) - 1, sub_idx))
    _label, accent, attr, body_color, mono = SUB_PANEL_SPECS[sub_idx]
    body = getattr(step, attr, "") or ""

    stage_color = STAGE_COLORS.get(step.stage, ACCENT)
    surf.blit(fonts["small"].render(step.agent[:48], True, stage_color),
              (inner.x + 8, inner.y))
    if step.has_error:
        t = fonts["tiny"].render("ERROR", True, ERROR)
        surf.blit(t, (inner.right - t.get_width() - 8, inner.y + 4))

    body_top = inner.y + 24
    body_rect = pygame.Rect(inner.x, body_top, inner.width,
                            inner.bottom - body_top)
    pygame.draw.rect(surf, accent,
                     (body_rect.x, body_rect.y, 3, body_rect.height),
                     border_radius=2)

    font = fonts["step_mono"] if mono else fonts["step_text"]
    line_h = font.get_height() + 4
    text_x = body_rect.x + 14
    if not body:
        surf.blit(font.render("—", True, TEXT_FAINT),
                  (text_x, body_rect.y + 4))
        return
    max_w = body_rect.width - 18
    if mono:
        lines = [ln[:200] for ln in body.split("\n")]
    else:
        lines = wrap_text(body, font, max_w)
    max_lines = max(body_rect.height // line_h, 1)
    lines = lines[:max_lines]
    lines, cursor = typewriter_slice(lines, reveal)
    text_lines(surf, font, lines, (text_x, body_rect.y + 4),
               body_color, line_h)
    if cursor is not None and (anim_t * 3.0) % 1.0 < 0.65:
        row, visible = cursor
        cx = text_x + font.size(visible)[0] + 2
        cy = body_rect.y + 4 + row * line_h
        pygame.draw.rect(surf, accent, (cx, cy + 2, 8, line_h - 6))


# ---------------------------------------------------------------------------
# Workflow PNG panel
# ---------------------------------------------------------------------------
_PNG_CACHE: dict[str, pygame.Surface] = {}


def get_workflow_png(path: Path) -> pygame.Surface | None:
    key = str(path)
    if key in _PNG_CACHE:
        return _PNG_CACHE[key]
    try:
        im = Image.open(path)
        surf = pil_to_surface(im)
        _PNG_CACHE[key] = surf
        return surf
    except Exception:
        return None


def _draw_png_placeholder(surf, area, fonts, anim_t: float):
    """Rotating radar rings shown when a workflow has no rendered graph."""
    cx, cy = area.centerx, area.centery - 8
    max_r = max(min(area.width, area.height) // 4, 24)
    for r in (max_r, int(max_r * 0.66), int(max_r * 0.33)):
        pygame.draw.circle(surf, BORDER, (cx, cy), r, 1)
    box = pygame.Rect(cx - max_r, cy - max_r, max_r * 2, max_r * 2)
    sweep = anim_t * 2 * math.pi / 4.0
    for start, span in ((sweep, 1.1), (sweep + math.pi, 0.6)):
        pygame.draw.arc(surf, ACCENT_DIM, box, start, start + span, 2)
    pygame.draw.circle(surf, ACCENT_DIM, (cx, cy), 3)
    msg = fonts["small"].render("NO WORKFLOW GRAPH", True, TEXT_FAINT)
    surf.blit(msg, (cx - msg.get_width() // 2, cy + max_r + 12))


def draw_workflow_png(surf, area, fonts, wf: Workflow, anim_t: float):
    if wf.png_path is None:
        _draw_png_placeholder(surf, area, fonts, anim_t)
        return
    src = get_workflow_png(wf.png_path)
    if src is None:
        return
    fitted = fit_surface(src, area.width - 12, area.height - 12)
    ox = area.x + (area.width - fitted.get_width()) // 2
    oy = area.y + (area.height - fitted.get_height()) // 2
    surf.blit(fitted, (ox, oy))


# ---------------------------------------------------------------------------
# Rubric evolution panel
# ---------------------------------------------------------------------------
def _claim_summary(workflows: list[Workflow]) -> list[tuple[str, int]]:
    """Return list of (claim_name, max_importance) ordered first-seen."""
    order: list[str] = []
    imp: dict[str, int] = {}
    for wf in workflows:
        if not wf.evaluation:
            continue
        for c in wf.evaluation.claims:
            if c.name not in imp:
                order.append(c.name)
            imp[c.name] = max(imp.get(c.name, 0), c.importance)
    return [(n, imp[n]) for n in order]


def _truncate_to_width(text: str, font, max_w: int) -> str:
    if font.size(text)[0] <= max_w:
        return text
    while text and font.size(text + "…")[0] > max_w:
        text = text[:-1]
    return text + "…" if text else ""


def draw_rubric(surf, area, fonts, workflows: list[Workflow],
                current_idx: int,
                completed_mask: list[bool] | None = None,
                col_flash: list[float] | None = None,
                anim_t: float = 0.0):
    """Heatmap of per-claim pass/fail per generation + overall-score line.

    Cells for workflows whose sub-animation has not yet played stay grey
    (``completed_mask[j]`` False) — they reveal pass/fail only after the
    user has scrubbed past their last frame. ``col_flash[j]`` ∈ [0, 1]
    briefly whitens a column right after it completes.
    """
    all_claims = _claim_summary(workflows)
    if not all_claims or not workflows:
        msg = fonts["small"].render("No evaluation data.", True, TEXT_DIM)
        surf.blit(msg, (area.x + 8, area.y + 8))
        return

    # Reserve room for the line chart at the bottom.
    line_panel_h = min(max(40, area.height // 5), 50)
    grid_top = area.y
    grid_bottom = area.bottom - line_panel_h
    grid_h = grid_bottom - grid_top

    # Tighter rows so the panel can be short without dropping rubric rows.
    label_font = fonts["tiny"]
    row_h_min = max(label_font.get_height(), 14)
    max_rows = max(1, grid_h // row_h_min)
    max_rows = min(max_rows, 10)

    # Prioritise claims that exist in the *current* workflow so the
    # heatmap is relevant to what the user is actually looking at.
    current_wf = (
        workflows[current_idx]
        if 0 <= current_idx < len(workflows)
        else None
    )
    current_names: set = set()
    if current_wf and current_wf.evaluation:
        current_names = {c.name for c in current_wf.evaluation.claims}

    current_claims = [c for c in all_claims if c[0] in current_names]
    other_claims = [c for c in all_claims if c[0] not in current_names]
    claims = current_claims[:max_rows]
    if len(claims) < max_rows:
        claims.extend(other_claims[: max_rows - len(claims)])
    # Keep original first-seen order so the heatmap reads top-down.
    name_order = {n: i for i, (n, _) in enumerate(all_claims)}
    claims.sort(key=lambda kv: name_order[kv[0]])

    name_w = min(220, area.width // 3)
    grid_x = area.x + name_w + 8
    grid_w = area.right - grid_x - 8
    cell_w = max(grid_w / len(workflows), 3.0)
    cell_h = grid_h / max(len(claims), 1)

    # claim labels
    for i, (name, _imp) in enumerate(claims):
        label = _truncate_to_width(name, label_font, name_w - 6)
        t = label_font.render(label, True, TEXT_DIM)
        y = grid_top + int(i * cell_h + cell_h / 2 - t.get_height() / 2)
        surf.blit(t, (area.x, y))

    # status grid
    pending_color = (78, 86, 100)
    for j, wf in enumerate(workflows):
        if not wf.evaluation:
            continue
        is_done = completed_mask[j] if completed_mask else True
        by_name = {c.name: c for c in wf.evaluation.claims}
        for i, (name, _imp) in enumerate(claims):
            c = by_name.get(name)
            if not is_done:
                color = pending_color
            elif c is None:
                color = (28, 36, 50)
            elif c.status == "pass":
                color = SUCCESS
            elif c.status == "fail":
                color = ERROR
            elif c.status == "error":
                color = WARN
            else:
                color = TEXT_FAINT
            if is_done and col_flash and col_flash[j] > 0:
                color = mix_color(color, (255, 255, 255),
                                  col_flash[j] * 0.55)
            x = grid_x + int(j * cell_w)
            y = grid_top + int(i * cell_h)
            pygame.draw.rect(surf, color,
                             (x + 1, y + 1, max(int(cell_w) - 2, 1),
                              max(int(cell_h) - 2, 1)))
    # current-column marker (drawn once, on top of all cells)
    if 0 <= current_idx < len(workflows):
        x = grid_x + int((current_idx + 0.5) * cell_w)
        pygame.draw.line(surf, ACCENT, (x, grid_top), (x, grid_bottom), 1)

    # overall-score line chart below the grid
    line_y0 = grid_bottom + 18
    line_y1 = area.bottom - 4
    line_h = line_y1 - line_y0
    pygame.draw.line(surf, BORDER, (grid_x, line_y0),
                     (grid_x + grid_w, line_y0), 1)
    pts: list[tuple[int, int]] = []
    for j, wf in enumerate(workflows):
        s = wf.evaluation.overall if wf.evaluation else 0.0
        x = grid_x + int(j * cell_w + cell_w / 2)
        y = int(line_y1 - line_h * max(0.0, min(s, 1.0)))
        pts.append((x, y))
    if len(pts) >= 2 and grid_w > 0 and line_h > 0:
        # translucent fill under the score curve
        overlay = pygame.Surface((grid_w, line_h + 4), pygame.SRCALPHA)
        local = [(x - grid_x, y - line_y0) for x, y in pts]
        poly = local + [(local[-1][0], line_h + 4), (local[0][0], line_h + 4)]
        pygame.draw.polygon(overlay, (*ACCENT, 34), poly)
        surf.blit(overlay, (grid_x, line_y0))
        pygame.draw.lines(surf, ACCENT, False, pts, 2)
    breath = 0.5 + 0.5 * math.sin(anim_t * 2 * math.pi / PULSE_PERIOD_S)
    scored_js = [j for j, w in enumerate(workflows) if w.evaluation]
    best_j = (max(scored_js,
                  key=lambda j: workflows[j].evaluation.overall)
              if scored_js else None)
    for j, (x, y) in enumerate(pts):
        if j == current_idx:
            glow_circle(surf, AMBER, (x, y), 3, layers=2, alpha=45)
            pygame.draw.circle(surf, AMBER, (x, y), int(3 + 1.5 * breath))
        else:
            pygame.draw.circle(surf, ACCENT, (x, y), 3)
    if best_j is not None:
        bx, by = pts[best_j]
        pygame.draw.polygon(surf, AMBER, [
            (bx, by - 9), (bx + 5, by - 4), (bx, by + 1), (bx - 5, by - 4)])
        tag = fonts["tiny"].render(
            f"BEST {workflows[best_j].evaluation.overall:.2f}", True, AMBER)
        # place the tag below the marker when the point sits near the top
        ty = by + 4 if by - 16 < line_y0 else by - 16
        surf.blit(tag, (min(bx + 8, area.right - tag.get_width()), ty))
    # one-line footer in the gap between grid and chart: current-gen
    # stats on the left, claim-count on the right
    footer_y = grid_bottom + 3
    wf = workflows[current_idx] if 0 <= current_idx < len(workflows) else None
    if wf and wf.evaluation:
        e = wf.evaluation
        info = (
            f"gen {wf.iteration:02d}  •  score {e.overall:.3f}  •  "
            f"pass {e.n_pass}  fail {e.n_fail}  err {e.n_error}"
        )
        surf.blit(label_font.render(info, True, TEXT), (area.x, footer_y))
    total_current = len(current_names)
    total_all = len(all_claims)
    if total_current > 0:
        count_txt = f"{len(claims)}/{total_current} claims ({total_all} total)"
    else:
        count_txt = f"{len(claims)}/{total_all} claims"
    count_surf = label_font.render(count_txt, True, TEXT_FAINT)
    surf.blit(count_surf,
              (grid_x + grid_w - count_surf.get_width(), footer_y))


# ---------------------------------------------------------------------------
# Bottom timeline / controls
# ---------------------------------------------------------------------------
def draw_timeline(surf, rect, fonts, total_frames, current_frame,
                  workflows: list[Workflow], frame_to_wf: list[int],
                  anim_t: float):
    pygame.draw.rect(surf, BG_PANEL, rect)
    pygame.draw.line(surf, BORDER, (rect.x, rect.y), (rect.right, rect.y), 1)

    bar = pygame.Rect(rect.x + 24, rect.y + 22, rect.width - 48, 10)
    pygame.draw.rect(surf, BG_PANEL_LIGHT, bar, border_radius=5)

    if total_frames > 1:
        # color segments per workflow
        for f in range(total_frames):
            wf = workflows[frame_to_wf[f]]
            color = EVO_COLORS.get(wf.kind, ACCENT_DIM)
            x = bar.x + int(bar.width * f / total_frames)
            x2 = bar.x + int(bar.width * (f + 1) / total_frames)
            pygame.draw.rect(surf, color, (x, bar.y + 3, max(x2 - x, 1), 4))
        # boundaries
        last_wf = -1
        for f in range(total_frames):
            wf_idx = frame_to_wf[f]
            if wf_idx != last_wf:
                last_wf = wf_idx
                x = bar.x + int(bar.width * f / total_frames)
                pygame.draw.line(surf, BORDER_BRIGHT, (x, bar.y - 2),
                                 (x, bar.y + bar.height + 2), 1)

    pos_x = bar.x + int(bar.width * current_frame / max(total_frames, 1))
    pos_y = bar.y + bar.height // 2
    breath = 0.5 + 0.5 * math.sin(anim_t * 2 * math.pi / PULSE_PERIOD_S)
    glow_circle(surf, ACCENT, (pos_x, pos_y), int(6 + 3 * breath),
                layers=2, alpha=40)
    pygame.draw.circle(surf, ACCENT, (pos_x, pos_y), 8)
    pygame.draw.circle(surf, BG, (pos_x, pos_y), 4)

    hint = (
        "SPACE play/pause   LEFT/RIGHT step   UP/DOWN generation   "
        "B best   [ ] speed   H help   click tree or timeline to jump"
    )
    t = fonts["tiny"].render(hint, True, TEXT_FAINT)
    surf.blit(t, (rect.x + 24, rect.bottom - 18))


# ---------------------------------------------------------------------------
# Help overlay
# ---------------------------------------------------------------------------
HELP_LINES = [
    "WORKFLOW EVOLUTION ANIMATION — KEYS",
    "",
    "  Space          play / pause",
    "  Left / Right   prev / next memory step",
    "  Up / Down      jump to prev / next generation",
    "  B              jump to the best-scoring generation",
    "  Home / End     first / last frame",
    "  [ / ]          slow down / speed up",
    "  H              toggle this help",
    "  Esc / Q        quit",
    "",
    "Click a node in the lineage tree to jump to that generation.",
    "Click the bottom timeline to scrub through all (workflow, step) frames.",
]


def draw_help(surf, fonts):
    w, h = surf.get_size()
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surf.blit(overlay, (0, 0))
    box = pygame.Rect(w // 2 - 320, h // 2 - 180, 640, 360)
    pygame.draw.rect(surf, BG_PANEL, box, border_radius=12)
    pygame.draw.rect(surf, ACCENT, box, width=2, border_radius=12)
    for i, line in enumerate(HELP_LINES):
        color = ACCENT if i == 0 else TEXT if line.strip() else TEXT_FAINT
        t = fonts["body"].render(line, True, color)
        surf.blit(t, (box.x + 24, box.y + 24 + i * 26))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class App:
    def __init__(self, workflows: list[Workflow],
                 size=(1600, 1000), fps: int = 60):
        if not workflows:
            raise SystemExit("No workflows with lineage files found.")
        self.workflows = workflows
        self.fps = fps

        pygame.init()
        pygame.display.set_caption("Mimosa — Workflow Evolution")
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.fonts = self._load_fonts()

        # flatten to (workflow_idx, step_idx, sub_idx) frames. Each step
        # expands into one frame per sub-panel (thought/code/observation) so
        # they play one-at-a-time. Workflows with no memory still get a single
        # "intro" frame so they appear in the timeline.
        self.frames: list[tuple[int, int, int]] = []
        n_sub = len(SUB_PANEL_SPECS)
        for wi, wf in enumerate(workflows):
            if not wf.steps:
                self.frames.append((wi, -1, 0))
            else:
                for si in range(len(wf.steps)):
                    for sub in range(n_sub):
                        self.frames.append((wi, si, sub))
        self.frame_to_wf = [wi for (wi, _, _) in self.frames]
        # Last frame index for each workflow — used by the rubric panel to
        # decide which columns are "complete" (cells colored) vs still
        # pending (cells grey).
        self._last_frame_of_wf: list[int] = [-1] * len(workflows)
        for fi, (wi, _, _) in enumerate(self.frames):
            self._last_frame_of_wf[wi] = fi

        self.cur = 0
        self.playing = True
        self.speed = 2.0     # steps per second at speed 1
        self._acc = 0.0
        self.anim_t = 0.0    # ambient animation clock (advanced by dt)
        self.show_help = False
        self._tree_nodes: dict[str, TreeNode] = {}
        self._timeline_rect: pygame.Rect | None = None
        self._prev_wi: int = self.frames[0][0] if self.frames else 0
        self.transition: Transition | None = None

        # Ambient FX state: fixed-seed starfield + cached overlay surfaces.
        rng = random.Random(STAR_SEED)
        self._stars = [
            (rng.random(), rng.random(),          # position (0..1 space)
             rng.uniform(0.004, 0.028),           # drift speed (screens/s)
             rng.choice((1, 1, 2)),               # radius px
             rng.uniform(0, math.tau))            # twinkle phase
            for _ in range(STAR_COUNT)
        ]
        self._vignette: pygame.Surface | None = None
        self._scan_band: pygame.Surface | None = None

    @staticmethod
    def _load_fonts() -> dict[str, pygame.font.Font]:
        def f(name, size, bold=False):
            try:
                return pygame.font.SysFont(name, size, bold=bold)
            except Exception:
                return pygame.font.Font(None, size)
        return {
            "title": f("Inter,Helvetica,Arial", 26, bold=True),
            "header": f("Inter,Helvetica,Arial", 18, bold=True),
            "body": f("Inter,Helvetica,Arial", 16),
            "small": f("Inter,Helvetica,Arial", 13),
            "tiny": f("Inter,Helvetica,Arial", 11),
            "mono": f("Menlo,Consolas,DejaVu Sans Mono,monospace", 12),
            "step_text": f("Inter,Helvetica,Arial", 18),
            "step_mono": f("Menlo,Consolas,DejaVu Sans Mono,monospace", 15),
        }

    # ----- state helpers -------------------------------------------------
    @property
    def current_workflow(self) -> Workflow:
        wi, _, _ = self.frames[self.cur]
        return self.workflows[wi]

    @property
    def current_step(self) -> StepInfo | None:
        wi, si, _ = self.frames[self.cur]
        wf = self.workflows[wi]
        return wf.steps[si] if si >= 0 and si < len(wf.steps) else None

    def step_index_in_wf(self) -> int:
        return max(self.frames[self.cur][1], 0)

    def current_sub_index(self) -> int:
        return self.frames[self.cur][2]

    def _seek(self, frame: int):
        """Jump to a frame and restart the typewriter reveal."""
        self.cur = max(0, min(len(self.frames) - 1, frame))
        self._acc = 0.0

    def jump_workflow(self, delta: int):
        wi = self.frames[self.cur][0]
        new_wi = max(0, min(len(self.workflows) - 1, wi + delta))
        for i, (w, _, _) in enumerate(self.frames):
            if w == new_wi:
                self._seek(i)
                return

    def jump_to_workflow_uuid(self, uuid: str):
        for i, (wi, _, _) in enumerate(self.frames):
            if self.workflows[wi].uuid == uuid:
                self._seek(i)
                return

    def jump_to_best(self):
        """Jump to the workflow with the highest overall score."""
        scored = [w for w in self.workflows if w.evaluation is not None]
        if scored:
            best = max(scored, key=lambda w: w.evaluation.overall)
            self.jump_to_workflow_uuid(best.uuid)

    # ----- main loop -----------------------------------------------------
    def run(self):
        while True:
            dt = self.clock.tick(self.fps) / 1000.0
            self.anim_t += dt
            if not self._handle_events():
                return
            if self.playing:
                self._acc += dt * self.speed
                while self._acc >= 1.0 and self.cur < len(self.frames) - 1:
                    self._acc -= 1.0
                    self.cur += 1
                if self.cur >= len(self.frames) - 1:
                    self.playing = False
            cur_wi = self.frames[self.cur][0]
            if cur_wi != self._prev_wi:
                self._start_transition(cur_wi)
                self._prev_wi = cur_wi
            if self.transition is not None:
                self.transition.elapsed += dt
                if self.transition.elapsed >= self.transition.duration:
                    self.transition = None
            self._draw()
            pygame.display.flip()

    def record(self, out_path: Path, fps: int = 30, hold_seconds: float = 1.0):
        """Render the full animation deterministically into an .mp4 via ffmpeg.

        Drives the same draw loop as ``run`` but at a fixed ``dt = 1/fps``
        instead of real wall-clock time, so the resulting video plays back
        at ``self.speed`` steps per second exactly. Requires ``ffmpeg`` on PATH.
        """
        w, h = self.screen.get_size()
        if w % 2 or h % 2:
            sys.exit(
                f"--record needs even width/height for H.264; got {w}x{h}."
            )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "20",
            str(out_path),
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except FileNotFoundError:
            sys.exit(
                "ffmpeg not found on PATH — install ffmpeg to use --record."
            )

        dt = 1.0 / fps
        hold_frames = max(int(hold_seconds * fps), 0)
        held = 0
        n_video_frames = 0
        self.cur = 0
        self._acc = 0.0
        self.anim_t = 0.0
        self.playing = True
        while True:
            self.anim_t += dt
            if self.playing:
                self._acc += dt * self.speed
                while self._acc >= 1.0 and self.cur < len(self.frames) - 1:
                    self._acc -= 1.0
                    self.cur += 1
                if self.cur >= len(self.frames) - 1:
                    self.playing = False
            cur_wi = self.frames[self.cur][0]
            if cur_wi != self._prev_wi:
                self._start_transition(cur_wi)
                self._prev_wi = cur_wi
            if self.transition is not None:
                self.transition.elapsed += dt
                if self.transition.elapsed >= self.transition.duration:
                    self.transition = None
            self._draw()
            proc.stdin.write(pygame.image.tostring(self.screen, "RGB"))
            n_video_frames += 1
            if not self.playing and self.transition is None:
                held += 1
                if held >= hold_frames:
                    break
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            sys.exit(f"ffmpeg exited with code {rc}")
        dur = n_video_frames / fps
        print(
            f"Wrote {out_path}  ({n_video_frames} frames, {dur:.1f}s @ "
            f"{fps} fps, playback speed x{self.speed:.2f})"
        )

    def _start_transition(self, new_wi: int):
        """Kick off the evolve animation from the parent (or prev wf) → new."""
        new_wf = self.workflows[new_wi]
        known = {w.uuid for w in self.workflows}
        parents = [p for p in new_wf.parents if p in known]
        from_uuid = parents[0] if parents else self.workflows[self._prev_wi].uuid
        if from_uuid == new_wf.uuid:
            self.transition = None   # drop any stale pulse toward old target
            return
        self.transition = Transition(
            from_uuid=from_uuid,
            to_uuid=new_wf.uuid,
            from_wi=self._prev_wi,
        )

    # ----- input ---------------------------------------------------------
    def _handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if ev.key == pygame.K_SPACE:
                    self.playing = not self.playing
                elif ev.key == pygame.K_RIGHT:
                    self._seek(self.cur + 1)
                elif ev.key == pygame.K_LEFT:
                    self._seek(self.cur - 1)
                elif ev.key == pygame.K_DOWN:
                    self.jump_workflow(+1)
                elif ev.key == pygame.K_UP:
                    self.jump_workflow(-1)
                elif ev.key == pygame.K_HOME:
                    self._seek(0)
                elif ev.key == pygame.K_END:
                    self._seek(len(self.frames) - 1)
                elif ev.key in (pygame.K_LEFTBRACKET,):
                    self.speed = max(0.25, self.speed / 1.4)
                elif ev.key in (pygame.K_RIGHTBRACKET,):
                    self.speed = min(16.0, self.speed * 1.4)
                elif ev.key == pygame.K_b:
                    self.jump_to_best()
                elif ev.key == pygame.K_h:
                    self.show_help = not self.show_help
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                self._handle_click(ev.pos)
        return True

    def _handle_click(self, pos):
        # bottom timeline scrub
        if self._timeline_rect and self._timeline_rect.collidepoint(pos):
            bar_x = self._timeline_rect.x + 24
            bar_w = self._timeline_rect.width - 48
            t = (pos[0] - bar_x) / max(bar_w, 1)
            self._seek(int(t * len(self.frames)))
            return
        # tree node click — closest node within its radius
        for n in self._tree_nodes.values():
            if math.hypot(pos[0] - n.x, pos[1] - n.y) <= n.radius + 4:
                self.jump_to_workflow_uuid(n.uuid)
                return

    # ----- draw ----------------------------------------------------------
    def _layout(self) -> dict[str, pygame.Rect]:
        w, h = self.screen.get_size()
        pad = 12
        header = pygame.Rect(0, 0, w, 80)
        timeline = pygame.Rect(0, h - 60, w, 60)
        body_top = header.bottom + pad
        body_bottom = timeline.y - pad
        left_w = int(w * 0.36)
        right_x = pad + left_w + pad
        right_w = w - right_x - pad

        tree_h = int((body_bottom - body_top) * 0.6) - pad // 2
        tree = pygame.Rect(pad, body_top, left_w, tree_h)
        wfpng = pygame.Rect(pad, tree.bottom + pad, left_w,
                            body_bottom - tree.bottom - pad)

        right_h = body_bottom - body_top
        rubric_h = max(232, int(right_h * 0.24))
        timelapse_h = right_h - rubric_h - pad
        timelapse = pygame.Rect(right_x, body_top, right_w, timelapse_h)
        rubric = pygame.Rect(right_x, timelapse.bottom + pad, right_w,
                             body_bottom - timelapse.bottom - pad)

        return {
            "header": header, "timeline": timeline, "tree": tree,
            "wfpng": wfpng, "timelapse": timelapse, "rubric": rubric,
        }

    def _draw(self):
        self.screen.fill(BG)
        rects = self._layout()
        wf = self.current_workflow
        w, h = self.screen.get_size()
        self._draw_background(w, h)

        # header
        draw_header(self.screen, rects["header"], self.fonts,
                    wf, self.workflows,
                    self.step_index_in_wf(),
                    len(wf.steps), self.playing, self.speed)

        # tree panel
        inner = draw_panel(self.screen, rects["tree"], "LINEAGE TREE",
                           self.fonts["header"])
        self._tree_nodes = draw_tree(self.screen, inner, self.fonts,
                                     self.workflows, wf.uuid, self.anim_t)
        if self.transition is not None:
            self._draw_transition_overlay(self._tree_nodes)

        # workflow png
        inner = draw_panel(self.screen, rects["wfpng"],
                           "WORKFLOW GRAPH", self.fonts["header"], INFO)
        has_fade_png = self.transition is not None and (
            wf.png_path is not None
            or self.workflows[self.transition.from_wi].png_path is not None
        )
        if has_fade_png and self.transition.from_wi != \
                self.frames[self.cur][0]:
            self._draw_png_crossfade(inner, wf, self.transition)
        else:
            draw_workflow_png(self.screen, inner, self.fonts, wf,
                              self.anim_t)

        # memory timelapse — text types itself out while playing
        inner = draw_panel(self.screen, rects["timelapse"],
                           "MEMORY TIMELAPSE", self.fonts["header"], AMBER)
        reveal = (min(1.0, self._acc * TYPE_REVEAL_BOOST)
                  if self.playing else 1.0)
        draw_step_panel(self.screen, inner, self.fonts,
                        wf, self.current_step, self.step_index_in_wf(),
                        self.current_sub_index(), reveal, self.anim_t)

        # rubric
        inner = draw_panel(self.screen, rects["rubric"],
                           "RUBRIC EVOLUTION", self.fonts["header"], SUCCESS)
        completed_mask = [
            last >= 0 and self.cur >= last
            for last in self._last_frame_of_wf
        ]
        # flash only during playback — a parked playhead (pause, end of
        # recording hold) must show true cell colors, not a frozen flash
        col_flash = [
            max(0.0, 1.0 - (self.cur - last) / RUBRIC_FLASH_FRAMES)
            if self.playing and last >= 0 and self.cur >= last else 0.0
            for last in self._last_frame_of_wf
        ]
        draw_rubric(self.screen, inner, self.fonts, self.workflows,
                    self.frames[self.cur][0], completed_mask,
                    col_flash, self.anim_t)

        # bottom timeline
        self._timeline_rect = rects["timeline"]
        draw_timeline(self.screen, rects["timeline"], self.fonts,
                      len(self.frames), self.cur,
                      self.workflows, self.frame_to_wf, self.anim_t)

        self.screen.blit(self._get_vignette(w, h), (0, 0))
        if self.show_help:
            draw_help(self.screen, self.fonts)

    # ----- ambient background ---------------------------------------------
    def _draw_background(self, w: int, h: int):
        """Grid + drifting starfield + slow scanline sweep."""
        for x in range(0, w, 48):
            pygame.draw.line(self.screen, (12, 16, 24), (x, 0), (x, h), 1)
        for y in range(0, h, 48):
            pygame.draw.line(self.screen, (12, 16, 24), (0, y), (w, y), 1)
        for sx, sy, spd, size, phase in self._stars:
            y = ((sy + self.anim_t * spd) % 1.0) * h
            twinkle = 0.55 + 0.45 * math.sin(self.anim_t * 1.7 + phase)
            c = int(45 + 65 * twinkle)
            color = (int(c * 0.55), int(c * 0.75), c)
            pygame.draw.circle(self.screen, color, (int(sx * w), int(y)),
                               size)
        band = self._get_scan_band(w)
        span = h + band.get_height()
        scan_y = ((self.anim_t / SCAN_PERIOD_S) % 1.0) * span
        self.screen.blit(band, (0, int(scan_y) - band.get_height()))

    def _get_scan_band(self, w: int) -> pygame.Surface:
        """Cached translucent horizontal band for the scanline sweep."""
        if self._scan_band is not None and \
                self._scan_band.get_width() == w:
            return self._scan_band
        band_h = 44
        band = pygame.Surface((w, band_h), pygame.SRCALPHA)
        for i in range(band_h):
            alpha = int(22 * (1 - abs(i - band_h / 2) / (band_h / 2)))
            pygame.draw.line(band, (*ACCENT, alpha), (0, i), (w, i))
        self._scan_band = band
        return band

    def _get_vignette(self, w: int, h: int) -> pygame.Surface:
        """Cached radial darkening — built small, smoothscaled up."""
        if self._vignette is not None and \
                self._vignette.get_size() == (w, h):
            return self._vignette
        sw, sh = 160, 100
        small = pygame.Surface((sw, sh), pygame.SRCALPHA)
        for y in range(sh):
            for x in range(sw):
                d = math.hypot((x - sw / 2) / (sw / 2),
                               (y - sh / 2) / (sh / 2))
                alpha = min(int(max(0.0, d - 0.62) * 150), 120)
                small.set_at((x, y), (0, 0, 0, alpha))
        self._vignette = pygame.transform.smoothscale(small, (w, h))
        return self._vignette

    # ----- transition rendering -----------------------------------------
    def _draw_png_crossfade(self, inner: pygame.Rect, new_wf: Workflow,
                            tr: Transition):
        """Fade the previous workflow's PNG out while the new one fades in."""
        ease = tr.ease
        old_wf = self.workflows[tr.from_wi]
        for w, alpha in ((old_wf, 1.0 - ease), (new_wf, ease)):
            if w.png_path is None:
                continue
            src = get_workflow_png(w.png_path)
            if src is None:
                continue
            fitted = fit_surface(src, inner.width - 12, inner.height - 12)
            fitted.set_alpha(int(255 * alpha))
            ox = inner.x + (inner.width - fitted.get_width()) // 2
            oy = inner.y + (inner.height - fitted.get_height()) // 2
            self.screen.blit(fitted, (ox, oy))

    def _draw_transition_overlay(self, tree_nodes: dict[str, TreeNode]):
        """Glowing pulse traveling parent → child, then a ring burst on arrival."""
        tr = self.transition
        if tr is None:
            return
        a = tree_nodes.get(tr.from_uuid)
        b = tree_nodes.get(tr.to_uuid)
        if a is None or b is None or a is b:
            return
        to_wf = next((w for w in self.workflows if w.uuid == tr.to_uuid), None)
        color = EVO_COLORS.get(to_wf.kind if to_wf else "mutation", ACCENT)
        ease = tr.ease
        px = a.x + (b.x - a.x) * ease
        py = a.y + (b.y - a.y) * ease
        # Bright trail from source up to current position.
        pygame.draw.line(self.screen, color, (a.x, a.y), (px, py), 2)
        # Layered additive glow around the pulse head.
        glow_circle(self.screen, color, (px, py), 4, layers=3, alpha=55)
        pygame.draw.circle(self.screen, color, (int(px), int(py)), 4)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(px), int(py)), 2)
        # Ring burst + radial sparks at destination in the final 40 %.
        if tr.t > 0.6:
            bt = (tr.t - 0.6) / 0.4
            rad = int(b.radius + 22 * bt)
            alpha = int(200 * (1 - bt))
            if alpha > 0:
                d = rad * 2 + 4
                ring = pygame.Surface((d, d), pygame.SRCALPHA)
                pygame.draw.circle(ring, (*color, alpha), (d // 2, d // 2),
                                   rad, 3)
                self.screen.blit(ring, (b.x - d // 2, b.y - d // 2))
            spark_r = max(1, int(3 * (1 - bt)))
            dist = b.radius + 6 + 30 * bt
            for i in range(BURST_PARTICLES):
                ang = i * 2 * math.pi / BURST_PARTICLES
                sx = int(b.x + math.cos(ang) * dist)
                sy = int(b.y + math.sin(ang) * dist)
                pygame.draw.circle(self.screen,
                                   mix_color(color, (255, 255, 255), 0.4),
                                   (sx, sy), spark_r)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Interactive Jarvis-style viewer of workflow evolution.")
    parser.add_argument("--workflows-dir", type=Path,
                        default=Path("sources/workflows"))
    parser.add_argument("--memory-dir", type=Path,
                        default=Path("sources/memory"))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--record", type=Path, default=None,
                        help="Render the full animation to this .mp4 path "
                             "instead of opening the GUI (requires ffmpeg).")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Playback speed in steps per second "
                             "(default 2.0). Applies in both GUI and record "
                             "modes.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video frame rate (record mode only, "
                             "default 30).")
    parser.add_argument("--hold-seconds", type=float, default=1.0,
                        help="How long to hold on the final frame at the "
                             "end of the recorded video (default 1.0).")
    args = parser.parse_args()

    if args.fps <= 0 or args.speed <= 0:
        sys.exit("--fps and --speed must be positive.")
    if not args.workflows_dir.exists():
        sys.exit(f"workflows dir not found: {args.workflows_dir}")
    workflows = load_all(args.workflows_dir, args.memory_dir)
    if not workflows:
        sys.exit("No workflows with lineage_<uuid>.json found.")
    print(f"Loaded {len(workflows)} workflows "
          f"({sum(len(w.steps) for w in workflows)} memory steps).")

    if args.record is not None:
        # Headless render: keep SDL from opening a real window.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        app = App(workflows, size=(args.width, args.height))
        app.speed = args.speed
        app.record(args.record, fps=args.fps, hold_seconds=args.hold_seconds)
    else:
        app = App(workflows, size=(args.width, args.height))
        app.speed = args.speed
        app.run()


# ---------------------------------------------------------------------------
# Smoke test — runs without launching the GUI when this file is imported
# or invoked with --smoke. Verifies the loaders parse real data.
# ---------------------------------------------------------------------------
def _smoke():
    # Pure-helper checks (no display needed).
    assert edge_curve_point(0, 0, 10, 10, 0.0) == (0.0, 0.0)
    assert edge_curve_point(0, 0, 10, 10, 1.0) == (10.0, 10.0)
    assert len(edge_curve(0, 0, 10, 10, segments=18)) == 19
    assert mix_color((0, 0, 0), (255, 255, 255), 0.5) == (127, 127, 127)
    full, cursor = typewriter_slice(["hello", "world"], 1.0)
    assert full == ["hello", "world"] and cursor is None
    part, cursor = typewriter_slice(["hello", "world"], 0.5)
    assert len(part) <= 2 and cursor is not None

    here = Path(__file__).resolve().parent
    candidates = [here]
    parent = here
    for _ in range(5):
        parent = parent.parent
        candidates.append(parent)
    for root in candidates:
        wf_dir = root / "sources/workflows"
        if wf_dir.exists():
            wfs = load_all(wf_dir, root / "sources/memory")
            iters = [w.iteration for w in wfs]
            assert iters == sorted(iters), "smoke: workflows must be iter-sorted"
            parsed = sum(1 for w in wfs if w.evaluation is not None)
            print(f"smoke OK [{root}] — {len(wfs)} workflows, "
                  f"{parsed} with evaluation, "
                  f"{sum(len(w.steps) for w in wfs)} memory steps")
            return
    print("smoke skipped — no sources/workflows found in or above this dir")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
    else:
        main()
