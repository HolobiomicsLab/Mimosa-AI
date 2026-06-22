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
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    claims: List[ClaimResult]


@dataclass
class Workflow:
    uuid: str
    iteration: int
    kind: str                # seed / mutation / crossover
    parents: List[str]
    steps: List[StepInfo] = field(default_factory=list)
    evaluation: Optional[Evaluation] = None
    png_path: Optional[Path] = None


def _parse_evaluation(path: Path) -> Optional[Evaluation]:
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
    claims: List[ClaimResult] = []
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


def _extract_previews(entry: Dict) -> Tuple[str, str, str]:
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


def _load_memory_steps(mem_dir: Path) -> List[StepInfo]:
    """Load every agent trace under ``mem_dir`` into a flat StepInfo list.

    Handles both layouts seen in this repo:
      * nested ``mem_dir/stage*/<agent>.json`` (the original timelapse layout)
      * flat ``mem_dir/<agent>.json`` (the current evolutionary runs)
    """
    if not mem_dir.exists():
        return []
    steps: List[StepInfo] = []
    raw_pool: List[Tuple[Dict, StepInfo]] = []

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


def _load_workflow(wf_dir: Path, memory_root: Path) -> Optional[Workflow]:
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


def load_all(workflows_dir: Path, memory_dir: Path) -> List[Workflow]:
    """Discover every workflow with a lineage file and load it."""
    out: List[Workflow] = []
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
    color: Tuple[int, int, int]


def layout_tree(
    workflows: List[Workflow],
    rect: pygame.Rect,
    pad: int = 26,
) -> Tuple[Dict[str, TreeNode], List[Tuple[str, str]]]:
    """Place each workflow on a horizontal band keyed by iteration."""
    nodes: Dict[str, TreeNode] = {}
    if not workflows:
        return nodes, []

    by_iter: Dict[int, List[Workflow]] = {}
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

    edges: List[Tuple[str, str]] = []
    for wf in workflows:
        for p in wf.parents:
            if p in nodes:
                edges.append((p, wf.uuid))
    return nodes, edges


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def draw_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    font: pygame.font.Font,
    accent: Tuple[int, int, int] = ACCENT,
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
    surf.blit(label, (rect.x + 26, rect.y + 7))
    return pygame.Rect(rect.x + 12, rect.y + 36, rect.width - 24,
                       rect.height - 46)


def draw_chevron(surf: pygame.Surface, x: int, y: int, color, size: int = 10):
    pts = [(x, y), (x + size, y + size // 2), (x, y + size)]
    pygame.draw.polygon(surf, color, pts)


def text_lines(
    surf: pygame.Surface, font: pygame.font.Font, lines: List[str],
    pos: Tuple[int, int], color, line_h: int, max_lines: Optional[int] = None
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
    if sw == 0 or sh == 0:
        return src
    scale = min(max_w / sw, max_h / sh)
    return pygame.transform.smoothscale(src, (int(sw * scale), int(sh * scale)))


def wrap_text(text: str, font: pygame.font.Font, max_w: int) -> List[str]:
    out: List[str] = []
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
def draw_header(surf, rect, fonts, wf: Workflow, total_wf: int,
                step_idx: int, n_steps: int, playing: bool, speed: float):
    pygame.draw.rect(surf, BG_PANEL, rect)
    pygame.draw.line(surf, BORDER_BRIGHT, (rect.x, rect.bottom - 1),
                     (rect.right, rect.bottom - 1), 1)

    title = fonts["title"].render("MIMOSA-AI", True, ACCENT)
    sub = fonts["body"].render("WORKFLOW EVOLUTION", True, TEXT)
    surf.blit(title, (rect.x + 24, rect.y + 10))
    surf.blit(sub, (rect.x + 24 + title.get_width() + 14, rect.y + 16))

    meta = (
        f"uuid {wf.uuid}    iter {wf.iteration + 1}/{total_wf}    "
        f"kind {wf.kind}    step {step_idx + 1}/{max(n_steps, 1)}"
    )
    surf.blit(fonts["small"].render(meta, True, TEXT_DIM),
              (rect.x + 24, rect.y + 44))

    badge = " ▶ PLAY " if playing else " ⏸ PAUSED "
    color = SUCCESS if playing else AMBER
    text = fonts["body"].render(badge, True, color)
    bw = text.get_width() + 16
    bx = rect.right - bw - 24
    pygame.draw.rect(surf, BG_PANEL_LIGHT,
                     (bx, rect.y + 14, bw, 30), border_radius=6)
    pygame.draw.rect(surf, color, (bx, rect.y + 14, bw, 30), width=1,
                     border_radius=6)
    surf.blit(text, (bx + 8, rect.y + 18))

    speed_txt = fonts["small"].render(f"x{speed:.1f}", True, TEXT_DIM)
    surf.blit(speed_txt, (bx - speed_txt.get_width() - 14, rect.y + 22))


# ---------------------------------------------------------------------------
# Lineage tree panel
# ---------------------------------------------------------------------------
def draw_tree(
    surf, area: pygame.Rect, fonts,
    workflows: List[Workflow], current_uuid: str,
) -> Dict[str, TreeNode]:
    nodes, edges = layout_tree(workflows, area)
    by_uuid = {w.uuid: w for w in workflows}

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

    # edges
    for parent, child in edges:
        a, b = nodes[parent], nodes[child]
        color = EVO_COLORS.get(by_uuid[child].kind, ACCENT_DIM)
        pygame.draw.line(surf, color, (a.x, a.y), (b.x, b.y), 1)

    # nodes
    for n in nodes.values():
        wf = by_uuid[n.uuid]
        is_current = n.uuid == current_uuid
        score = wf.evaluation.overall if wf.evaluation else 0.0
        r = n.radius + (4 if is_current else 0)
        if is_current:
            for k in range(3, 0, -1):
                pygame.draw.circle(surf, (*ACCENT, 0), (int(n.x), int(n.y)),
                                   int(r + k * 3), 1)
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
        # iteration number inside
        lbl = fonts["tiny"].render(str(n.iteration), True, BG)
        surf.blit(lbl, (n.x - lbl.get_width() / 2, n.y - lbl.get_height() / 2))

    # legend
    lx = area.x + 12
    ly = area.bottom - 22
    for kind, color in EVO_COLORS.items():
        pygame.draw.circle(surf, color, (lx + 6, ly + 8), 5)
        t = fonts["tiny"].render(kind, True, TEXT_DIM)
        surf.blit(t, (lx + 16, ly + 2))
        lx += 18 + t.get_width() + 16
    return nodes


# ---------------------------------------------------------------------------
# Memory timelapse panel
# ---------------------------------------------------------------------------
def draw_step_panel(surf, area: pygame.Rect, fonts, wf: Workflow,
                    step: Optional[StepInfo], step_idx: int):
    inner = area
    line_h = 20

    if step is None:
        msg = fonts["body"].render("No memory trace for this workflow.",
                                   True, TEXT_DIM)
        surf.blit(msg, (inner.x + 12, inner.y + 12))
        return

    # Agent badge row
    stage_color = STAGE_COLORS.get(step.stage, ACCENT)
    badge_w = 240
    pygame.draw.rect(surf, BG_PANEL_LIGHT,
                     (inner.x, inner.y, badge_w, 38), border_radius=8)
    pygame.draw.rect(surf, stage_color,
                     (inner.x, inner.y, 4, 38), border_radius=2)
    surf.blit(fonts["body"].render(step.agent[:22], True, stage_color),
              (inner.x + 14, inner.y + 4))
    surf.blit(fonts["tiny"].render(step.stage, True, TEXT_DIM),
              (inner.x + 14, inner.y + 22))

    # right-of-badge stat strip
    stats = [
        ("step", f"#{step.step}"),
        ("dur", f"{step.duration:.2f}s"),
        ("tok", f"{step.input_tokens}/{step.output_tokens}"),
        ("tools", str(len(step.tool_calls)) if step.tool_calls else "-"),
        ("status", step.action_status or "—"),
    ]
    sx = inner.x + badge_w + 16
    for label, value in stats:
        v_surf = fonts["body"].render(value, True, TEXT)
        l_surf = fonts["tiny"].render(label.upper(), True, TEXT_FAINT)
        surf.blit(l_surf, (sx, inner.y + 2))
        surf.blit(v_surf, (sx, inner.y + 16))
        sx += max(v_surf.get_width(), l_surf.get_width()) + 26
        if sx > inner.right - 80:
            break

    # error pill
    if step.has_error:
        text = fonts["tiny"].render("ERROR", True, ERROR)
        pygame.draw.rect(surf, (40, 16, 22),
                         (inner.right - 84, inner.y + 8, 70, 22),
                         border_radius=4)
        pygame.draw.rect(surf, ERROR,
                         (inner.right - 84, inner.y + 8, 70, 22), width=1,
                         border_radius=4)
        surf.blit(text, (inner.right - 84 + 18, inner.y + 11))

    # Three sub-panels: thought / code / observation
    top = inner.y + 50
    h_total = inner.bottom - top
    thought_h = int(h_total * 0.26)
    obs_h = int(h_total * 0.18)
    code_h = h_total - thought_h - obs_h - 18

    _sub_panel(surf, fonts,
               pygame.Rect(inner.x, top, inner.width, thought_h),
               "▌ THOUGHT", ACCENT, step.thought_preview, line_h,
               TEXT, mono=False)
    _sub_panel(surf, fonts,
               pygame.Rect(inner.x, top + thought_h + 9, inner.width, code_h),
               "▌ CODE", SUCCESS, step.code_preview, line_h,
               (180, 220, 250), mono=True)
    _sub_panel(surf, fonts,
               pygame.Rect(inner.x, top + thought_h + code_h + 18,
                           inner.width, obs_h),
               "▌ OBSERVATION", AMBER, step.observation_preview, line_h,
               (200, 230, 200), mono=False)

    # step progress strip at the very bottom of the inner area
    n = max(len(wf.steps), 1)
    _mini_strip(surf, fonts, inner.x, inner.bottom - 6,
                inner.width, step_idx, n)


def _sub_panel(surf, fonts, rect, label, accent, body, line_h, body_color,
               mono):
    pygame.draw.rect(surf, BG_PANEL_LIGHT, rect, border_radius=6)
    pygame.draw.rect(surf, BORDER, rect, width=1, border_radius=6)
    surf.blit(fonts["body"].render(label, True, accent),
              (rect.x + 10, rect.y + 6))
    font = fonts["mono"] if mono else fonts["small"]
    if not body:
        surf.blit(font.render("—", True, TEXT_FAINT),
                  (rect.x + 14, rect.y + 32))
        return
    max_w = rect.width - 28
    if mono:
        lines = [ln[:200] for ln in body.split("\n")]
    else:
        lines = wrap_text(body, font, max_w)
    max_lines = max((rect.height - 36) // line_h, 1)
    text_lines(surf, font, lines, (rect.x + 14, rect.y + 32),
               body_color, line_h, max_lines)


def _mini_strip(surf, fonts, x, y, w, idx, n):
    pygame.draw.rect(surf, BG_PANEL_LIGHT, (x, y, w, 3), border_radius=2)
    fill = int(w * (idx + 1) / n)
    pygame.draw.rect(surf, ACCENT, (x, y, fill, 3), border_radius=2)


# ---------------------------------------------------------------------------
# Workflow PNG panel
# ---------------------------------------------------------------------------
_PNG_CACHE: Dict[str, pygame.Surface] = {}


def get_workflow_png(path: Path) -> Optional[pygame.Surface]:
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


def draw_workflow_png(surf, area, fonts, wf: Workflow):
    if wf.png_path is None:
        msg = fonts["small"].render("No workflow_<uuid>.png", True, TEXT_DIM)
        surf.blit(msg, (area.x + 8, area.y + 8))
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
def _claim_summary(workflows: List[Workflow]) -> List[Tuple[str, int]]:
    """Return list of (claim_name, max_importance) ordered first-seen."""
    order: List[str] = []
    imp: Dict[str, int] = {}
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


def draw_rubric(surf, area, fonts, workflows: List[Workflow],
                current_idx: int):
    """Heatmap of per-claim pass/fail per generation + overall-score line."""
    all_claims = _claim_summary(workflows)
    if not all_claims or not workflows:
        msg = fonts["small"].render("No evaluation data.", True, TEXT_DIM)
        surf.blit(msg, (area.x + 8, area.y + 8))
        return

    # Reserve room for the line chart at the bottom.
    line_panel_h = min(max(45, area.height // 5), 60)
    grid_top = area.y
    grid_bottom = area.bottom - line_panel_h
    grid_h = grid_bottom - grid_top

    # Pick the most-important claims that fit (≥12 px per row).
    label_font = fonts["tiny"]
    row_h_min = max(label_font.get_height(), 12)
    max_rows = max(1, grid_h // row_h_min)

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
    for j, wf in enumerate(workflows):
        if not wf.evaluation:
            continue
        by_name = {c.name: c for c in wf.evaluation.claims}
        for i, (name, _imp) in enumerate(claims):
            c = by_name.get(name)
            if c is None:
                color = (28, 36, 50)
            elif c.status == "pass":
                color = SUCCESS
            elif c.status == "fail":
                color = ERROR
            elif c.status == "error":
                color = WARN
            else:
                color = TEXT_FAINT
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
    pts: List[Tuple[int, int]] = []
    for j, wf in enumerate(workflows):
        s = wf.evaluation.overall if wf.evaluation else 0.0
        x = grid_x + int(j * cell_w + cell_w / 2)
        y = int(line_y1 - line_h * max(0.0, min(s, 1.0)))
        pts.append((x, y))
    if len(pts) >= 2:
        pygame.draw.lines(surf, ACCENT, False, pts, 2)
    for j, (x, y) in enumerate(pts):
        col = AMBER if j == current_idx else ACCENT
        pygame.draw.circle(surf, col, (x, y), 3)
    surf.blit(label_font.render("overall score 0 → 1", True, TEXT_FAINT),
              (area.x, line_y0 + 4))
    # rolling stats for the current generation
    wf = workflows[current_idx] if 0 <= current_idx < len(workflows) else None
    if wf and wf.evaluation:
        e = wf.evaluation
        info = (
            f"gen {wf.iteration:02d}  •  score {e.overall:.3f}  •  "
            f"pass {e.n_pass}  fail {e.n_fail}  err {e.n_error}"
        )
        t = fonts["small"].render(info, True, TEXT)
        surf.blit(t, (area.x, line_y1 - t.get_height() - 2))
    # claim count indicator
    total_current = len(current_names)
    total_all = len(all_claims)
    if total_current > 0:
        count_txt = (
            f"showing {len(claims)}/{total_current} current claims"
            f"  ({total_all} total)"
        )
    else:
        count_txt = f"showing {len(claims)}/{total_all} claims"
    count_surf = label_font.render(count_txt, True, TEXT_FAINT)
    # place at the right edge of the line-chart area
    cx = grid_x + grid_w - count_surf.get_width()
    cy = line_y1 - count_surf.get_height() - 2
    surf.blit(count_surf, (cx, cy))


# ---------------------------------------------------------------------------
# Bottom timeline / controls
# ---------------------------------------------------------------------------
def draw_timeline(surf, rect, fonts, total_frames, current_frame,
                  workflows: List[Workflow], frame_to_wf: List[int]):
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
    pygame.draw.circle(surf, ACCENT, (pos_x, bar.y + bar.height // 2), 8)
    pygame.draw.circle(surf, BG, (pos_x, bar.y + bar.height // 2), 4)

    hint = (
        "SPACE play/pause   ← → step   ↑ ↓ generation   "
        "[ ] speed   H help   click tree node to jump"
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
    def __init__(self, workflows: List[Workflow],
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

        # flatten to (workflow_idx, step_idx) frames; workflows with no
        # memory still get a single "intro" frame so they appear in the timeline
        self.frames: List[Tuple[int, int]] = []
        for wi, wf in enumerate(workflows):
            if not wf.steps:
                self.frames.append((wi, -1))
            else:
                for si in range(len(wf.steps)):
                    self.frames.append((wi, si))
        self.frame_to_wf = [wi for (wi, _) in self.frames]

        self.cur = 0
        self.playing = True
        self.speed = 2.0     # steps per second at speed 1
        self._acc = 0.0
        self.show_help = False
        self._tree_nodes: Dict[str, TreeNode] = {}
        self._timeline_rect: Optional[pygame.Rect] = None

    @staticmethod
    def _load_fonts() -> Dict[str, pygame.font.Font]:
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
        }

    # ----- state helpers -------------------------------------------------
    @property
    def current_workflow(self) -> Workflow:
        wi, _ = self.frames[self.cur]
        return self.workflows[wi]

    @property
    def current_step(self) -> Optional[StepInfo]:
        wi, si = self.frames[self.cur]
        wf = self.workflows[wi]
        return wf.steps[si] if si >= 0 and si < len(wf.steps) else None

    def step_index_in_wf(self) -> int:
        _, si = self.frames[self.cur]
        return max(si, 0)

    def jump_workflow(self, delta: int):
        wi = self.frames[self.cur][0]
        new_wi = max(0, min(len(self.workflows) - 1, wi + delta))
        for i, (w, _) in enumerate(self.frames):
            if w == new_wi:
                self.cur = i
                return

    def jump_to_workflow_uuid(self, uuid: str):
        for i, (wi, _) in enumerate(self.frames):
            if self.workflows[wi].uuid == uuid:
                self.cur = i
                return

    # ----- main loop -----------------------------------------------------
    def run(self):
        while True:
            dt = self.clock.tick(self.fps) / 1000.0
            if not self._handle_events():
                return
            if self.playing:
                self._acc += dt * self.speed
                while self._acc >= 1.0 and self.cur < len(self.frames) - 1:
                    self._acc -= 1.0
                    self.cur += 1
                if self.cur >= len(self.frames) - 1:
                    self.playing = False
            self._draw()
            pygame.display.flip()

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
                    self.cur = min(self.cur + 1, len(self.frames) - 1)
                elif ev.key == pygame.K_LEFT:
                    self.cur = max(self.cur - 1, 0)
                elif ev.key == pygame.K_DOWN:
                    self.jump_workflow(+1)
                elif ev.key == pygame.K_UP:
                    self.jump_workflow(-1)
                elif ev.key == pygame.K_HOME:
                    self.cur = 0
                elif ev.key == pygame.K_END:
                    self.cur = len(self.frames) - 1
                elif ev.key in (pygame.K_LEFTBRACKET,):
                    self.speed = max(0.25, self.speed / 1.4)
                elif ev.key in (pygame.K_RIGHTBRACKET,):
                    self.speed = min(16.0, self.speed * 1.4)
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
            self.cur = max(0, min(len(self.frames) - 1,
                                  int(t * len(self.frames))))
            return
        # tree node click — closest node within its radius
        for n in self._tree_nodes.values():
            if math.hypot(pos[0] - n.x, pos[1] - n.y) <= n.radius + 4:
                self.jump_to_workflow_uuid(n.uuid)
                return

    # ----- draw ----------------------------------------------------------
    def _layout(self) -> Dict[str, pygame.Rect]:
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

        timelapse_h = int((body_bottom - body_top) * 0.50) - pad // 2
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

        # subtle grid pattern over background
        w, h = self.screen.get_size()
        for x in range(0, w, 48):
            pygame.draw.line(self.screen, (12, 16, 24), (x, 0), (x, h), 1)
        for y in range(0, h, 48):
            pygame.draw.line(self.screen, (12, 16, 24), (0, y), (w, y), 1)

        # header
        draw_header(self.screen, rects["header"], self.fonts,
                    wf, len(self.workflows),
                    self.step_index_in_wf(),
                    len(wf.steps), self.playing, self.speed)

        # tree panel
        inner = draw_panel(self.screen, rects["tree"], "▌ LINEAGE TREE",
                           self.fonts["header"])
        self._tree_nodes = draw_tree(self.screen, inner, self.fonts,
                                     self.workflows, wf.uuid)

        # workflow png
        inner = draw_panel(self.screen, rects["wfpng"],
                           "▌ WORKFLOW GRAPH", self.fonts["header"], INFO)
        draw_workflow_png(self.screen, inner, self.fonts, wf)

        # memory timelapse
        inner = draw_panel(self.screen, rects["timelapse"],
                           "▌ MEMORY TIMELAPSE", self.fonts["header"], AMBER)
        draw_step_panel(self.screen, inner, self.fonts,
                        wf, self.current_step, self.step_index_in_wf())

        # rubric
        inner = draw_panel(self.screen, rects["rubric"],
                           "▌ RUBRIC EVOLUTION", self.fonts["header"], SUCCESS)
        draw_rubric(self.screen, inner, self.fonts, self.workflows,
                    self.frames[self.cur][0])

        # bottom timeline
        self._timeline_rect = rects["timeline"]
        draw_timeline(self.screen, rects["timeline"], self.fonts,
                      len(self.frames), self.cur,
                      self.workflows, self.frame_to_wf)

        if self.show_help:
            draw_help(self.screen, self.fonts)


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
    args = parser.parse_args()

    if not args.workflows_dir.exists():
        sys.exit(f"workflows dir not found: {args.workflows_dir}")
    workflows = load_all(args.workflows_dir, args.memory_dir)
    if not workflows:
        sys.exit("No workflows with lineage_<uuid>.json found.")
    print(f"Loaded {len(workflows)} workflows "
          f"({sum(len(w.steps) for w in workflows)} memory steps).")
    App(workflows, size=(args.width, args.height)).run()


# ---------------------------------------------------------------------------
# Smoke test — runs without launching the GUI when this file is imported
# or invoked with --smoke. Verifies the loaders parse real data.
# ---------------------------------------------------------------------------
def _smoke():
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
