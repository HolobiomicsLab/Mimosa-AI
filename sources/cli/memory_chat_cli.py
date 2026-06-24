"""
Interactive Memory Chat CLI — RAG-based interrogation of a run's memory.

Loads every JSON file under ``sources/memory/<run_uuid>/``, splits each step
trace (or single LLM call) into searchable chunks, embeds them with a local
sentence-transformer, and answers user questions via the judge model after
retrieving the most relevant chunks by cosine similarity.

UI is a small curses interface (inspired by ``memory_explorer.py``):
conversation pane on top, code preview from the top retrieved step below,
help bar at the bottom. Press ``a`` to ask a new question, arrows to scroll.

Usage::

    python main.py --memory_cli                 # latest run
    python main.py --memory_cli --memory_uuid 20260604_133122_06d6d38f
"""

from __future__ import annotations

import curses
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import Config
from sources.core.llm_provider import LLMConfig, LLMProvider, extract_model_pattern
from sources.transparency.memory_trace import extract_code as _extract_code_from_step
from sources.transparency.memory_trace import trim as _trim


# ── Tunables (constants — no magic numbers) ────────────────────────────────
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
MAX_EMBED_CHARS = 2000
MAX_CONTEXT_CHARS_PER_CHUNK = 2500
MAX_CONTEXT_CHUNKS_FOR_ANSWER = TOP_K
ANSWER_MAX_TOKENS = 1024
QUERY_REWRITE_MAX_TOKENS = 128


# ── Data model ────────────────────────────────────────────────────────────
@dataclass
class MemoryChunk:
    """A retrieval-searchable slice of a memory file."""

    uuid: str
    file_name: str
    agent_name: str
    step_number: int | None
    code: str
    summary: str
    embed_text: str
    raw: dict[str, Any] = field(default_factory=dict)

    def header(self) -> str:
        """Compact one-line label used in the UI and as LLM context tag."""
        if self.step_number is not None:
            return f"{self.file_name} · step {self.step_number}"
        return f"{self.file_name}"


# ── Chunk extraction ──────────────────────────────────────────────────────
# Code/trim primitives live in sources.transparency.memory_trace so the ASTRA
# exporter and this RAG CLI never drift on "the code the agent ran" — imported
# above as ``_extract_code_from_step`` / ``_trim``.


def _summarize_step(step: dict[str, Any], code: str) -> str:
    """Compose a human-readable summary of one step (UI + LLM context)."""
    parts: list[str] = []
    out = step.get("model_output") or ""
    if out:
        parts.append(f"Output:\n{_trim(out, 800)}")
    if code:
        parts.append(f"Code:\n{_trim(code, 800)}")
    obs = step.get("observations") or ""
    if obs:
        parts.append(f"Observations:\n{_trim(str(obs), 500)}")
    act = step.get("action_output")
    if act:
        parts.append(f"Action output: {_trim(str(act), 400)}")
    err = step.get("error")
    if err:
        parts.append(f"Error: {_trim(str(err), 300)}")
    return "\n\n".join(parts)


def _build_step_chunk(uuid: str, file_name: str, agent: str,
                      step: dict[str, Any]) -> MemoryChunk:
    """Build a MemoryChunk from one smolagent step dict."""
    code = _extract_code_from_step(step)
    summary = _summarize_step(step, code)
    embed_text = _trim(f"[{agent}] {summary}", MAX_EMBED_CHARS)
    return MemoryChunk(
        uuid=uuid,
        file_name=file_name,
        agent_name=agent,
        step_number=step.get("step_number"),
        code=code,
        summary=summary,
        embed_text=embed_text,
        raw=step,
    )


def _build_call_chunk(uuid: str, file_name: str, agent: str,
                      data: dict[str, Any]) -> MemoryChunk:
    """Build a MemoryChunk from a single-LLM-call JSON (workflow_creator-style)."""
    response = data.get("response") or ""
    if not response:
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            response = choices[0].get("message", {}).get("content", "") or ""
    summary = f"LLM call response:\n{_trim(response, 1500)}"
    embed_text = _trim(f"[{agent}] {summary}", MAX_EMBED_CHARS)
    return MemoryChunk(
        uuid=uuid,
        file_name=file_name,
        agent_name=agent,
        step_number=None,
        code=response,
        summary=summary,
        embed_text=embed_text,
        raw=data,
    )


def _load_file_chunks(uuid: str, path: Path) -> list[MemoryChunk]:
    """Parse one JSON memory file and return its chunks."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    agent = path.stem
    if isinstance(data, list):
        return [
            _build_step_chunk(uuid, path.name, agent, step)
            for step in data if isinstance(step, dict)
        ]
    if isinstance(data, dict):
        return [_build_call_chunk(uuid, path.name, agent, data)]
    return []


def load_memory_chunks(memory_dir: Path) -> list[MemoryChunk]:
    """Load every JSON in *memory_dir* into a flat list of MemoryChunks."""
    uuid = memory_dir.name
    chunks: list[MemoryChunk] = []
    for json_path in sorted(memory_dir.glob("*.json")):
        chunks.extend(_load_file_chunks(uuid, json_path))
    return chunks


def _available_runs(base: Path) -> list[str]:
    """Return run-directory names under *base* that contain memory JSONs,
    sorted newest-first by mtime.
    """
    try:
        candidates = [
            p for p in base.iterdir()
            if p.is_dir() and any(p.glob("*.json"))
        ]
    except OSError:
        return []
    return [p.name for p in sorted(
        candidates, key=lambda p: p.stat().st_mtime, reverse=True,
    )]


def resolve_run_dir(config: Config, run_uuid: str | None) -> Path:
    """Return the memory directory for *run_uuid* or the latest run if None."""
    base = Path(config.memory_dir)
    if not base.is_dir():
        raise FileNotFoundError(
            f"Memory base directory missing: {base}\n"
            "Run Mimosa at least once to populate sources/memory/<run_uuid>/."
        )
    if run_uuid:
        target = base / run_uuid
        if not target.is_dir():
            available = _available_runs(base)
            hint = ""
            if available:
                preview = ", ".join(available[:5])
                more = f" (+{len(available) - 5} more)" if len(available) > 5 else ""
                hint = f"\nAvailable runs (newest first): {preview}{more}"
            else:
                hint = f"\nNo populated run directories found under {base}."
            raise FileNotFoundError(
                f"Memory run directory missing: {target}{hint}"
            )
        return target
    available = _available_runs(base)
    if not available:
        raise FileNotFoundError(
            f"No run subdirectories with memory JSONs found under {base}."
        )
    return base / available[0]


# ── Retrieval index ───────────────────────────────────────────────────────
class MemoryIndex:
    """Sentence-transformer cosine-similarity index over MemoryChunks."""

    def __init__(self, chunks: list[MemoryChunk],
                 model_name: str = DEFAULT_EMBED_MODEL) -> None:
        """Embed all *chunks* with *model_name* (downloaded on first use)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for memory chat. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self.chunks = chunks
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}': {exc}. "
                "Check your network connection (the model is downloaded "
                "on first use) and that the model name is correct."
            ) from exc
        texts = [c.embed_text for c in chunks]
        if texts:
            try:
                vectors = self.model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to embed {len(texts)} memory chunk(s): {exc}"
                ) from exc
            self.matrix = np.asarray(vectors, dtype=np.float32)
        else:
            self.matrix = np.zeros((0, 0), dtype=np.float32)

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[MemoryChunk, float]]:
        """Return the *top_k* chunks most similar (cosine) to *query*."""
        if self.matrix.size == 0:
            return []
        q_vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        q_vec = np.asarray(q_vec, dtype=np.float32)[0]
        scores = self.matrix @ q_vec
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]


# ── LLM glue ──────────────────────────────────────────────────────────────
_QUERY_REWRITE_SYSTEM = """\
You generate a single short search query (1 sentence, <=20 words) that will be
embedded and matched against multi-agent run-memory chunks. The query should
focus on the technical concepts/keywords in the user's question (model names,
file names, tools, errors). Return ONLY the query string, no quotes, no prefix.
"""

_ANSWER_SYSTEM = """\
You are answering questions about a completed multi-agent workflow run.

You will be given the user's question and a set of retrieved memory chunks
from the run (each chunk is one agent step or one LLM call). Use ONLY the
information present in those chunks. If the answer is not supported by the
chunks, say so plainly.

Cite the chunks you used by their header tag, e.g. "(task_builder.json · step 4)".
Keep the answer concise. If code is relevant, quote the smallest useful slice.
"""


def _build_llm(config: Config, max_tokens: int) -> LLMProvider:
    """Create an LLMProvider bound to ``config.judge_model``."""
    provider, model = extract_model_pattern(config.judge_model)
    llm_config = LLMConfig(
        model=model,
        provider=provider,
        temperature=0.0,
        reasoning_effort="low",
        max_tokens=max_tokens,
        openrouter_provider=config.openrouter_provider_for(config.judge_model),
        openrouter_quantizations=config.openrouter_quantizations_for(config.judge_model),
    )
    return LLMProvider(agent_name=None, memory_path=None,
                       system_msg=None, config=llm_config)


def _format_chunks_for_prompt(hits: list[tuple[MemoryChunk, float]]) -> str:
    """Render retrieved chunks as a bulleted prompt context section."""
    blocks: list[str] = []
    for chunk, score in hits[:MAX_CONTEXT_CHUNKS_FOR_ANSWER]:
        body = _trim(chunk.summary, MAX_CONTEXT_CHARS_PER_CHUNK)
        blocks.append(
            f"=== {chunk.header()}  (score={score:.3f}) ===\n{body}"
        )
    return "\n\n".join(blocks)


def rewrite_query(llm: LLMProvider, question: str) -> str:
    """Use the LLM to compress *question* into a retrieval-friendly query."""
    llm.sys_msg = _QUERY_REWRITE_SYSTEM
    try:
        out = llm(question, use_cache=False).strip()
    except Exception:
        return question
    return out or question


def answer_with_context(llm: LLMProvider, question: str,
                        hits: list[tuple[MemoryChunk, float]]) -> str:
    """Call the LLM with retrieved context and return its answer string."""
    llm.sys_msg = _ANSWER_SYSTEM
    context = _format_chunks_for_prompt(hits)
    user_msg = (
        f"# Question\n{question}\n\n"
        f"# Retrieved memory chunks\n{context}\n"
    )
    try:
        return llm(user_msg, use_cache=False).strip()
    except Exception as exc:
        return f"[LLM error: {exc}]"


# ── UI helpers ────────────────────────────────────────────────────────────
@dataclass
class _Turn:
    """One Q/A exchange + the chunks retrieved for it."""

    question: str
    answer: str
    hits: list[tuple[MemoryChunk, float]]


def _wrap_lines(text: str, width: int) -> list[str]:
    """Word-wrap *text* to *width*, preserving paragraph breaks."""
    if width <= 1:
        return [text]
    out: list[str] = []
    for raw in text.splitlines() or [""]:
        if not raw:
            out.append("")
            continue
        out.extend(textwrap.wrap(raw, width=width) or [""])
    return out


# ── Main CLI class ────────────────────────────────────────────────────────
class MemoryChatCLI:
    """Interactive RAG chat over the memory of one workflow run."""

    HELP = ("a:ask  ↑↓:scroll  ←→:prev/next answer  c:code-scroll  "
            "r:reload  q:quit")

    def __init__(self, config: Config, run_uuid: str | None = None) -> None:
        """Resolve the run directory but defer heavy loading to ``run()``."""
        self.config = config
        self.run_dir = resolve_run_dir(config, run_uuid)
        self.chunks: list[MemoryChunk] = []
        self.index: MemoryIndex | None = None
        self.llm: LLMProvider | None = None
        self.turns: list[_Turn] = []
        self.cursor = 0  # index into self.turns (which one we're viewing)
        self.scroll = 0
        self.code_scroll = 0
        self._status_message: str = ""  # transient banner shown on row 1

    # -- bootstrap -----------------------------------------------------
    def _bootstrap(self) -> None:
        """Load chunks, build embeddings, prepare LLM (called once)."""
        self.chunks = load_memory_chunks(self.run_dir)
        if not self.chunks:
            raise RuntimeError(f"No memory chunks found in {self.run_dir}")
        self.index = MemoryIndex(self.chunks)
        self.llm = _build_llm(self.config, ANSWER_MAX_TOKENS)

    def _ask(self, question: str) -> _Turn:
        """Run the full RAG cycle for one question and return the Turn."""
        assert self.index is not None and self.llm is not None
        rewrite_llm = _build_llm(self.config, QUERY_REWRITE_MAX_TOKENS)
        query = rewrite_query(rewrite_llm, question)
        hits = self.index.search(query, top_k=TOP_K)
        answer = answer_with_context(self.llm, question, hits)
        return _Turn(question=question, answer=answer, hits=hits)

    # -- rendering -----------------------------------------------------
    def _current_turn(self) -> _Turn | None:
        if not self.turns:
            return None
        self.cursor = max(0, min(self.cursor, len(self.turns) - 1))
        return self.turns[self.cursor]

    def _turn_lines(self, turn: _Turn, width: int) -> list[str]:
        """Render the visible answer pane for *turn* at *width* columns."""
        lines = [f"❓ Q: {turn.question}", ""]
        lines += _wrap_lines(turn.answer, width)
        lines.append("")
        lines.append("─ Top retrieved chunks ─")
        for chunk, score in turn.hits:
            lines.append(f"  · {chunk.header()}   score={score:.3f}")
        return lines

    def _code_lines(self, turn: _Turn, width: int) -> list[str]:
        """Render the code pane (executed code of the #1 retrieved chunk)."""
        if not turn.hits:
            return ["(no retrieved chunk)"]
        top, _ = turn.hits[0]
        header = f"📎 Source: {top.header()}"
        if not top.code:
            return [header, "", "(no executed code on this chunk)"]
        body = _wrap_lines(top.code, width)
        return [header, ""] + body

    def _draw_header(self, stdscr, width: int) -> None:
        idx = self.cursor + 1 if self.turns else 0
        total = len(self.turns)
        title = (f"📊 Memory Chat — run {self.run_dir.name}   "
                 f"Q{idx}/{total}   chunks={len(self.chunks)}")
        try:
            stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(0, 0, title.ljust(width)[:width])
            stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
        except curses.error:
            pass
        # Transient status banner (one line, cleared after the next keypress).
        if self._status_message:
            try:
                stdscr.attron(curses.color_pair(3))
                stdscr.addstr(1, 0, self._status_message.ljust(width)[:width])
                stdscr.attroff(curses.color_pair(3))
            except curses.error:
                pass

    def _draw_help(self, stdscr, h: int, w: int) -> None:
        try:
            stdscr.attron(curses.color_pair(6))
            stdscr.addstr(h - 1, 0, self.HELP.ljust(w)[:w])
            stdscr.attroff(curses.color_pair(6))
        except curses.error:
            pass

    def _draw_pane(self, stdscr, lines: list[str], y0: int,
                   height: int, width: int, scroll: int) -> None:
        for i in range(height):
            idx = scroll + i
            if idx >= len(lines):
                break
            try:
                stdscr.addstr(y0 + i, 0, lines[idx][:width])
            except curses.error:
                pass

    def _render(self, stdscr) -> None:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        self._draw_header(stdscr, w)

        turn = self._current_turn()
        if turn is None:
            msg = ("No questions yet — press 'a' to ask one. "
                   f"Loaded {len(self.chunks)} chunks from {self.run_dir.name}.")
            self._draw_pane(stdscr, _wrap_lines(msg, w), 2, h - 3, w, 0)
            self._draw_help(stdscr, h, w)
            stdscr.refresh()
            return

        # Split available rows: ~60% answer, ~40% code
        usable = h - 3
        answer_h = max(5, int(usable * 0.6))
        code_h = max(3, usable - answer_h - 1)

        ans_lines = self._turn_lines(turn, w)
        code_lines = self._code_lines(turn, w)

        self._draw_pane(stdscr, ans_lines, 2, answer_h, w, self.scroll)
        try:
            stdscr.attron(curses.color_pair(3))
            stdscr.addstr(2 + answer_h, 0, ("─" * w)[:w])
            stdscr.attroff(curses.color_pair(3))
        except curses.error:
            pass
        self._draw_pane(stdscr, code_lines, 3 + answer_h, code_h, w,
                        self.code_scroll)

        self._draw_help(stdscr, h, w)
        stdscr.refresh()

    # -- input handling ------------------------------------------------
    def _prompt_question(self, stdscr) -> str:
        """Suspend curses, read a question from stdin, then restore."""
        curses.endwin()
        print("\n  Ask a question about this run (empty = cancel):")
        try:
            text = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            text = ""
        # curses resumes automatically when next refresh() is called
        stdscr.clear()
        return text

    def _handle_ask(self, stdscr) -> None:
        question = self._prompt_question(stdscr)
        if not question:
            return
        # Show a busy banner so the user knows we're querying
        stdscr.clear()
        try:
            stdscr.addstr(0, 0, "  Thinking… (retrieving + querying judge model)")
            stdscr.refresh()
        except curses.error:
            pass
        try:
            turn = self._ask(question)
        except Exception as exc:
            # Never let an LLM/retrieval error crash the curses UI — surface
            # the error inside a Turn so the user can keep browsing.
            turn = _Turn(
                question=question,
                answer=f"[Error while answering this question: {exc}]",
                hits=[],
            )
            self._status_message = f"⚠ ask failed: {exc}"
        self.turns.append(turn)
        self.cursor = len(self.turns) - 1
        self.scroll = 0
        self.code_scroll = 0

    def _handle_reload(self) -> None:
        """Reload memory chunks from disk and rebuild the embedding index."""
        try:
            reloaded = load_memory_chunks(self.run_dir)
        except Exception as exc:
            self._status_message = f"⚠ reload failed: {exc}"
            return
        if not reloaded:
            self._status_message = (
                f"⚠ reload found no memory chunks in {self.run_dir.name} "
                "— keeping current index."
            )
            return
        try:
            new_index = MemoryIndex(reloaded)
        except Exception as exc:
            self._status_message = f"⚠ reload index rebuild failed: {exc}"
            return
        self.chunks = reloaded
        self.index = new_index
        self._status_message = f"✓ reloaded {len(self.chunks)} chunks."

    def _loop(self, stdscr) -> None:
        """Curses main loop — runs until the user quits."""
        curses.curs_set(0)
        while True:
            self._render(stdscr)
            key = stdscr.getch()
            # Clear transient status banner on the next keypress so it doesn't
            # linger forever.
            self._status_message = ""
            if key in (ord("q"), ord("Q")):
                return
            if key in (ord("a"), ord("A")):
                self._handle_ask(stdscr)
            elif key == curses.KEY_UP:
                self.scroll = max(0, self.scroll - 1)
            elif key == curses.KEY_DOWN:
                self.scroll += 1
            elif key == curses.KEY_LEFT:
                if self.cursor > 0:
                    self.cursor -= 1
                    self.scroll = self.code_scroll = 0
            elif key == curses.KEY_RIGHT:
                if self.cursor < len(self.turns) - 1:
                    self.cursor += 1
                    self.scroll = self.code_scroll = 0
            elif key == ord("c"):
                self.code_scroll += 5
            elif key == ord("C"):
                self.code_scroll = max(0, self.code_scroll - 5)
            elif key in (ord("r"), ord("R")):
                self._handle_reload()

    # -- public entry --------------------------------------------------
    def run(self) -> None:
        """Bootstrap the index/LLM and launch the curses UI."""
        print(f"  Loading memory from {self.run_dir} …")
        try:
            self._bootstrap()
        except RuntimeError as exc:
            # Surfaced from MemoryIndex (embedding model load/encoding) or
            # missing chunks. Print a clean message and bail without curses.
            print(f"\n  ❌  Could not initialise memory chat: {exc}\n")
            return
        print(f"  Loaded {len(self.chunks)} chunks. Embedding ready.")
        print("  Starting interactive UI…")

        def _wrapped(stdscr: "curses._CursesWindow") -> None:
            curses.start_color()
            curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)
            self._loop(stdscr)

        try:
            curses.wrapper(_wrapped)
        except KeyboardInterrupt:
            pass
        print("\n  Goodbye!\n")


# ── Smoke check ───────────────────────────────────────────────────────────
def _smoke() -> None:
    """Tiny offline sanity check: load chunks + run retrieval on synthetic data.

    Avoids hitting the LLM or downloading the embedding model — just exercises
    the chunk extraction + summarization helpers.
    """
    fake_step = {
        "step_number": 7,
        "model_output": "Plan: train a MultitaskClassifier on ClinTox",
        "code_action": "import deepchem as dc\nmodel = dc.models.MultitaskClassifier(...)",
        "observations": "Model trained. accuracy=0.81",
        "action_output": {"status": "ok"},
        "error": None,
        "tool_calls": [],
    }
    chunk = _build_step_chunk("fake_uuid", "task_builder.json",
                              "task_builder", fake_step)
    assert chunk.step_number == 7
    assert "MultitaskClassifier" in chunk.code
    assert "MultitaskClassifier" in chunk.summary
    assert chunk.header() == "task_builder.json · step 7"

    fake_call = {
        "response": "WORKFLOW_CODE_HERE",
        "choices": [{"message": {"content": "fallback"}}],
    }
    call_chunk = _build_call_chunk("fake_uuid", "workflow_creator.json",
                                   "workflow_creator", fake_call)
    assert call_chunk.code.startswith("WORKFLOW_CODE_HERE")
    assert call_chunk.step_number is None
    print("memory_chat_cli smoke check: OK")


if __name__ == "__main__":
    _smoke()
