# Installation

Mimosa-AI is a Python project. It needs a recent Python, a running Toolomics
MCP server, and at least one LLM API key.

## Prerequisites

- **Python 3.11–3.13** (3.14 is not yet supported).
- **[uv](https://github.com/astral-sh/uv)** (recommended) or `pip`.
- A running **[Toolomics MCP server](https://github.com/HolobiomicsLab/toolomics)**.
- At least one LLM API key — see [API keys & environment](env.md).

!!! info "Why Toolomics?"
    Toolomics is Mimosa's companion platform: it exposes scientific tools as
    discoverable MCP services and owns the shared workspace where Mimosa reads
    and writes task artefacts. Setup takes only a few minutes — see [Tool
    discovery & MCP](../concepts/tools-and-mcp.md) for the full story.

## 1. Install Mimosa-AI

=== "uv (recommended)"

    ```bash
    pip install uv
    git clone https://github.com/HolobiomicsLab/Mimosa-AI.git
    cd Mimosa-AI
    uv sync
    ```

    `uv sync` creates a virtualenv and installs everything from
    `pyproject.toml` in one step.

=== "pip"

    ```bash
    git clone https://github.com/HolobiomicsLab/Mimosa-AI.git
    cd Mimosa-AI
    python3 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install .
    ```

## 2. Set API keys

Create a `.env` file at the project root with at least one of:

```env
ANTHROPIC_API_KEY=...       # Claude — recommended for workflow orchestration
OPENAI_API_KEY=...          # OpenAI models
MISTRAL_API_KEY=...         # Mistral models
DEEPSEEK_API_KEY=...        # DeepSeek
HF_TOKEN=...                # HuggingFace
OPENROUTER_API_KEY=...      # Any model via OpenRouter

# Optional: observability via Langfuse
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_PRIVATE_KEY=...
```

Only the keys you actually use are required. See [API keys & environment](env.md)
for which provider fits which role.

## 3. Start the Toolomics MCP server

Follow the setup instructions at
[HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics).
Configure it to run on a port range — `5000–5100` by default.

You can add custom MCP tools via the Toolomics docs; Mimosa picks them up
automatically through discovery.

## 4. (Optional) Start Perspicacité for scientific grounding

[Perspicacité](https://github.com/HolobiomicsLab/Perspicacite-AI) is an
optional companion AI that grounds Mimosa's workflow synthesis and evaluation
in the literature. When it's running, Mimosa interacts with it automatically.

```bash
git clone https://github.com/HolobiomicsLab/Perspicacite-AI.git
cd Perspicacite-AI
uv sync
uv run web_app_full.py
```

That's it — Perspicacité starts and is ready to interact. Launch Mimosa in
another terminal and the two will find each other.

## 5. Verify the install

A quick sanity check that nothing is broken:

```bash
uv run python -c "from sources.core.evolution_engine import EvolutionEngine; print('ok')"
```

Then continue to the [Quickstart](quickstart.md).

!!! warning "Common pitfalls"
    - **`mcp not found`** — Toolomics isn't running. Start it before launching Mimosa.
    - **`ModuleNotFoundError: dotenv`** — you ran `python main.py` directly instead
      of `uv run main.py`. Either activate the venv or prefix with `uv run`.
    - **OpenRouter quantization errors** — some providers fail fidelity checks on
      escape sequences. See [Troubleshooting](../reference/troubleshooting.md).
