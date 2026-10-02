# Contributing

Thank you for your interest in contributing. This page summarizes the
contribution workflow; for code-level architecture see
[Developer guide](../DEVELOPER_GUIDE.md).

## Licensing & CLA

The repository is publicly distributed under the **Apache License 2.0**.

Non-trivial external contributions require a signed short
[Individual Contributor Agreement](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/CLA/INDIVIDUAL_CLA.md)
before they can be merged. If a contribution is made in the course of
employment or under institutional intellectual-property rules, maintainers
may also request the optional
[Employer / Institutional Authorization](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/CLA/EMPLOYER_AUTHORIZATION.md).

The CLA Assistant Lite workflow prompts contributors for the individual
agreement directly in the PR thread. The `CLA Check` status check must
pass before merge. See [CLA process](../cla-process.md) for details.

Contact for contribution-governance questions: `dr20.spv@listes.cnrs.fr`.

## Before submitting a PR

1. ✅ `pytest tests/` passes locally.
2. ✅ Code follows existing style — ruff config is in `pyproject.toml`.
3. ✅ Add docstrings on public functions.
4. ✅ Update the relevant doc page(s):
   - User-facing change → corresponding page under `docs/`.
   - Architecture change → also update the `.mermaid` source under
     `docs/diagrams/` and re-render the `.png` (see below).
   - Evolution-layer change → update [`docs/DEVELOPER_GUIDE.md`](../DEVELOPER_GUIDE.md).

## PR description

Please include:

- **Problem** — what does this solve?
- **Solution** — how does it solve it?
- **Testing** — how was it tested? (pytest output, benchmark numbers).
- **Backwards compatibility** — any breaking changes? Any config field renamed?

## Running tests

```bash
# All tests
uv run pytest tests/

# Specific test
uv run pytest tests/evaluator_test.py

# Verbose with coverage
uv run pytest tests/ -v --cov=sources
```

## Code style

- Ruff config in `pyproject.toml` (line length 88, isort, flake8-bugbear,
  pyupgrade, flake8-simplify enabled).
- Google-style docstrings on public functions.
- No imports inside functions unless avoiding a circular dep.

## Re-rendering diagrams

If you change a `.mermaid` source:

```bash
cd docs
npx -y -p @mermaid-js/mermaid-cli mmdc \
  -i diagrams/<name>.mermaid -o images/<name>.png \
  -t neutral -b white -w 1800
```

Commit the updated PNG alongside the mermaid change.

## Building the docs site

```bash
uvx --with mkdocs-material mkdocs serve     # live preview at :8000
uvx --with mkdocs-material mkdocs build     # static HTML to ./site
```

The site uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).
Config is in [`mkdocs.yml`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/mkdocs.yml).

## Questions & support

Open an Issue on
[GitHub](https://github.com/HolobiomicsLab/Mimosa-AI/issues) for any
question or feature request.
