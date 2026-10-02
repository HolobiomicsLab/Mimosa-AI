"""Make the tests import the working tree, not an installed snapshot.

``pyproject.toml`` force-includes ``main.py`` and ``config.py`` as top-level
modules in the wheel, so a ``uv sync`` / ``pip install`` of this project drops
copies of both into ``site-packages``. Under pytest those copies won every
import: ``import config`` in ``tests/config_roundtrip_test.py`` resolved to
``.venv/lib/python3.11/site-packages/config.py`` — a snapshot frozen at install
time — while ``sources/`` (not shipped as a top-level module) correctly
resolved to the tree.

The result was a test suite that could pass green against code the repository
no longer contained: a config field added in the working tree was simply
invisible to the round-trip test that exists to check exactly that.

Putting the repo root at the FRONT of ``sys.path`` makes the tree win.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Front, not append: site-packages is already on the path and would otherwise
# keep winning for `main` and `config`.
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
