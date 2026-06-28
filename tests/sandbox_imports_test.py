"""Guard the agent sandbox's scientific-computing capability: scipy/sklearn must
stay authorized in both agent factories and installed in the runner env.

Protects `feat(sandbox): allow scientific-python imports` — a fair run showed
the agent cannot reproduce a data-analysis task (RandomForest, PCoA) without
these, so silently dropping them would re-break execution."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import Config

ROOT = Path(__file__).parent.parent

_FACTORIES = (
    "sources/modules/smolagent_factory.py",
    "sources/core/single_agent_factory.py",
)


def test_factories_authorize_scientific_imports():
    for rel in _FACTORIES:
        src = (ROOT / rel).read_text()
        assert "'sklearn'" in src, f"{rel} must authorize sklearn in the sandbox"
        assert "'scipy'" in src, f"{rel} must authorize scipy in the sandbox"


def test_runner_requirements_install_scientific_stack():
    reqs = " ".join(Config().runner_requirements).lower()
    assert "scikit-learn" in reqs, "runner env must install scikit-learn"
    assert "scipy" in reqs, "runner env must install scipy"
    # the data-handling base must remain too
    assert "pandas" in reqs and "numpy" in reqs


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
