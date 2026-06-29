"""Tests for the optional cheap judge-extraction model tier.

Covers the config surface (default and round-trip). The routing of the
verifier's mechanical judge calls through this tier is exercised by the
verifier tests; this file has no import dependency on the evaluator stack."""

import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import Config


def test_judge_extraction_defaults_to_none():
    """Unset means reuse judge_model, so behaviour is unchanged by default."""
    assert Config().judge_extraction_model is None


def test_judge_extraction_round_trips(tmp_path):
    """A configured extraction model survives dump and load."""
    cheap = "openrouter/deepseek/deepseek-v4-flash"
    config = Config()
    config.judge_extraction_model = cheap
    path = tmp_path / "cfg.json"
    config.dump(str(path))

    loaded = Config()
    loaded.load(str(path))
    assert loaded.judge_extraction_model == cheap


if __name__ == "__main__":
    test_judge_extraction_defaults_to_none()
    with tempfile.TemporaryDirectory() as d:
        test_judge_extraction_round_trips(Path(d))
    print("judge_extraction tests passed")
