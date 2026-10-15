#!/usr/bin/env python3
"""The ScienceAgentBench goal must state the real grading environment.

2026-10-14 campaign: agents pinned post-era versions (pandas==2.2.2,
deepchem==2.8.0, numpy==2.2.6) that the Python 3.10 grading sandbox cannot
install under PINNED_CONSTRAINTS. The goal now lists the sandbox's own Python
version and caps, so the two can never drift apart.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation.csv_mode import CsvEvaluationMode  # noqa: E402
from sources.benchmark_evaluation.execution_sandbox import (  # noqa: E402
    PINNED_CONSTRAINTS,
    SANDBOX_PYTHON_VERSION,
)

ROW = {
    "instance_id": "1",
    "task_inst": "Train a model and save predictions.",
    "output_fname": "pred_results/clintox_test_pred.csv",
    "gold_program_name": "clintox_nn.py",
}


def test_sab_goal_states_grading_python_and_every_version_cap():
    goal, scenario_id, _ = CsvEvaluationMode._generate_task_science_agent_bench(None, ROW)
    assert scenario_id == "1"
    assert f"Python {SANDBOX_PYTHON_VERSION}" in goal
    missing = [cap for cap in PINNED_CONSTRAINTS if cap not in goal]
    assert missing == []
    assert goal.index("EXECUTION ENVIRONMENT") < goal.index("EXPECTED OUTPUT")
