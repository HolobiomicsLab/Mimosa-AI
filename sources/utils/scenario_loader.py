#!/usr/bin/env python3
"""
Scenario loader for managing evaluation test scenarios.
Loads JSON scenario definitions with assertions and test configurations.
"""

import json
from pathlib import Path
from typing import Any


class ScenarioLoader:
    """Manages loading and validation of evaluation scenarios."""

    def __init__(self, scenarios_dir: str = "sources/evaluation/scenarios"):
        self.scenarios_dir = Path(scenarios_dir)
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        self._scenario_cache = {}

    def load_scenario(self, scenario_id: str) -> dict[str, Any] | None:
        """
        Load a scenario definition by ID.

        Args:
            scenario_id: Unique identifier for the scenario

        Returns:
            Scenario dictionary or None if not found
        """
        if scenario_id in self._scenario_cache:
            return self._scenario_cache[scenario_id]

        scenario_file = self.scenarios_dir / f"{scenario_id}.json"

        if not scenario_file.exists():
            print(f"Scenario file not found: {scenario_file}")
            return None

        try:
            with open(scenario_file) as f:
                scenario = json.load(f)

            # Validate scenario structure
            if self._validate_scenario(scenario):
                self._scenario_cache[scenario_id] = scenario
                return scenario
            else:
                print(f"Invalid scenario structure: {scenario_id}")
                return None

        except json.JSONDecodeError as e:
            print(f"Error parsing scenario {scenario_id}: {e}")
            return None
        except Exception as e:
            print(f"Error loading scenario {scenario_id}: {e}")
            return None

    def _validate_scenario(self, scenario: dict[str, Any]) -> bool:
        """
        Validate scenario structure and required fields.

        Args:
            scenario: Scenario dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["id", "goal", "assertions"]

        # Check required top-level fields
        for field in required_fields:
            if field not in scenario:
                print(f"Missing required field: {field}")
                return False

        # Validate assertions
        if not self._validate_assertions(scenario["assertions"]):
            return False

        # Validate optional section if present
        return not ("optional" in scenario and not self._validate_optional_config(scenario["optional"]))
    
    def _validate_assertions(self, assertions: list[dict]) -> bool:
        """
        Validate assertion structure.

        Args:
            assertions: List of assertion dictionaries

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(assertions, list):
            print("assertions must be a list")
            return False

        if not assertions:  # Empty list is valid
            return True

        required_assertion_fields = ["id", "description", "evaluation_criteria"]

        for i, assertion in enumerate(assertions):
            if not isinstance(assertion, dict):
                print(f"assertion {i} must be a dictionary")
                return False

            for field in required_assertion_fields:
                if field not in assertion:
                    print(f"assertion {i} missing field: {field}")
                    return False

            # ID should be unique
            assertion_ids = [a["id"] for a in assertions]
            if len(set(assertion_ids)) != len(assertion_ids):
                print("Duplicate assertion IDs found")
                return False

        return True

    def _validate_optional_config(self, config: dict[str, Any]) -> bool:
        """
        Validate optional configuration.

        Args:
            config: Optional configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(config, dict):
            print("optional must be a dictionary")
            return False

        # Optional fields with type validation
        optional_fields = {"required_tools": list, "judge_model": str}

        for field, expected_types in optional_fields.items():
            if field in config and not isinstance(config[field], expected_types):
                print(f"optional.{field} has invalid type")
                return False

        return True
