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

    def __init__(self, scenarios_dir: str = "datasets/ScienceAgentBench/scoring_rubrics"):
        self.scenarios_dir = Path(scenarios_dir)
        self._scenario_cache = {}

    def load_scenario(self, scenario_rubric: str) -> dict[str, Any] | None:
        """
        Load a scenario definition by ID.
        Args:
            scenario_rubric: Unique identifier for the scenario
        Returns:
            Scenario dictionary or None if not found
        """
        if scenario_rubric in self._scenario_cache:
            return self._scenario_cache[scenario_rubric]

        scenario_file = self.scenarios_dir / f"{scenario_rubric}"
        if not scenario_file.exists():
            print(f"Scenario file not found: {scenario_file}")
            return None
        try:
            with open(scenario_file) as f:
                scenario = json.load(f)
            if self._validate_scenario(scenario):
                self._scenario_cache[scenario_rubric] = scenario
                return scenario
            else:
                print(f"Invalid scenario structure: {scenario_rubric}")
                return None
        except json.JSONDecodeError as e:
            print(f"Error parsing scenario {scenario_rubric}: {e}")
            return None
        except Exception as e:
            print(f"Error loading scenario {scenario_rubric}: {e}")
            return None

    def _validate_scenario(self, scenario: dict[str, Any]) -> bool:
        """
        Validate scenario structure and required fields.
        Supports ScienceAgentBench rubric format with categories.

        Args:
            scenario: Scenario dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        # Check if it's the new rubric format (has category keys and total_points)
        if "total_points" in scenario:
            return self._validate_rubric_format(scenario)

        # Legacy format validation (keeping for backwards compatibility)
        required_fields = ["id", "goal", "assertions"]

        for field in required_fields:
            if field not in scenario:
                print(f"Missing required field: {field}")
                return False

        if not self._validate_assertions(scenario["assertions"]):
            return False
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
        optional_fields = {"required_tools": list, "judge_model": str}
        for field, expected_types in optional_fields.items():
            if field in config and not isinstance(config[field], expected_types):
                print(f"optional.{field} has invalid type")
                return False

        return True

    def _validate_rubric_format(self, scenario: dict[str, Any]) -> bool:
        """
        Validate ScienceAgentBench rubric format.
        Args:
            scenario: Scenario dictionary to validate
        Returns:
            True if valid, False otherwise
        """
        if "total_points" not in scenario:
            print("Missing required field: total_points")
            return False

        # Validate total_points is a number
        if not isinstance(scenario["total_points"], (int, float)):
            print("total_points must be a number")
            return False

        # Standard ScienceAgentBench categories
        valid_categories = {
            "data_loading",
            "data_processing",
            "modeling_or_analysis_or_visualization",
            "output_formatting",
            "output_saving"
        }
        # Check that at least one category exists
        category_keys = [k for k in scenario.keys() if k in valid_categories]
        if not category_keys:
            print("No valid rubric categories found")
            return False
        # Validate each category
        total_calculated = 0
        for category_name in category_keys:
            category_items = scenario[category_name]

            if not isinstance(category_items, list):
                print(f"Category '{category_name}' must be a list")
                return False

            # Validate each item in the category
            for i, item in enumerate(category_items):
                if not isinstance(item, dict):
                    print(f"Item {i} in category '{category_name}' must be a dictionary")
                    return False

                required_item_fields = ["name", "description", "points"]
                for field in required_item_fields:
                    if field not in item:
                        print(f"Item {i} in category '{category_name}' missing field: {field}")
                        return False

                if not isinstance(item["points"], (int, float)):
                    print(f"Points must be a number in item {i} of category '{category_name}'")
                    return False

                total_calculated += item["points"]

        # Verify total_points matches sum of all item points
        if abs(total_calculated - scenario["total_points"]) > 0.01:
            print(f"Warning: total_points ({scenario['total_points']}) doesn't match sum of item points ({total_calculated})")

        return True
