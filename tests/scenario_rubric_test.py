#!/usr/bin/env python3
"""
Test script to verify the new rubric format works correctly with the evaluator.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.evaluation.scenario_loader import ScenarioLoader
from sources.evaluation.evaluator import ScenarioEvaluator
from config import Config
import dotenv

def test_scenario_loader():
    """Test that ScenarioLoader can load both formats."""
    print("=" * 60)
    print("Testing ScenarioLoader with new rubric format")
    print("=" * 60)
    
    loader = ScenarioLoader()
    
    # Test loading the new rubric format (feature_detection.json)
    print("\n1. Loading feature_detection.json (new rubric format)...")
    scenario = loader.load_scenario("feature_detection")
    
    if scenario:
        print("✓ Successfully loaded feature_detection scenario")
        print(f"  - Total points: {scenario.get('total_points', 'N/A')}")
        
        # Count items in each category
        categories = [
            "data_loading",
            "data_processing",
            "modeling_or_analysis_or_visualization",
            "output_formatting",
            "output_saving"
        ]
        
        total_items = 0
        for cat in categories:
            if cat in scenario and isinstance(scenario[cat], list):
                count = len(scenario[cat])
                if count > 0:
                    print(f"  - {cat}: {count} items")
                    total_items += count
        
        print(f"  - Total rubric items: {total_items}")
        return True
    else:
        print("✗ Failed to load feature_detection scenario")
        return False

def test_scenario_format_detection():
    """Test that the evaluator can detect format type."""
    print("\n" + "=" * 60)
    print("Testing format detection")
    print("=" * 60)
    
    loader = ScenarioLoader()
    
    # Load new format
    print("\n2. Testing format detection...")
    scenario = loader.load_scenario("feature_detection")
    
    if scenario:
        is_rubric = "total_points" in scenario
        print(f"✓ feature_detection.json detected as: {'Rubric format' if is_rubric else 'Legacy format'}")
        
        if is_rubric:
            print("  Format detection is working correctly!")
            return True
        else:
            print("  ✗ Format detection failed - should be rubric format")
            return False
    else:
        print("✗ Could not load scenario")
        return False

def test_validation():
    """Test that validation works for the rubric format."""
    print("\n" + "=" * 60)
    print("Testing rubric format validation")
    print("=" * 60)
    
    loader = ScenarioLoader()
    
    print("\n3. Testing validation of feature_detection.json...")
    scenario = loader.load_scenario("feature_detection")
    
    if scenario:
        print("✓ Validation passed for feature_detection.json")
        
        # Verify total_points matches sum
        total = scenario.get("total_points", 0)
        calculated = 0
        
        categories = [
            "data_loading",
            "data_processing",
            "modeling_or_analysis_or_visualization",
            "output_formatting",
            "output_saving"
        ]
        
        for cat in categories:
            if cat in scenario:
                for item in scenario[cat]:
                    calculated += item.get("points", 0)
        
        print(f"  - Declared total_points: {total}")
        print(f"  - Calculated sum: {calculated}")
        
        if abs(total - calculated) < 0.01:
            print("  ✓ Total points match!")
            return True
        else:
            print("  ✗ Total points mismatch!")
            return False
    else:
        print("✗ Validation failed")
        return False

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "SCENARIO RUBRIC FORMAT TEST SUITE" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = []
    
    # Run tests
    results.append(("ScenarioLoader", test_scenario_loader()))
    results.append(("Format Detection", test_scenario_format_detection()))
    results.append(("Validation", test_validation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
