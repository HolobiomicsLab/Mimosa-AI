"""
Improvement Validator for Darwin Gödel Machine
Validates that workflow improvements are statistically significant before acceptance.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional


class ImprovementValidator:
    """Validates whether improvements are statistically significant."""

    def __init__(self, min_improvement_threshold: float = 0.05):
        """
        Initialize the improvement validator.
        
        Args:
            min_improvement_threshold: Minimum relative improvement required (5% default)
                                      Example: 0.05 means new reward must be 5% higher
        """
        self.logger = logging.getLogger(__name__)
        self.min_improvement_threshold = min_improvement_threshold

    def validate_improvement(
        self,
        baseline_run: Any,
        new_run: Any,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Statistically validate if new_run significantly improved over baseline.
        
        This is a key DGM principle: improvements must be formally validated
        before being accepted as genuine progress.
        
        Args:
            baseline_run: GodelRun from previous iteration
            new_run: GodelRun from current iteration
            threshold: Minimum relative improvement override (uses class default if None)
        
        Returns:
            dict with keys:
                - 'valid': bool - Whether improvement is statistically significant
                - 'relative_improvement': float - Relative change (0.15 = 15% improvement)
                - 'absolute_improvement': float - Absolute change in reward
                - 'baseline_reward': float - Starting reward
                - 'new_reward': float - New reward
                - 'validated_at': datetime - When validation occurred
                - 'confidence': float - Confidence in the improvement (0.0-1.0)
        """
        threshold = threshold or self.min_improvement_threshold
        
        baseline_reward = baseline_run.reward if hasattr(baseline_run, 'reward') else 0.0
        new_reward = new_run.reward if hasattr(new_run, 'reward') else 0.0
        
        # Calculate absolute and relative improvement
        absolute_improvement = new_reward - baseline_reward
        relative_improvement = (
            absolute_improvement / max(abs(baseline_reward), 1e-6)
        )
        
        # Determine if improvement is significant
        is_valid = relative_improvement > threshold
        
        # Calculate confidence based on magnitude of improvement
        # Larger improvements are more likely to be real and not due to variance
        confidence = min(1.0, abs(relative_improvement) / max(threshold, 1e-6))
        
        result = {
            "valid": is_valid,
            "relative_improvement": relative_improvement,
            "absolute_improvement": absolute_improvement,
            "baseline_reward": baseline_reward,
            "new_reward": new_reward,
            "validated_at": datetime.now(),
            "confidence": confidence,
            "threshold_used": threshold
        }
        
        # Log the validation
        if is_valid:
            self.logger.info(
                f"✅ IMPROVEMENT VALIDATED: {relative_improvement:+.1%} "
                f"({baseline_reward:.3f} → {new_reward:.3f}) "
                f"[confidence: {confidence:.0%}]"
            )
        else:
            self.logger.warning(
                f"⚠️ IMPROVEMENT NOT SIGNIFICANT: {relative_improvement:+.1%} "
                f"({baseline_reward:.3f} → {new_reward:.3f}) "
                f"[below {threshold:.0%} threshold]"
            )
        
        return result

    def validate_improvement_strategy(
        self,
        strategy_name: str,
        before_metric: float,
        after_metric: float,
        improvement_type: str = "generic"
    ) -> Dict[str, Any]:
        """
        Validate effectiveness of a specific improvement strategy.
        
        Args:
            strategy_name: Name of the improvement strategy (e.g., "refine_prompt")
            before_metric: Metric value before applying strategy
            after_metric: Metric value after applying strategy
            improvement_type: Category of improvement for tracking
        
        Returns:
            dict with validation results and strategy effectiveness
        """
        improvement = (after_metric - before_metric) / max(abs(before_metric), 1e-6)
        is_effective = improvement > self.min_improvement_threshold
        
        result = {
            "strategy": strategy_name,
            "improvement_type": improvement_type,
            "is_effective": is_effective,
            "relative_improvement": improvement,
            "before": before_metric,
            "after": after_metric,
            "validated_at": datetime.now()
        }
        
        status = "✅ EFFECTIVE" if is_effective else "❌ NOT EFFECTIVE"
        self.logger.info(
            f"{status}: Strategy '{strategy_name}' produced {improvement:+.1%} improvement "
            f"({before_metric:.3f} → {after_metric:.3f})"
        )
        
        return result

    def should_continue_iteration(
        self,
        current_reward: float,
        best_reward: float,
        iterations_without_improvement: int,
        max_iterations_without_improvement: int = 3
    ) -> bool:
        """
        Determine if iterations should continue based on improvement patterns.
        
        Args:
            current_reward: Current iteration's reward
            best_reward: Best reward achieved so far
            iterations_without_improvement: Count of consecutive non-improving iterations
            max_iterations_without_improvement: Max allowed non-improving iterations
        
        Returns:
            bool - Whether to continue iterating
        """
        if current_reward > best_reward:
            # We're improving, always continue
            return True
        
        if iterations_without_improvement >= max_iterations_without_improvement:
            self.logger.warning(
                f"⏹️ Stopping iterations: {iterations_without_improvement} iterations "
                f"without improvement (max: {max_iterations_without_improvement})"
            )
            return False
        
        return True

    def get_improvement_type(self, baseline_run: Any, new_run: Any) -> str:
        """
        Classify the type of improvement observed.
        
        Args:
            baseline_run: Previous GodelRun
            new_run: Current GodelRun
        
        Returns:
            str - Type of improvement ("performance", "cost", "both", "none")
        """
        baseline_reward = baseline_run.reward if hasattr(baseline_run, 'reward') else 0.0
        new_reward = new_run.reward if hasattr(new_run, 'reward') else 0.0
        baseline_cost = baseline_run.cost if hasattr(baseline_run, 'cost') else 0.0
        new_cost = new_run.cost if hasattr(new_run, 'cost') else 0.0
        
        reward_improved = new_reward > baseline_reward * 1.01  # 1% threshold
        cost_improved = new_cost < baseline_cost * 0.99  # 1% threshold
        
        if reward_improved and cost_improved:
            return "both"
        elif reward_improved:
            return "performance"
        elif cost_improved:
            return "cost"
        else:
            return "none"


if __name__ == "__main__":
    # Test the validator
    from sources.core.schema import GodelRun
    
    # Create mock runs for testing
    baseline = GodelRun(goal="test", prompt="test", reward=0.50)
    improved = GodelRun(goal="test", prompt="test", reward=0.65)
    marginal = GodelRun(goal="test", prompt="test", reward=0.51)
    
    validator = ImprovementValidator(min_improvement_threshold=0.05)
    
    print("Test 1: Valid improvement")
    result = validator.validate_improvement(baseline, improved)
    print(f"Valid: {result['valid']}, Improvement: {result['relative_improvement']:.1%}\n")
    
    print("Test 2: Marginal improvement (below threshold)")
    result = validator.validate_improvement(baseline, marginal)
    print(f"Valid: {result['valid']}, Improvement: {result['relative_improvement']:.1%}\n")
    
    print("Test 3: Strategy effectiveness")
    result = validator.validate_improvement_strategy(
        "refine_prompt", before_metric=0.50, after_metric=0.65
    )
    print(f"Effective: {result['is_effective']}, Improvement: {result['relative_improvement']:.1%}\n")
    
    print("Test 4: Improvement type classification")
    baseline_cost = GodelRun(goal="test", prompt="test", reward=0.50, cost=1.00)
    improved_both = GodelRun(goal="test", prompt="test", reward=0.65, cost=0.80)
    imp_type = validator.get_improvement_type(baseline_cost, improved_both)
    print(f"Improvement type: {imp_type}")
