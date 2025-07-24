"""
Massive Testing Module for parallel DGM execution
"""

import asyncio
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any
import json
from pathlib import Path

from config import Config
from sources.core.dgm import GodelMachine


class ParallelTesting:
    """Class for running multiple DGM instances in parallel for mass testing."""
    
    def __init__(self, config: Config):
        """Initialize ParallelTesting with configuration.
        
        Args:
            config: Configuration object for the DGM system
        """
        self.config = config
        self.results_dir = Path("parallel_testing_results")
        self.results_dir.mkdir(exist_ok=True)
        
    @staticmethod
    def _run_single_dgm(goal_data: dict[str, Any]) -> dict[str, Any]:
        """Run a single DGM instance in a separate process.
        
        Args:
            goal_data: Dictionary containing goal information and parameters
            
        Returns:
            Dictionary with execution results
        """
        goal = goal_data["goal"]
        process_id = goal_data["process_id"]
        template_uuid = goal_data.get("template_uuid")
        judge = goal_data.get("judge", False)
        human_validation = goal_data.get("human_validation", False)
        config_data = goal_data["config_data"]
        
        print(f"🚀 Starting DGM process {process_id} for goal: {goal[:50]}...")
        
        start_time = time.time()
        result = {
            "process_id": process_id,
            "goal": goal,
            "template_uuid": template_uuid,
            "judge": judge,
            "human_validation": human_validation,
            "start_time": start_time,
            "status": "running",
            "error": None,
            "execution_time": 0,
            "final_uuid": None,
            "total_cost": 0.0,
            "total_rewards": 0.0
        }
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            config = Config()
            config.from_json(config_data)
            
            dgm = GodelMachine(config)
            
            loop.run_until_complete(
                dgm.start_dgm(
                    goal_prompt=goal,
                    template_uuid=template_uuid,
                    judge=judge,
                    human_validation=human_validation
                )
            )
            
            result["status"] = "completed"
            result["execution_time"] = time.time() - start_time
            
            print(f"✅ {process_id} completed in {result['execution_time']:.2f}s")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["execution_time"] = time.time() - start_time
            print(f"❌ Process {process_id} failed: {str(e)}")
            
        finally:
            if 'loop' in locals():
                loop.close()
                
        return result
    
    def start_parallel_testing(
        self,
        goals: list[str],
        template_uuid: str = None,
        judge: bool = False,
        human_validation: bool = False,
        max_workers: int = None
    ) -> list[dict[str, Any]]:
        """Start massive testing with multiple goals in parallel.
        
        Args:
            goals: List of goal prompts to test
            template_uuid: Optional workflow template UUID to use for all goals
            judge: Whether to enable judge evaluation
            human_validation: Whether to enable human validation
            max_workers: Maximum number of parallel processes (defaults to CPU count)
            
        Returns:
            List of result dictionaries for each goal
        """
        if not goals:
            raise ValueError("No goals provided for massive testing")
            
        if max_workers is None:
            max_workers = min(len(goals), mp.cpu_count())
            
        print(f"Starting testing with {len(goals)} goals using {max_workers} processes")
        print(f"📊 Results will be saved to: {self.results_dir}")
        
        config_data = self.config.jsonify()
        goal_data_list = []
        for i, goal in enumerate(goals):
            goal_data = {
                "goal": goal,
                "process_id": i + 1,
                "template_uuid": template_uuid,
                "judge": judge,
                "human_validation": human_validation,
                "config_data": config_data
            }
            goal_data_list.append(goal_data)
        
        results = []
        start_time = time.time()
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_goal = {
                    executor.submit(self._run_single_dgm, goal_data): goal_data
                    for goal_data in goal_data_list
                }
                
                for future in future_to_goal:
                    try:
                        result = future.result()
                        results.append(result)
                        self._save_individual_result(result)
                        
                    except Exception as e:
                        goal_data = future_to_goal[future]
                        error_result = {
                            "process_id": goal_data["process_id"],
                            "goal": goal_data["goal"],
                            "status": "failed",
                            "error": f"Process execution failed: {str(e)}",
                            "execution_time": 0
                        }
                        results.append(error_result)
                        print(f"❌ Process {goal_data['process_id']} failed : {str(e)}")
                        
        except KeyboardInterrupt:
            print("\n⚠️ Parallel testing interrupted by user")
            raise
            
        total_time = time.time() - start_time
        
        summary = self._create_summary(results, total_time)
        self._save_summary(summary)
        
        print(f"\n🎉 Massive testing completed in {total_time:.2f}s")
        print(f"📈 Summary: {summary['completed']}/{summary['total']} goals completed.")
        
        return results
    
    def _save_individual_result(self, result: dict[str, Any]) -> None:
        """Save individual result to file."""
        filename = f"result_process_{result['process_id']:03d}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _create_summary(self, results: list[dict[str, Any]],
                        total_time: float) -> dict[str, Any]:
        """Create summary statistics from results."""
        total = len(results)
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")
        
        avg_execution_time = 0
        total_cost = 0
        total_rewards = 0
        
        if completed > 0:
            completed_results = [r for r in results if r["status"] == "completed"]
            avg_execution_time = sum(r["execution_time"] for r in completed_results) / completed
            total_cost = sum(r.get("total_cost", 0) for r in completed_results)
            total_rewards = sum(r.get("total_rewards", 0) for r in completed_results)
        
        return {
            "timestamp": time.time(),
            "total": total,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total) * 100 if total > 0 else 0,
            "total_execution_time": total_time,
            "avg_execution_time": avg_execution_time,
            "total_cost": total_cost,
            "total_rewards": total_rewards,
            "avg_rewards": total_rewards / completed if completed > 0 else 0,
            "results": results
        }
    
    def _save_summary(self, summary: dict[str, Any]) -> None:
        """Save summary to file."""
        timestamp = int(summary["timestamp"])
        filename = f"parallel_testing_summary_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Summary saved to: {filepath}")
