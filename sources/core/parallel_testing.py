"""
Simplified Parallel Testing Module for DGM execution
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from config import Config
from sources.core.dgm import GodelMachine
from sources.utils.shared_visualization import (
    ParallelPlotManager,
    SharedVisualizationData,
)
from sources.utils.visualization import VisualizationUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelTesting:
    """Simplified class for running multiple DGM instances in parallel."""

    def __init__(self, config: Config):
        """Initialize ParallelTesting with configuration."""
        self.config = config
        self.results_dir = Path("parallel_testing_results")
        self.results_dir.mkdir(exist_ok=True)
        self._shutdown_requested = False

    @staticmethod
    def _run_single_dgm(goal_data: dict[str, Any], results_dir_path: str) -> dict[str, Any]:
        """Run a single DGM instance in a separate process."""
        # Extract parameters
        goal = goal_data["goal"]
        process_id = goal_data["process_id"]
        template_uuid = goal_data.get("template_uuid")
        judge = goal_data.get("judge", False)
        human_validation = goal_data.get("human_validation", False)
        config_data = goal_data["config_data"]

        logger.info(f"Starting DGM process {process_id} (PID: {os.getpid()}) for goal: {goal[:50]}")

        # Initialize result structure
        result = {
            "process_id": process_id,
            "pid": os.getpid(),
            "goal": goal,
            "template_uuid": template_uuid,
            "judge": judge,
            "human_validation": human_validation,
            "start_time": time.time(),
            "status": "running",
            "error": None,
            "execution_time": 0,
            "final_uuid": None,
            "total_cost": 0.0,
            "total_rewards": 0.0,
        }

        loop = None
        try:
            # Set up asyncio loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Initialize configuration and components
            config = Config()
            config.from_json(config_data)
            
            viz_utils = VisualizationUtils()
            shared_viz_data = SharedVisualizationData(Path(results_dir_path))

            # Create and run DGM
            dgm = GodelMachine(
                config,
                viz_utils=viz_utils,
                shared_viz_data=shared_viz_data,
                process_id=process_id,
            )

            uuid = loop.run_until_complete(
                dgm.start_dgm(
                    goal=goal,
                    template_uuid=template_uuid,
                    judge=judge,
                    human_validation=human_validation,
                    max_iteration=5,
                )
            )

            # Update result with success
            result.update({
                "status": "completed",
                "final_uuid": uuid,
                "execution_time": time.time() - result["start_time"]
            })

            logger.info(f"✅ Process {process_id} completed in {result['execution_time']:.2f}s")

        except KeyboardInterrupt:
            result.update({
                "status": "interrupted",
                "error": "Process interrupted",
                "execution_time": time.time() - result["start_time"]
            })
            logger.info(f"🛑 Process {process_id} interrupted")

        except Exception as e:
            result.update({
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - result["start_time"]
            })
            logger.error(f"❌ Process {process_id} failed: {str(e)}")

        finally:
            # Clean up asyncio loop
            if loop:
                try:
                    # Cancel pending tasks
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                finally:
                    loop.close()

        return result

    def _setup_visualization(self, num_goals: int) -> tuple[Optional[SharedVisualizationData], Optional[ParallelPlotManager]]:
        """Set up visualization components."""
        try:
            viz_utils = VisualizationUtils()
            shared_viz_data = SharedVisualizationData(self.results_dir)
            shared_viz_data.cleanup_old_data()
            
            plot_manager = ParallelPlotManager(shared_viz_data, viz_utils)
            plot_manager.start_real_time_plotting(
                title=f"Parallel Testing - {num_goals} Goals"
            )
            
            return shared_viz_data, plot_manager
        except Exception as e:
            logger.warning(f"Failed to setup visualization: {e}")
            return None, None

    def _cleanup_visualization(self, plot_manager: Optional[ParallelPlotManager], 
                             shared_viz_data: Optional[SharedVisualizationData],
                             goal_data_list: list[dict]) -> None:
        """Clean up visualization components."""
        if plot_manager:
            try:
                # Save final plot
                timestamp = int(time.time())
                plot_filename = self.results_dir / f"parallel_curves_{timestamp}.png"
                plot_manager.save_combined_plot(str(plot_filename))
                plot_manager.stop_plotting()
            except Exception as e:
                logger.warning(f"Error during plot cleanup: {e}")

        if shared_viz_data:
            # Mark all processes as completed
            for goal_data in goal_data_list:
                shared_viz_data.mark_process_completed(goal_data["process_id"])

    def _prepare_goal_data(self, goals: list[str], template_uuid: Optional[str], 
                          judge: bool, human_validation: bool) -> list[dict[str, Any]]:
        """Prepare goal data for parallel execution."""
        config_data = self.config.jsonify()
        goal_data_list = []
        
        for i, goal in enumerate(goals):
            goal_data = {
                "goal": goal,
                "process_id": i + 1,
                "template_uuid": template_uuid,
                "judge": judge,
                "human_validation": human_validation,
                "config_data": config_data,
            }
            goal_data_list.append(goal_data)
        
        return goal_data_list

    def _execute_parallel_tasks(self, goal_data_list: list[dict], max_workers: int) -> list[dict[str, Any]]:
        """Execute tasks in parallel and collect results."""
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_goal = {
                executor.submit(self._run_single_dgm, goal_data, str(self.results_dir)): goal_data
                for goal_data in goal_data_list
            }

            logger.info(f"Submitted {len(future_to_goal)} tasks to executor")

            # Process completed futures
            completed_count = 0
            for future in as_completed(future_to_goal):
                if self._shutdown_requested:
                    logger.info("Shutdown requested, cancelling remaining tasks...")
                    break

                goal_data = future_to_goal[future]
                try:
                    result = future.result(timeout=1.0)
                    results.append(result)
                    self._save_individual_result(result)
                    completed_count += 1
                    logger.info(f"Completed {completed_count}/{len(goal_data_list)} tasks")

                except Exception as e:
                    error_result = {
                        "process_id": goal_data["process_id"],
                        "goal": goal_data["goal"],
                        "status": "failed",
                        "error": f"Process execution failed: {str(e)}",
                        "execution_time": 0,
                    }
                    results.append(error_result)
                    logger.error(f"❌ Process {goal_data['process_id']} failed: {str(e)}")

        return results

    def start_parallel_testing(
        self,
        goals: list[str],
        template_uuid: Optional[str] = None,
        judge: bool = False,
        human_validation: bool = False,
        max_workers: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Start parallel testing with multiple goals.
        
        Args:
            goals: List of goal prompts to test
            template_uuid: Optional workflow template UUID
            judge: Whether to enable judge evaluation
            human_validation: Whether to enable human validation
            max_workers: Maximum number of parallel processes
            
        Returns:
            List of result dictionaries for each goal
        """
        if not goals:
            raise ValueError("No goals provided for testing")

        if max_workers is None:
            max_workers = min(len(goals), mp.cpu_count())

        print(f"Starting testing with {len(goals)} goals using {max_workers} processes")
        print(f"📊 Results will be saved to: {self.results_dir}")

        # Prepare data and setup
        goal_data_list = self._prepare_goal_data(goals, template_uuid, judge, human_validation)
        shared_viz_data, plot_manager = self._setup_visualization(len(goals))
        
        start_time = time.time()
        results = []

        # Set up signal handler for graceful shutdown
        original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            results = self._execute_parallel_tasks(goal_data_list, max_workers)
            
        except KeyboardInterrupt:
            logger.info("\n⚠️ Parallel testing interrupted by user (Ctrl+C)")
            self._shutdown_requested = True
            
        except Exception as e:
            logger.error(f"Unexpected error during parallel testing: {e}")
            self._shutdown_requested = True
            
        finally:
            # Restore signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            # Cleanup visualization
            self._cleanup_visualization(plot_manager, shared_viz_data, goal_data_list)

        # Generate and save summary
        total_time = time.time() - start_time
        summary = self._create_summary(results, total_time)
        self._save_summary(summary)

        # Print final statistics
        self._print_final_stats(summary, shared_viz_data, total_time)

        return results

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_requested = True

    def _save_individual_result(self, result: dict[str, Any]) -> None:
        """Save individual result to file."""
        filename = f"result_process_{result['process_id']:03d}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)

    def _create_summary(self, results: list[dict[str, Any]], total_time: float) -> dict[str, Any]:
        """Create summary statistics from results."""
        total = len(results)
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")

        # Calculate averages for completed results
        completed_results = [r for r in results if r["status"] == "completed"]
        avg_execution_time = (
            sum(r["execution_time"] for r in completed_results) / completed
            if completed > 0 else 0
        )
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
            "results": results,
        }

    def _save_summary(self, summary: dict[str, Any]) -> None:
        """Save summary to file."""
        timestamp = int(summary["timestamp"])
        filename = f"parallel_testing_summary_{timestamp}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"📄 Summary saved to: {filepath}")

    def _print_final_stats(self, summary: dict[str, Any], 
                          shared_viz_data: Optional[SharedVisualizationData], 
                          total_time: float) -> None:
        """Print final statistics."""
        # Print visualization summary if available
        if shared_viz_data:
            try:
                viz_stats = shared_viz_data.get_summary_stats()
                print("\n📊 Visualization Summary:")
                print(f"   Total processes: {viz_stats['total_processes']}")
                print(f"   Completed processes: {viz_stats['completed_processes']}")
                print(f"   Average reward: {viz_stats['average_reward']:.3f}")
                print(f"   Max reward: {viz_stats['max_reward']:.3f}")
                print(f"   Total iterations: {viz_stats['total_iterations']}")
            except Exception as e:
                logger.warning(f"Error getting visualization stats: {e}")

        print(f"\n🎉 Parallel testing completed in {total_time:.2f}s")
        print(f"📈 Summary: {summary['completed']}/{summary['total']} goals completed.")
