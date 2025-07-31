"""
Massive Testing Module for parallel DGM execution
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import psutil

from config import Config
from sources.core.dgm import GodelMachine
from sources.utils.shared_visualization import (
    ParallelPlotManager,
    SharedVisualizationData,
)
from sources.utils.visualization import VisualizationUtils

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessManager:
    """Thread-safe process manager for tracking and controlling subprocesses."""

    def __init__(self):
        self._processes: dict[int, mp.Process] = {}
        self._pids: set[int] = set()
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()

    def register_process(self, process_id: int, process: mp.Process) -> None:
        """Register a process for tracking."""
        with self._lock:
            self._processes[process_id] = process
            if process.pid:
                self._pids.add(process.pid)
                logger.info(f"Registered process {process_id} with PID {process.pid}")

    def unregister_process(self, process_id: int) -> None:
        """Unregister a completed process."""
        with self._lock:
            if process_id in self._processes:
                process = self._processes.pop(process_id)
                if process.pid and process.pid in self._pids:
                    self._pids.remove(process.pid)
                logger.info(f"Unregistered process {process_id}")

    def get_active_pids(self):
        """Get set of active process PIDs."""
        with self._lock:
            return self._pids.copy()

    def terminate_all(self, timeout: float = 10.0) -> None:
        """Terminate all tracked processes gracefully."""
        logger.info("Initiating graceful shutdown of all processes...")
        self._shutdown_event.set()

        with self._lock:
            active_pids = self._pids.copy()

        if not active_pids:
            logger.info("No active processes to terminate")
            return

        # First, try graceful termination
        for pid in active_pids:
            try:
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    logger.info(f"Sending SIGTERM to process {pid}")
                    process.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Could not terminate process {pid}: {e}")

        # Wait for processes to terminate gracefully
        start_time = time.time()
        while time.time() - start_time < timeout:
            remaining_pids = []
            for pid in active_pids:
                try:
                    if psutil.pid_exists(pid):
                        remaining_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass

            if not remaining_pids:
                logger.info("All processes terminated gracefully")
                break

            time.sleep(0.1)
        else:
            # Force kill remaining processes
            for pid in remaining_pids:
                try:
                    if psutil.pid_exists(pid):
                        process = psutil.Process(pid)
                        logger.warning(f"Force killing process {pid}")
                        process.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.error(f"Could not kill process {pid}: {e}")

        # Clear tracking
        with self._lock:
            self._processes.clear()
            self._pids.clear()

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()


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
        self.viz_utils = VisualizationUtils()
        self.shared_viz_data = SharedVisualizationData(self.results_dir)
        self.plot_manager = None
        self.process_manager = ProcessManager()
        self._original_sigint_handler = None
        self._original_sigterm_handler = None
        self._shutdown_requested = False

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_requested = True
            self.process_manager.terminate_all()

        self._original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, signal_handler)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint_handler:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
        if self._original_sigterm_handler:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)

    @staticmethod
    def _run_single_dgm(goal_data: dict[str, Any], results_dir_path: str):
        """Run a single DGM instance in a separate process.
        Args:
            goal_data: Dictionary containing goal information and parameters
            results_dir_path: Path to results directory for shared visualization
        Returns:
            Dictionary with execution results
        """

        # Set up signal handling in subprocess
        def subprocess_signal_handler(signum, frame):
            logger.info(f"Subprocess {os.getpid()} received signal {signum}")
            raise KeyboardInterrupt("Process interrupted by signal")

        signal.signal(signal.SIGINT, subprocess_signal_handler)
        signal.signal(signal.SIGTERM, subprocess_signal_handler)

        goal = goal_data["goal"]
        process_id = goal_data["process_id"]
        template_uuid = goal_data.get("template_uuid")
        judge = goal_data.get("judge", False)
        human_validation = goal_data.get("human_validation", False)
        config_data = goal_data["config_data"]

        logger.info(f"Starting DGM process (PID: {os.getpid()}) for goal: {goal[:50]}")

        start_time = time.time()
        result = {
            "process_id": process_id,
            "pid": os.getpid(),
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
            "total_rewards": 0.0,
        }

        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            config = Config()
            config.from_json(config_data)

            # Create visualization objects within the process
            viz_utils = VisualizationUtils()
            shared_viz_data = SharedVisualizationData(Path(results_dir_path))

            dgm = GodelMachine(
                config,
                viz_utils=viz_utils,
                shared_viz_data=shared_viz_data,
                process_id=process_id,
            )

            loop.run_until_complete(
                dgm.start_dgm(
                    goal_prompt=goal,
                    template_uuid=template_uuid,
                    judge=judge,
                    human_validation=human_validation,
                    max_iteration=5,
                )
            )

            result["status"] = "completed"
            result["execution_time"] = time.time() - start_time

            logger.info(f"✅ Process {process_id} : {result['execution_time']:.2f}s")

        except KeyboardInterrupt:
            result["status"] = "interrupted"
            result["error"] = "Process interrupted by signal"
            result["execution_time"] = time.time() - start_time
            logger.info(f"🛑 Process {process_id} interrupted")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["execution_time"] = time.time() - start_time
            logger.error(f"❌ Process {process_id} failed: {str(e)}")

        finally:
            if loop:
                try:
                    # Cancel all pending tasks
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()

                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception as e:
                    logger.warning(f"Error during loop cleanup: {e}")
                finally:
                    loop.close()

        return result

    def start_parallel_testing(
        self,
        goals: list[str],
        template_uuid: str = None,
        judge: bool = False,
        human_validation: bool = False,
        max_workers: int = None,
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

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

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

        # Initialize shared visualization and start real-time plotting
        self.shared_viz_data.cleanup_old_data()  # Clean up old data files
        self.plot_manager = ParallelPlotManager(self.shared_viz_data, self.viz_utils)
        self.plot_manager.start_real_time_plotting(
            title=f"Parallel Testing - {len(goals)} Goals"
        )

        results = []
        start_time = time.time()
        executor = None

        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
            # Submit all tasks and track futures
            future_to_goal = {}
            for goal_data in goal_data_list:
                future = executor.submit(
                    self._run_single_dgm, goal_data, str(self.results_dir)
                )
                future_to_goal[future] = goal_data

            logger.info(f"Submitted {len(future_to_goal)} tasks to executor")

            # Process completed futures as they finish
            completed_count = 0
            for future in as_completed(future_to_goal):
                if self._shutdown_requested:
                    logger.info("Shutdown requested, cancelling remaining tasks...")
                    break

                goal_data = future_to_goal[future]
                try:
                    result = future.result(
                        timeout=1.0
                    )  # Short timeout to check for interruption
                    results.append(result)
                    self._save_individual_result(result)
                    completed_count += 1
                    # Track PID if available
                    if "pid" in result:
                        self.process_manager.unregister_process(result["process_id"])
                    logger.info(
                        f"Completed {completed_count}/{len(goal_data_list)} tasks"
                    )

                except Exception as e:
                    error_result = {
                        "process_id": goal_data["process_id"],
                        "goal": goal_data["goal"],
                        "status": "failed",
                        "error": f"Process execution failed: {str(e)}",
                        "execution_time": 0,
                    }
                    results.append(error_result)
                    logger.error(
                        f"❌ Process {goal_data['process_id']} failed: {str(e)}"
                    )

        except KeyboardInterrupt:
            logger.info("\n⚠️ Parallel testing interrupted by user (Ctrl+C)")
            self._shutdown_requested = True
            # Cancel remaining futures
            for future in future_to_goal:
                if not future.done():
                    future.cancel()
            # Terminate all processes
            self.process_manager.terminate_all(timeout=15.0)

        except Exception as e:
            logger.error(f"Unexpected error during parallel testing: {e}")
            self._shutdown_requested = True

        finally:
            # Cleanup executor
            if executor:
                logger.info("Shutting down executor...")
                executor.shutdown(wait=False)
                # Force terminate any remaining processes
                active_pids = self.process_manager.get_active_pids()
                if active_pids:
                    self.process_manager.terminate_all(timeout=5.0)
            # Restore signal handlers
            self._restore_signal_handlers()
            # Stop the real-time plotting and save final plot
            if self.plot_manager:
                try:
                    # Save the final combined plot
                    timestamp = int(time.time())
                    plot_filename = (
                        self.results_dir / f"parallel_curves_{timestamp}.png"
                    )
                    self.plot_manager.save_combined_plot(str(plot_filename))
                    # Stop the plotting thread
                    self.plot_manager.stop_plotting()
                except Exception as e:
                    logger.warning(f"Error during plot cleanup: {e}")
                # Mark all processes as completed in shared data
                for goal_data in goal_data_list:
                    self.shared_viz_data.mark_process_completed(goal_data["process_id"])

        total_time = time.time() - start_time

        summary = self._create_summary(results, total_time)
        self._save_summary(summary)

        # Print visualization summary
        viz_stats = self.shared_viz_data.get_summary_stats()
        print("\n📊 Visualization Summary:")
        print(f"   Total processes: {viz_stats['total_processes']}")
        print(f"   Completed processes: {viz_stats['completed_processes']}")
        print(f"   Average reward: {viz_stats['average_reward']:.3f}")
        print(f"   Max reward: {viz_stats['max_reward']:.3f}")
        print(f"   Total iterations: {viz_stats['total_iterations']}")

        print(f"\n🎉 Massive testing completed in {total_time:.2f}s")
        print(f"📈 Summary: {summary['completed']}/{summary['total']} goals completed.")

        return results

    def _save_individual_result(self, result: dict[str, Any]) -> None:
        """Save individual result to file."""
        filename = f"result_process_{result['process_id']:03d}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)

    def _create_summary(
        self, results: list[dict[str, Any]], total_time: float
    ) -> dict[str, Any]:
        """Create summary statistics from results."""
        total = len(results)
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")

        avg_execution_time = 0
        total_cost = 0
        total_rewards = 0

        if completed > 0:
            completed_results = [r for r in results if r["status"] == "completed"]
            avg_execution_time = (
                sum(r["execution_time"] for r in completed_results) / completed
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
