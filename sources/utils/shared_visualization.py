"""
Shared Visualization Data Management for Parallel Processing
"""

import json
import threading
import time
from pathlib import Path
from typing import Any

import matplotlib

# Set matplotlib to use a non-interactive backend to avoid threading issues
matplotlib.use("Agg")

# Handle fcntl import for cross-platform compatibility
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    # fcntl is not available on Windows
    HAS_FCNTL = False


class SharedVisualizationData:
    """Manages shared visualization data across multiple parallel processes."""

    def __init__(self, results_dir: Path = Path("run_notes/")):
        """Initialize shared visualization data manager.

        Args:
            results_dir: Directory where parallel testing results are stored
        """
        self.results_dir = results_dir
        self.viz_data_dir = results_dir / "visualization_data"
        self.viz_data_dir.mkdir(exist_ok=True)
        self.master_config_file = self.viz_data_dir / "master_plot_config.json"

    def _get_process_file(self, process_id: int) -> Path:
        """Get the file path for a specific process."""
        return self.viz_data_dir / f"process_{process_id}_curve_data.json"

    def _safe_write_json(self, filepath: Path, data: dict[str, Any]) -> None:
        """Thread-safe JSON writing with file locking."""
        with open(filepath, "w") as f:
            if HAS_FCNTL:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(data, f, indent=2)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except OSError:
                    json.dump(data, f, indent=2)
            else:
                # No file locking on Windows - rely on atomic writes
                json.dump(data, f, indent=2)

    def _safe_read_json(self, filepath: Path) -> dict[str, Any] | None:
        """Thread-safe JSON reading."""
        if not filepath.exists():
            return None

        try:
            with open(filepath) as f:
                if HAS_FCNTL:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        data = json.load(f)
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        return data
                    except OSError:
                        return json.load(f)
                else:
                    # No file locking on Windows - direct read
                    return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def write_curve_data(
        self,
        process_id: int,
        iterations: list[int],
        rewards: list[float],
        goal: str = "",
        status: str = "running",
    ) -> None:
        """Write curve data for a specific process.

        Args:
            process_id: Unique identifier for the process
            iterations: List of iteration numbers
            rewards: List of reward values corresponding to iterations
            goal: Goal description for this process
            status: Current status of the process
        """
        process_file = self._get_process_file(process_id)

        colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        styles = ["-o", "--s", "-.^", ":d", "-v", "--<", "-.>", ":p", "-*", "--+"]

        data = {
            "process_id": process_id,
            "goal": goal,
            "curve_data": {"iterations": iterations, "rewards": rewards},
            "metadata": {
                "start_time": time.time()
                if not process_file.exists()
                else self._get_start_time(process_id),
                "last_update": time.time(),
                "status": status,
                "color": colors[process_id % len(colors)],
                "style": styles[process_id % len(styles)],
                "label": f"Process {process_id}",
            },
        }

        self._safe_write_json(process_file, data)

    def _get_start_time(self, process_id: int) -> float:
        """Get the start time for a process, or current time if not found."""
        existing_data = self._safe_read_json(self._get_process_file(process_id))
        if existing_data and "metadata" in existing_data:
            return existing_data["metadata"].get("start_time", time.time())
        return time.time()

    def read_all_curve_data(self) -> dict[int, dict[str, Any]]:
        """Read curve data from all active processes.

        Returns:
            Dictionary mapping process_id to curve data
        """
        all_data = {}

        for process_file in self.viz_data_dir.glob("process_*_curve_data.json"):
            try:
                process_id = int(process_file.stem.split("_")[1])
                data = self._safe_read_json(process_file)

                if data:
                    all_data[process_id] = data

            except (ValueError, IndexError):
                continue

        return all_data

    def get_active_processes(self) -> list[int]:
        """Get list of currently active process IDs.

        Returns:
            List of process IDs that are currently running
        """
        active_processes = []
        current_time = time.time()

        for process_id, data in self.read_all_curve_data().items():
            metadata = data.get("metadata", {})
            last_update = metadata.get("last_update", 0)
            status = metadata.get("status", "unknown")

            # Consider a process active if it was updated within the last 30 seconds
            if (current_time - last_update < 30) or status == "running":
                active_processes.append(process_id)

        return sorted(active_processes)

    def mark_process_completed(self, process_id: int) -> None:
        """Mark a process as completed.

        Args:
            process_id: ID of the process to mark as completed
        """
        process_file = self._get_process_file(process_id)
        data = self._safe_read_json(process_file)

        if data:
            data["metadata"]["status"] = "completed"
            data["metadata"]["last_update"] = time.time()
            self._safe_write_json(process_file, data)

    def cleanup_old_data(self, max_age_hours: int = 24) -> None:
        """Clean up old visualization data files.

        Args:
            max_age_hours: Maximum age in hours for data files to keep
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for process_file in self.viz_data_dir.glob("process_*_curve_data.json"):
            try:
                file_age = current_time - process_file.stat().st_mtime
                if file_age > max_age_seconds:
                    process_file.unlink()
                    print(f"Cleaned up old visualization data: {process_file.name}")
            except OSError:
                continue

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for all processes.

        Returns:
            Dictionary with summary statistics
        """
        all_data = self.read_all_curve_data()

        if not all_data:
            return {"total_processes": 0, "active_processes": 0}

        active_count = len(self.get_active_processes())
        completed_count = sum(
            1
            for data in all_data.values()
            if data.get("metadata", {}).get("status") == "completed"
        )

        # Calculate average rewards across all processes
        all_rewards = []
        for data in all_data.values():
            rewards = data.get("curve_data", {}).get("rewards", [])
            if rewards:
                all_rewards.extend(rewards)

        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        max_reward = max(all_rewards) if all_rewards else 0

        return {
            "total_processes": len(all_data),
            "active_processes": active_count,
            "completed_processes": completed_count,
            "average_reward": avg_reward,
            "max_reward": max_reward,
            "total_iterations": sum(
                len(data.get("curve_data", {}).get("iterations", []))
                for data in all_data.values()
            ),
        }


class ParallelPlotManager:
    """Manages real-time plotting for multiple parallel processes."""

    def __init__(self, shared_data: SharedVisualizationData, viz_utils):
        """Initialize parallel plot manager.

        Args:
            shared_data: SharedVisualizationData instance
            viz_utils: VisualizationUtils instance
        """
        self.shared_data = shared_data
        self.viz_utils = viz_utils
        self.plot_data = None
        self.update_interval = 2.0
        self.is_running = False
        self.update_thread = None
        self._lock = threading.Lock()

    def start_real_time_plotting(
        self, title: str = "Parallel Processes - Rewards Curves"
    ) -> None:
        """Start real-time plotting with background updates.

        Args:
            title: Title for the combined plot
        """
        if self.is_running:
            return

        # Create initial empty multi-curve plot
        self.plot_data = self.viz_utils.create_multi_curve_plot(
            title=title,
            xlabel="Iteration",
            ylabel="Rewards",
            curve_configs=[],
            figsize=[14, 8],
        )

        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        print(f"🎯 Started real-time parallel plotting: {title}")

    def _update_loop(self) -> None:
        """Background thread loop for updating the plot."""
        while self.is_running:
            try:
                self.update_plot()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in plot update loop: {e}")
                time.sleep(self.update_interval)

    def update_plot(self) -> None:
        """Update the plot with latest data from all processes."""
        if not self.plot_data:
            return

        with self._lock:
            all_data = self.shared_data.read_all_curve_data()

            if not all_data:
                return

            fig, ax, lines = self.plot_data

            ax.clear()
            ax.set_title("Parallel Processes - Rewards Curves")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Rewards")
            ax.grid(True, alpha=0.3)

            # Plot each process curve
            for process_id, data in all_data.items():
                curve_data = data.get("curve_data", {})
                metadata = data.get("metadata", {})

                iterations = curve_data.get("iterations", [])
                rewards = curve_data.get("rewards", [])

                if iterations and rewards:
                    color = metadata.get("color", "blue")
                    style = metadata.get("style", "-o")
                    label = f"Process {process_id} - {data.get('goal', '')[:30]}..."

                    ax.plot(
                        iterations,
                        rewards,
                        style,
                        color=color,
                        linewidth=2,
                        markersize=4,
                        label=label,
                        alpha=0.8,
                    )

            if len(all_data) > 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            fig.tight_layout()
            # Remove problematic GUI operations that cause threading issues on macOS
            # fig.canvas.draw() and fig.canvas.flush_events() removed

    def stop_plotting(self) -> None:
        """Stop the real-time plotting."""
        self.is_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        print("🛑 Stopped real-time parallel plotting")

    def save_combined_plot(self, filename: str) -> None:
        """Save the current combined plot to file.

        Args:
            filename: Name of the file to save the plot
        """
        if self.plot_data:
            self.viz_utils.save_plot(self.plot_data, filename)
