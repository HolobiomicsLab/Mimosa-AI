"""
Pygame visualization for Planner task execution.
Displays a timeline with task progress, statuses, and outputs.
"""

import pygame
import sys
from typing import List
from pathlib import Path
from sources.core.schema import Task, TaskStatus, Plan


class PlannerVisualizer:
    """Real-time visualization of planner task execution using pygame."""

    # Colors
    COLOR_BG = (30, 30, 40)
    COLOR_LINE = (100, 100, 120)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_DIM = (150, 150, 160)
    COLOR_PENDING = (100, 100, 120)
    COLOR_RUNNING = (255, 165, 0)  # Orange
    COLOR_SUCCESS = (50, 205, 50)  # Green
    COLOR_FAILED = (220, 50, 50)   # Red
    COLOR_SKIPPED = (150, 150, 150)  # Grey

    # Layout constants
    DOT_RADIUS = 12
    LINE_THICKNESS = 4
    MARGIN_TOP = 80
    MARGIN_BOTTOM = 100
    MARGIN_LEFT = 50
    MARGIN_RIGHT = 50
    DOT_SPACING_MIN = 100
    UUID_OFFSET_Y = 35
    OUTPUT_OFFSET_Y = 60
    MAX_OUTPUT_LINES = 3

    def __init__(self, plan: Plan, width: int = 1200, height: int = 600):
        """
        Initialize the pygame visualizer.

        Args:
            plan: The execution plan to visualize
            width: Window width in pixels
            height: Window height in pixels
        """
        pygame.init()

        self.plan = plan
        self.tasks: List[Task] = []
        self.total_cost: float = 0.0
        self.width = width
        self.height = height

        # Calculate dot spacing based on number of tasks
        num_tasks = len(plan.steps)
        available_width = width - self.MARGIN_LEFT - self.MARGIN_RIGHT
        self.dot_spacing = max(self.DOT_SPACING_MIN, available_width // max(num_tasks - 1, 1) if num_tasks > 1 else available_width)

        # Calculate total timeline width and center it
        if num_tasks > 1:
            timeline_width = (num_tasks - 1) * self.dot_spacing
        else:
            timeline_width = 0

        # Calculate left offset to center the timeline
        self.timeline_offset_x = (width - timeline_width) // 2

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Mimosa-AI Planner Execution")

        # Fonts
        self.font_title = pygame.font.SysFont('arial', 24, bold=True)
        self.font_task = pygame.font.SysFont('arial', 14, bold=True)
        self.font_small = pygame.font.SysFont('arial', 10)
        self.font_tiny = pygame.font.SysFont('arial', 9)

        self.running = True

    def update_tasks(self, tasks: List[Task], total_cost: float = None):
        """
        Update the task list and refresh the display.

        Args:
            tasks: Current list of tasks with their statuses
            total_cost: Optional total cost to display
        """
        self.tasks = tasks
        if total_cost is not None:
            self.total_cost = total_cost
        self.render()

    def _get_status_color(self, status: TaskStatus) -> tuple:
        """Get the color for a given task status."""
        color_map = {
            TaskStatus.PENDING: self.COLOR_PENDING,
            TaskStatus.RUNNING: self.COLOR_RUNNING,
            TaskStatus.COMPLETED: self.COLOR_SUCCESS,
            TaskStatus.FAILED: self.COLOR_FAILED,
            TaskStatus.SKIPPED: self.COLOR_SKIPPED,
        }
        return color_map.get(status, self.COLOR_PENDING)

    def _get_dot_position(self, index: int) -> tuple:
        """Calculate the x, y position for a task dot."""
        x = self.timeline_offset_x + index * self.dot_spacing
        y = self.MARGIN_TOP + (self.height - self.MARGIN_TOP - self.MARGIN_BOTTOM) // 2
        return (x, y)

    def _draw_text_wrapped(self, text: str, x: int, y: int, font: pygame.font.Font,
                          color: tuple, max_width: int, max_lines: int = None) -> int:
        """
        Draw text with word wrapping.

        Args:
            text: Text to draw
            x, y: Position to start drawing
            font: Font to use
            color: Text color
            max_width: Maximum width before wrapping
            max_lines: Maximum number of lines to display

        Returns:
            int: Total height of rendered text
        """
        words = text.split(' ')
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + word + " "
            test_surface = font.render(test_line, True, color)
            if test_surface.get_width() <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "

        if current_line:
            lines.append(current_line.strip())

        # Limit number of lines if specified
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines - 1]
            lines.append("...")

        # Draw lines
        total_height = 0
        for i, line in enumerate(lines):
            surface = font.render(line, True, color)
            self.screen.blit(surface, (x, y + i * (font.get_height() + 2)))
            total_height = (i + 1) * (font.get_height() + 2)

        return total_height

    def render(self):
        """Render the complete visualization."""
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Draw title
        title_text = f"Plan: {self.plan.goal[:80]}..."
        title_surface = self.font_title.render(title_text, True, self.COLOR_TEXT)
        title_x = (self.width - title_surface.get_width()) // 2
        self.screen.blit(title_surface, (title_x, 20))

        # Draw cost in bottom right corner
        cost_text = f"Cost: {round(self.total_cost, 2)} $"
        cost_surface = self.font_task.render(cost_text, True, self.COLOR_TEXT)
        cost_x = self.width - cost_surface.get_width() - 20
        cost_y = self.height - 30
        self.screen.blit(cost_surface, (cost_x, cost_y))

        # Draw progress line
        if len(self.plan.steps) > 1:
            start_pos = self._get_dot_position(0)
            end_pos = self._get_dot_position(len(self.plan.steps) - 1)
            pygame.draw.line(self.screen, self.COLOR_LINE, start_pos, end_pos, self.LINE_THICKNESS)

        # Draw task dots and information
        for i, step in enumerate(self.plan.steps):
            pos = self._get_dot_position(i)

            # Find corresponding task in task list
            task = next((t for t in self.tasks if t.name == step.name), None)
            status = task.status if task else TaskStatus.PENDING

            # Draw dot
            color = self._get_status_color(status)
            pygame.draw.circle(self.screen, color, pos, self.DOT_RADIUS)
            pygame.draw.circle(self.screen, self.COLOR_TEXT, pos, self.DOT_RADIUS, 2)

            # Draw task name above dot
            task_name_surface = self.font_task.render(step.name[:20], True, self.COLOR_TEXT)
            task_name_x = pos[0] - task_name_surface.get_width() // 2
            self.screen.blit(task_name_surface, (task_name_x, pos[1] - 30))

            # Draw UUID below dot
            if task and task.final_uuid:
                uuid_text = f"UUID: {task.final_uuid}..."
                uuid_surface = self.font_small.render(uuid_text, True, self.COLOR_TEXT_DIM)
                uuid_x = pos[0] - uuid_surface.get_width() // 2
                self.screen.blit(uuid_surface, (uuid_x, pos[1] + self.UUID_OFFSET_Y))

            # Draw expected outputs below UUID with color coding
            if step.expected_outputs:
                output_y = pos[1] + self.OUTPUT_OFFSET_Y
                output_label = self.font_tiny.render("Expected Outputs:", True, self.COLOR_TEXT_DIM)
                output_x = pos[0] - output_label.get_width() // 2
                self.screen.blit(output_label, (output_x, output_y))

                output_y += self.font_tiny.get_height() + 2

                # Get produced outputs for this task
                produced_outputs = task.produced_outputs if task else []

                # Show up to MAX_OUTPUT_LINES expected outputs
                outputs_to_show = step.expected_outputs[:self.MAX_OUTPUT_LINES]
                for expected_output in outputs_to_show:
                    # Check if this output was produced
                    is_produced = any(expected_output in produced for produced in produced_outputs)
                    # Color based on whether it was produced
                    output_color = self.COLOR_SUCCESS if is_produced else self.COLOR_FAILED
                    filename = Path(expected_output).name
                    if len(filename) > 25:
                        filename = filename[:22] + "..."
                    # Add checkmark or X indicator
                    indicator = "✓" if is_produced else "✗"
                    output_surface = self.font_tiny.render(f"{indicator} {filename}", True, output_color)
                    output_x = pos[0] - output_surface.get_width() // 2
                    self.screen.blit(output_surface, (output_x, output_y))
                    output_y += self.font_tiny.get_height() + 1

                # Show count if more outputs exist
                if len(step.expected_outputs) > self.MAX_OUTPUT_LINES:
                    more_text = f"+{len(step.expected_outputs) - self.MAX_OUTPUT_LINES} more"
                    more_surface = self.font_tiny.render(more_text, True, self.COLOR_TEXT_DIM)
                    more_x = pos[0] - more_surface.get_width() // 2
                    self.screen.blit(more_surface, (more_x, output_y))

        # Draw legend at bottom
        legend_y = self.height - 60
        legend_items = [
            ("Pending", self.COLOR_PENDING),
            ("Running", self.COLOR_RUNNING),
            ("Success", self.COLOR_SUCCESS),
            ("Failed", self.COLOR_FAILED),
        ]

        legend_x = self.MARGIN_LEFT
        legend_spacing = 120

        legend_title = self.font_small.render("Status Legend:", True, self.COLOR_TEXT)
        self.screen.blit(legend_title, (legend_x, legend_y - 25))

        for i, (label, color) in enumerate(legend_items):
            x = legend_x + i * legend_spacing
            # Draw small dot
            pygame.draw.circle(self.screen, color, (x, legend_y), 8)
            pygame.draw.circle(self.screen, self.COLOR_TEXT, (x, legend_y), 8, 1)
            # Draw label
            label_surface = self.font_small.render(label, True, self.COLOR_TEXT)
            self.screen.blit(label_surface, (x + 15, legend_y - 8))

        # Update display
        pygame.display.flip()

    def handle_events(self):
        """Handle pygame events (close window, etc.)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.close()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                    self.close()

    def is_running(self) -> bool:
        """Check if the visualizer is still running."""
        return self.running

    def close(self):
        """Close the pygame window and cleanup."""
        pygame.quit()
        self.running = False


# Example usage for testing
if __name__ == "__main__":
    from sources.core.schema import Plan, PlanStep, TaskStatus

    # Create a sample plan
    steps = [
        PlanStep(name="Task1", task="First task", expected_outputs=["output1.txt"]),
        PlanStep(name="Task2", task="Second task", expected_outputs=["output2.txt", "output3.txt"]),
        PlanStep(name="Task3", task="Third task", expected_outputs=["output4.txt"]),
        PlanStep(name="Task4", task="Fourth task", expected_outputs=["output5.txt"]),
    ]
    plan = Plan(goal="Test Plan for Visualization", steps=steps)

    # Create visualizer
    viz = PlannerVisualizer(plan)

    # Create sample tasks
    tasks = [
        Task(name="Task1", description="First task", status=TaskStatus.COMPLETED,
             final_uuid="abc123def456", produced_outputs=["output1.txt", "extra1.txt"]),
        Task(name="Task2", description="Second task", status=TaskStatus.RUNNING,
             final_uuid="xyz789uvw012", produced_outputs=["output2.txt"]),
        Task(name="Task3", description="Third task", status=TaskStatus.PENDING),
        Task(name="Task4", description="Fourth task", status=TaskStatus.PENDING),
    ]

    viz.update_tasks(tasks)

    # Keep window open
    clock = pygame.time.Clock()
    while viz.is_running():
        viz.handle_events()
        clock.tick(30)  # 30 FPS
