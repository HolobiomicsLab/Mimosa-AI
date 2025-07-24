"""
VisualizationUtils class providing various curve plot functions.
"""

from typing import Any
import matplotlib.pyplot as plt


class VisualizationUtils:
    """Utility class for creating and managing types of plots and visualizations."""
    
    def __init__(self):
        """Initialize the VisualizationUtils class."""
        self.active_plots = {}
        
    def create_real_time_curve_plot(
        self, 
        title: str, 
        xlabel: str = "X", 
        ylabel: str = "Y",
        figsize: list[int, int] = (10, 6),
        grid: bool = True,
        grid_alpha: float = 0.3,
        line_style: str = 'b-o',
        line_width: int = 2,
        marker_size: int = 6
    ) -> list[Any, Any, Any]:
        """
        Create a real-time curve plot that can be updated dynamically.
        
        Args:
            title: Title of the plot
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            figsize: Figure size as (width, height)
            grid: Whether to show grid
            grid_alpha: Grid transparency
            line_style: Line style (e.g., 'b-o', 'r--', 'g-')
            line_width: Width of the line
            marker_size: Size of markers
            
        Returns:
            list containing (figure, axis, line) objects
        """
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if grid:
            ax.grid(True, alpha=grid_alpha)
            
        line, = ax.plot([], [], line_style,
                        linewidth=line_width, markersize=marker_size)
        plt.show()
        
        # Store the plot for later reference
        plot_id = f"{title}_{id(fig)}"
        self.active_plots[plot_id] = (fig, ax, line)
        
        return fig, ax, line
    
    def update_curve_plot(
        self, 
        plot_data: list[Any, Any, Any], 
        x_data: list[float], 
        y_data: list[float],
        auto_scale: bool = True,
        x_margin: float = 1.0
    ) -> None:
        """
        Update a real-time curve plot with new data.
        
        Args:
            plot_data: list containing (figure, axis, line) objects
            x_data: list of x-axis values
            y_data: list of y-axis values
            auto_scale: Whether to auto-scale the axes
            x_margin: Margin to add to x-axis limits
        """
        fig, ax, line = plot_data
        
        # Update line data
        line.set_data(x_data, y_data)
        
        if auto_scale:
            ax.relim()
            ax.autoscale_view()
            
            # Set x-axis limits with margin
            if x_data:
                ax.set_xlim(0, max(5, max(x_data) + x_margin))
        
        plt.draw()
        plt.pause(0.01)
    
    def create_rewards_curve_plot(
        self, 
        goal_prompt: str,
        figsize: list[int, int] = (10, 6)
    ) -> list[Any, Any, Any]:
        """
        Create a specialized rewards curve plot for tracking algorithm performance.
        
        Args:
            goal_prompt: The goal prompt to include in the title
            figsize: Figure size as (width, height)
            
        Returns:
            list containing (figure, axis, line) objects
        """
        return self.create_real_time_curve_plot(
            title=f'Rewards Curve - {goal_prompt}',
            xlabel='Iteration',
            ylabel='Rewards',
            figsize=figsize,
            grid=True,
            grid_alpha=0.3,
            line_style='b-o',
            line_width=2,
            marker_size=6
        )
    
    def update_rewards_curve(
        self, 
        plot_data: list[Any, Any, Any], 
        rewards_history: list[float]
    ) -> None:
        """
        Update the rewards curve plot with new rewards data.
        
        Args:
            plot_data: list containing (figure, axis, line) objects
            rewards_history: list of reward values
        """
        iterations = list(range(1, len(rewards_history) + 1))
        self.update_curve_plot(plot_data, iterations, rewards_history)
    
    def create_multi_curve_plot(
        self,
        title: str,
        xlabel: str = "X",
        ylabel: str = "Y",
        curve_configs: list[dict] = None,
        figsize: list[int, int] = (10, 6),
        grid: bool = True,
        grid_alpha: float = 0.3
    ) -> list[Any, Any, list[Any]]:
        """
        Create a plot with multiple curves.
        
        Args:
            title: Title of the plot
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            curve_configs: list of dictionaries with curve configurations
                          Each dict can contain: 'style', 'width', 'size', 'label'
            figsize: Figure size as (width, height)
            grid: Whether to show grid
            grid_alpha: Grid transparency
            
        Returns:
            list containing (figure, axis, list_of_lines) objects
        """
        if curve_configs is None:
            curve_configs = [
                {'style': 'b-o', 'width': 2, 'size': 6, 'label': 'Curve 1'}
            ]
        
        plt.ion()
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if grid:
            ax.grid(True, alpha=grid_alpha)
        
        lines = []
        for config in curve_configs:
            style = config.get('style', 'b-o')
            width = config.get('width', 2)
            size = config.get('size', 6)
            label = config.get('label', f'Curve {len(lines) + 1}')
            
            line, = ax.plot([], [], style, linewidth=width,
                            markersize=size, label=label)
            lines.append(line)
        
        if len(curve_configs) > 1:
            ax.legend()
        
        plt.show()
        return fig, ax, lines
    
    def update_multi_curve_plot(
        self,
        plot_data: list[Any, Any, list[Any]],
        curves_data: list[list[list[float], list[float]]],
        auto_scale: bool = True
    ) -> None:
        """
        Update a multi-curve plot with new data.
        
        Args:
            plot_data: list containing (figure, axis, list_of_lines) objects
            curves_data: list of lists, each containing (x_data, y_data) for a curve
            auto_scale: Whether to auto-scale the axes
        """
        fig, ax, lines = plot_data
        
        for i, (x_data, y_data) in enumerate(curves_data):
            if i < len(lines):
                lines[i].set_data(x_data, y_data)
        
        if auto_scale:
            ax.relim()
            ax.autoscale_view()
        
        plt.draw()
        plt.pause(0.01)
    
    def create_comparison_plot(
        self,
        title: str,
        data_sets: list[list[list[float], list[float], str]],
        xlabel: str = "X",
        ylabel: str = "Y",
        figsize: list[int, int] = (12, 8)
    ) -> list[Any, Any]:
        """
        Create a comparison plot with multiple data sets.
        
        Args:
            title: Title of the plot
            data_sets: list of lists containing (x_data, y_data, label)
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            figsize: Figure size as (width, height)
            
        Returns:
            list containing (figure, axis) objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray']
        styles = ['-o', '--s', '-.^', ':d', '-v', '--<', '-.>', ':p']
        
        for i, (x_data, y_data, label) in enumerate(data_sets):
            color = colors[i % len(colors)]
            style = styles[i % len(styles)]
            ax.plot(x_data, y_data, color + style[1:], 
                   linestyle=style[0], linewidth=2, markersize=6, label=label)
        
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def save_plot(
        self, 
        plot_data: list[Any, Any, Any], 
        filename: str, 
        dpi: int = 300,
        bbox_inches: str = 'tight'
    ) -> None:
        """
        Save a plot to file.
        
        Args:
            plot_data: list containing plot objects (figure is first element)
            filename: Name of the file to save
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box setting
        """
        fig = plot_data[0]
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Plot saved to {filename}")
    
    def close_plot(self, plot_data: list[Any, Any, Any]) -> None:
        """
        Close a plot and free up memory.
        
        Args:
            plot_data: list containing plot objects (figure is first element)
        """
        fig = plot_data[0]
        plt.close(fig)
    
    def close_all_plots(self) -> None:
        """Close all active plots and clear the registry."""
        plt.close('all')
        self.active_plots.clear()
    
    def create_histogram(
        self,
        data: list[float],
        title: str,
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        bins: int = 30,
        figsize: list[int, int] = (10, 6),
        color: str = 'blue',
        alpha: float = 0.7
    ) -> list[Any, Any]:
        """
        Create a histogram plot.
        
        Args:
            data: list of values to plot
            title: Title of the plot
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            bins: Number of bins
            figsize: Figure size as (width, height)
            color: Color of the bars
            alpha: Transparency of the bars
            
        Returns:
            list containing (figure, axis) objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(data, bins=bins, color=color, alpha=alpha, edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
