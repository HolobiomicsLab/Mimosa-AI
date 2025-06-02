import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from collections import deque
from typing import Dict, List, Tuple, Optional, Set

class TreeNode:
    def __init__(self, value, position: Tuple[float, float, float]):
        self.value = value
        self.position = position
        self.children = []
        self.parent = None
        self.level = 0
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        child.level = self.level + 1

class Tree3DVisualizer:
    def __init__(self, figsize=(14, 10)):
        self.nodes: Dict[int, TreeNode] = {}
        self.root: Optional[TreeNode] = None
        self.active_nodes: Set[int] = set()
        self.fig = None
        self.ax = None
        self.animation = None
        self.figsize = figsize
        
        # Visual settings
        self.node_colors = {
            'active': 'lime',
            'inactive': 'lightblue',
            'root': 'gold'
        }
        self.edge_colors = {
            'active': 'lime',
            'inactive': 'gray'
        }
        self.node_sizes = {
            'active': 200,
            'inactive': 80,
            'root': 250
        }
    
    def add_root(self, value: int, position: Tuple[float, float, float] = (0, 0, 0)) -> TreeNode:
        """Add root node to the tree"""
        if self.root is not None:
            raise ValueError("Root already exists")
        
        self.root = TreeNode(value, position)
        self.nodes[value] = self.root
        return self.root
    
    def add_node(self, parent_value: int, child_value: int, position: Optional[Tuple[float, float, float]] = None) -> TreeNode:
        """Add a child node to specified parent"""
        if parent_value not in self.nodes:
            raise ValueError(f"Parent node {parent_value} not found")
        
        parent = self.nodes[parent_value]
        
        # Auto-generate position if not provided
        if position is None:
            position = self._calculate_child_position(parent)
        
        child = TreeNode(child_value, position)
        parent.add_child(child)
        self.nodes[child_value] = child
        
        return child
    
    def _calculate_child_position(self, parent: TreeNode) -> Tuple[float, float, float]:
        """Calculate position for new child based on parent and existing children"""
        num_children = len(parent.children)
        angle_step = 2 * np.pi / max(2, num_children + 1)
        angle = num_children * angle_step + np.random.uniform(-0.2, 0.2)
        
        radius = 3 + parent.level * 1.5
        x = parent.position[0] + radius * np.cos(angle)
        y = parent.position[1] + radius * np.sin(angle)
        z = parent.position[2] - 2.5  # Move down by level
        
        return (x, y, z)
    
    def remove_node(self, value: int) -> bool:
        """Remove node and all its descendants"""
        if value not in self.nodes:
            return False
        
        node = self.nodes[value]
        
        # Remove from parent's children
        if node.parent:
            node.parent.children.remove(node)
        
        # Remove all descendants recursively
        to_remove = [value]
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            queue.extend(current.children)
            to_remove.append(current.value)
        
        # Clean up
        for node_value in to_remove:
            if node_value in self.nodes:
                del self.nodes[node_value]
            self.active_nodes.discard(node_value)
        
        return True
    
    def set_active_node(self, value: int) -> bool:
        """Set single active node"""
        if value not in self.nodes:
            return False
        
        self.active_nodes = {value}
        return True
    
    def add_active_node(self, value: int) -> bool:
        """Add node to active set"""
        if value not in self.nodes:
            return False
        
        self.active_nodes.add(value)
        return True
    
    def remove_active_node(self, value: int) -> bool:
        """Remove node from active set"""
        if value in self.active_nodes:
            self.active_nodes.remove(value)
            return True
        return False
    
    def clear_active_nodes(self):
        """Clear all active nodes"""
        self.active_nodes.clear()
    
    def set_active_level(self, level: int):
        """Set all nodes at specified level as active"""
        self.active_nodes = {node.value for node in self.nodes.values() if node.level == level}
    
    def set_active_subtree(self, root_value: int):
        """Set entire subtree as active"""
        if root_value not in self.nodes:
            return False
        
        subtree_nodes = set()
        queue = deque([self.nodes[root_value]])
        
        while queue:
            current = queue.popleft()
            subtree_nodes.add(current.value)
            queue.extend(current.children)
        
        self.active_nodes = subtree_nodes
        return True
    
    def get_traversal_order(self, method='breadth_first') -> List[int]:
        """Get node traversal order"""
        if not self.root:
            return []
        
        if method == 'breadth_first':
            return self._breadth_first_traversal()
        elif method == 'depth_first':
            return self._depth_first_traversal()
        elif method == 'level_order':
            return self._level_order_groups()
        else:
            raise ValueError(f"Unknown traversal method: {method}")
    
    def _breadth_first_traversal(self) -> List[int]:
        """Breadth-first traversal"""
        result = []
        queue = deque([self.root])
        
        while queue:
            current = queue.popleft()
            result.append(current.value)
            queue.extend(current.children)
        
        return result
    
    def _depth_first_traversal(self) -> List[int]:
        """Depth-first traversal"""
        def dfs(node):
            result = [node.value]
            for child in node.children:
                result.extend(dfs(child))
            return result
        
        return dfs(self.root) if self.root else []
    
    def _level_order_groups(self) -> List[List[int]]:
        """Get nodes grouped by level"""
        levels = []
        if not self.root:
            return levels
        
        queue = deque([self.root])
        
        while queue:
            level_nodes = []
            level_size = len(queue)
            
            for _ in range(level_size):
                current = queue.popleft()
                level_nodes.append(current.value)
                queue.extend(current.children)
            
            if level_nodes:
                levels.append(level_nodes)
        
        return levels
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """Get all edges in the tree"""
        edges = []
        for node in self.nodes.values():
            for child in node.children:
                edges.append((node.value, child.value))
        return edges
    
    def plot_static(self, title="3D Tree"):
        """Plot static tree visualization"""
        if not self.nodes:
            print("No nodes to plot")
            return None, None
        
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self._draw_tree(title)
        return self.fig, self.ax
    
    def _draw_tree(self, title="3D Tree"):
        """Internal method to draw the tree"""
        if not self.ax:
            return
        
        self.ax.clear()
        edges = self.get_edges()
        
        # Draw edges
        for edge in edges:
            start_pos = self.nodes[edge[0]].position
            end_pos = self.nodes[edge[1]].position
            
            color = self.edge_colors['active'] if (edge[0] in self.active_nodes or edge[1] in self.active_nodes) else self.edge_colors['inactive']
            alpha = 0.9 if color == self.edge_colors['active'] else 0.4
            width = 3 if color == self.edge_colors['active'] else 1
            
            self.ax.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        [start_pos[2], end_pos[2]], 
                        color=color, alpha=alpha, linewidth=width)
        
        # Draw nodes
        for node_value, node in self.nodes.items():
            pos = node.position
            
            # Determine node appearance
            if node_value in self.active_nodes:
                color = self.node_colors['active']
                size = self.node_sizes['active']
                alpha = 1.0
                edge_color = 'darkgreen'
            elif node_value == self.root.value if self.root else False:
                color = self.node_colors['root']
                size = self.node_sizes['root']
                alpha = 0.9
                edge_color = 'darkorange'
            else:
                color = self.node_colors['inactive']
                size = self.node_sizes['inactive']
                alpha = 0.7
                edge_color = 'navy'
            
            self.ax.scatter(pos[0], pos[1], pos[2], 
                           c=color, s=size, alpha=alpha, 
                           edgecolors=edge_color, linewidth=2)
            
            # Labels
            weight = 'bold' if node_value in self.active_nodes else 'normal'
            self.ax.text(pos[0], pos[1], pos[2] + 0.5, str(node_value), 
                        fontsize=10, ha='center', weight=weight)
        
        self._set_axis_properties(title)
    
    def _set_axis_properties(self, title):
        """Set axis properties and view"""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(title)
        
        # Auto-scale based on node positions
        if self.nodes:
            positions = [node.position for node in self.nodes.values()]
            x_coords, y_coords, z_coords = zip(*positions)
            
            margin = 2
            self.ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            self.ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
            self.ax.set_zlim(min(z_coords) - margin, max(z_coords) + margin)
        
        self.ax.view_init(elev=20, azim=45)
    
    def animate_traversal(self, traversal_order: List[int], interval=800, repeat=True):
        """Animate node traversal"""
        if not traversal_order:
            print("Empty traversal order")
            return None
        
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        def update_frame(frame):
            active_node = traversal_order[frame % len(traversal_order)]
            self.set_active_node(active_node)
            self._draw_tree(f'Traversing - Active: {active_node}')
        
        self.animation = animation.FuncAnimation(
            self.fig, update_frame, 
            frames=len(traversal_order), 
            interval=interval, repeat=repeat
        )
        
        return self.animation
    
    def animate_level_traversal(self, interval=1200, repeat=True):
        """Animate level-by-level traversal"""
        level_groups = self._level_order_groups()
        if not level_groups:
            print("No levels to animate")
            return None
        
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        def update_level_frame(frame):
            active_level = frame % len(level_groups)
            self.active_nodes = set(level_groups[active_level])
            self._draw_tree(f'Level Traversal - Level {active_level}: {level_groups[active_level]}')
        
        self.animation = animation.FuncAnimation(
            self.fig, update_level_frame, 
            frames=len(level_groups), 
            interval=interval, repeat=repeat
        )
        
        return self.animation
    
    def save_animation(self, filename: str, fps=2):
        """Save current animation to file"""
        if self.animation:
            self.animation.save(filename, writer='pillow', fps=fps)
            print(f"Animation saved as {filename}")
        else:
            print("No animation to save")
    
    def show(self):
        """Display the plot"""
        plt.show()
    
    def get_tree_info(self) -> Dict:
        """Get tree statistics"""
        if not self.nodes:
            return {}
        
        levels = {}
        for node in self.nodes.values():
            if node.level not in levels:
                levels[node.level] = []
            levels[node.level].append(node.value)
        
        return {
            'total_nodes': len(self.nodes),
            'max_level': max(node.level for node in self.nodes.values()),
            'root': self.root.value if self.root else None,
            'active_nodes': list(self.active_nodes),
            'nodes_by_level': levels
        }

# Usage example
def demo():
    # Create visualizer
    viz = Tree3DVisualizer()
    
    # Build tree
    viz.add_root(0)
    viz.add_node(0, 1)
    viz.add_node(0, 2)
    viz.add_node(1, 3)
    viz.add_node(1, 4)
    viz.add_node(2, 5)
    viz.add_node(2, 6)
    viz.add_node(3, 7)
    viz.add_node(4, 8)
    
    print("Tree info:", viz.get_tree_info())
    
    # Static plot
    viz.set_active_node(3)
    viz.plot_static("Static Tree - Node 3 Active")
    viz.show()
    
    # Animate traversal
    traversal = viz.get_traversal_order('breadth_first')
    print("Traversal order:", traversal)
    
    viz.animate_traversal(traversal, interval=600)
    viz.show()
    
    # Animate by levels
    viz.animate_level_traversal(interval=1000)
    viz.show()

if __name__ == "__main__":
    demo()