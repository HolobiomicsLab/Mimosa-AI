import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Example network data
nodes = {
    'A': (0, 0, 0),
    'B': (1, 2, 3),
    'C': (3, 1, 2),
    'D': (2, 3, 1),
    'E': (4, 0, 3),
    'F': (1, 4, 2)
}

# Define connections (edges)
edges = [
    ('A', 'B'), ('A', 'C'), ('B', 'D'),
    ('C', 'D'), ('D', 'E'), ('E', 'F'),
    ('C', 'F'), ('B', 'E')
]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot edges (connections)
for edge in edges:
    start = nodes[edge[0]]
    end = nodes[edge[1]]
    ax.plot([start[0], end[0]], 
            [start[1], end[1]], 
            [start[2], end[2]], 
            'b-', alpha=0.6, linewidth=2)

# Plot nodes
x_coords = [pos[0] for pos in nodes.values()]
y_coords = [pos[1] for pos in nodes.values()]
z_coords = [pos[2] for pos in nodes.values()]

ax.scatter(x_coords, y_coords, z_coords, 
           c='red', s=100, alpha=0.8)

# Label nodes
for label, pos in nodes.items():
    ax.text(pos[0], pos[1], pos[2], f'  {label}', fontsize=12)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Network Visualization')

plt.show()

# Alternative: Random network generation
def generate_random_network(n_nodes=10, connection_prob=0.3):
    """Generate a random 3D network"""
    # Random 3D positions
    positions = {i: (np.random.uniform(-5, 5), 
                    np.random.uniform(-5, 5), 
                    np.random.uniform(-5, 5)) 
                for i in range(n_nodes)}
    
    # Random connections
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if np.random.random() < connection_prob:
                edges.append((i, j))
    
    return positions, edges

def plot_3d_network(positions, edges, node_colors=None, edge_colors=None):
    """Plot a 3D network with customizable colors"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot edges
    for i, edge in enumerate(edges):
        start = positions[edge[0]]
        end = positions[edge[1]]
        color = edge_colors[i] if edge_colors else 'blue'
        ax.plot([start[0], end[0]], 
                [start[1], end[1]], 
                [start[2], end[2]], 
                color=color, alpha=0.6, linewidth=1.5)
    
    # Plot nodes
    x_coords = [pos[0] for pos in positions.values()]
    y_coords = [pos[1] for pos in positions.values()]
    z_coords = [pos[2] for pos in positions.values()]
    
    colors = node_colors if node_colors else 'red'
    ax.scatter(x_coords, y_coords, z_coords, 
               c=colors, s=50, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Network Visualization')
    
    return fig, ax

# Example usage
positions, edges = generate_random_network(15, 0.2)
fig, ax = plot_3d_network(positions, edges)
plt.show()