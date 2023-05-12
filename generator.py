# Parker Dunn (pgdunn@bu.edu)
# April 2023

"""==================================================
    Classes & functions for creating random graphs
=================================================="""

import numpy as np

from src import Graph, Vertex
from graph_functions import BFS

# Implementation of "Euclidean Square"

class SquarePoint:
    def __init__(self, x: float = None, y: float = None):
        self.x = np.random.default_rng().random(dtype=np.float32) if x is None else x
        self.y = np.random.default_rng().random(dtype=np.float32) if y is None else y

    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f})"

    def __sub__(self, other):
        return self.dist(other)

    def dist(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx**2 + dy**2)

def simple_generate_graph(d: int, size: int):
    """
    A method that generates a graph with an average vertex
    degree of "d" and a total of "size" vertices.
    :param d:
    :param size:
    :return: An object of type Graph (from src.py)
    """
    min_edges_to_add = max((1, d//2-2))
    max_edges_to_add = min((d//2+2, size-1))
    rng = np.random.default_rng()

    # Create graph with all vertices
    vertices = [Vertex(str(i)) for i in range(size)]  # Vertices 0 to (size-1)
    g = Graph()
    g.add_vertices(vertices)

    for src in range(size):
        num_edges = rng.integers(low=min_edges_to_add, high=max_edges_to_add, size=1)
        dsts = rng.integers(low=0, high=(size-1), size=num_edges)
        for dst in dsts:
            if vertices[src] != vertices[dst]:
                g.add_edge(vertices[src], vertices[dst])
    return g

def generate_graph_with_degree(d: int, size: int = None):
    """
    A method for generating a graph with average vertex of degree d.
    This function uses the "euclidean square" method for generating
    the graph. This means assigning values within a Euclidean Square
    to each vertex, then creating an edge to each of the closest
    d vertices in the graph.

    :param d: An integer representing the desired degree for each
        vertex in the graph
    :param size: The number of vertices in the graph
    """
    rng = np.random.default_rng()
    if size is None:
        size = rng.integers(2*d, 4*d, 1)
        size = size[0]
    vertices = [Vertex(str(i)) for i in range(1, size+1)]
    vertex_point_map = {v.name: SquarePoint() for v in vertices}

    for v1 in rng.permutation(len(vertices)):
        "v1 is an integer corresponding with vertices[v1]"
        distances = []
        edge_options = []
        src = vertices[v1].name
        for v2 in rng.permutation(len(vertices)):
            if v1 == v2:
                continue
            edge_options.append((v1, v2))
            dst = vertices[v2].name
            distances.append(vertex_point_map[src] - vertex_point_map[dst])

        index_sort_by_dist = np.argsort(distances)
        for idx in index_sort_by_dist[:d]:
            v1, v2 = edge_options[idx]
            vertices[v1].add_neighbor(vertices[v2])  # REVERSE IS ALSO ADDED (below unneeded)
            # vertices[v2].add_neighbor(vertices[v1])
    g = Graph()
    g.add_vertices(vertices)
    return g

def generate_graph(d: int, size: int):
    """
    This method uses the 'generate_graph_with_degree' method
    and adds in a feature that ensures the graph is not
    disconnected.
    :param d:
    :param size:
    :return: A graph with a degree of roughly 'd' and size of 'size'
    """
    g = generate_graph_with_degree(d, size)
    reached = BFS(g)
    while len(g) != len(reached):
        g = generate_graph_with_degree(d, size)
        reached = BFS(g)
    return g

