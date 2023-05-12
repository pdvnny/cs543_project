# Parker Dunn (pgdunn@bu.edu)
# April 2023

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from copy import deepcopy

# ==============================================================
# --------- UNDIRECTED GRAPH IMPLEMENTATION --------------------
# ==============================================================
# Source for core of code: https://pythonandr.com/2016/07/28/implementing-undirected-graphs-in-python/

class Vertex:
    def __init__(self, vertex: str):
        self.name = vertex
        self.neighbors = []

    def add_neighbor(self, neighbor):
        if isinstance(neighbor, Vertex):
            if neighbor.name not in self.neighbors:
                self.neighbors.append(neighbor.name)
                neighbor.neighbors.append(self.name)
                self.neighbors = sorted(self.neighbors)
                neighbor.neighbors = sorted(neighbor.neighbors)
        else:
            return False

    def add_neighbors(self, neighbors):
        for neighbor in neighbors:
            if isinstance(neighbor, Vertex):
                if neighbor.name not in self.neighbors:
                    self.neighbors.append(neighbor.name)
                    neighbor.neighbors.append(self.name)
                    self.neighbors = sorted(self.neighbors)
                    neighbor.neighbors = sorted(neighbor.neighbors)
            else:
                return False

    def __repr__(self):
        return str(self.neighbors)

    def __eq__(self, other):
        return self.name == other.name

"""
A graph implementation that stores
the graph as an adjacency list. There is a dictionary
of "vertices" that contains 
"""
class Graph:
    def __init__(self):
        self.vertices = {}
        self.vertex_names = None
        self.adjacency_matrix = None
        self.vertex_indices = None
        self.rng = np.random.default_rng()
        self.vertex_ranks = None

    def __len__(self):
        return len(self.vertices)

    def size_edges(self):
        edges = 0
        for src, neighbors in self.vertices.items():
            for n in neighbors:
                edges += 1
        return edges / 2  # each edge is counted twice

    def visualize(self, title: str = None):  # Inspiration: https://www.geeksforgeeks.org/visualize-graphs-in-python/
        G = nx.Graph()
        self.adjacency_matrix = self.get_adjacency_matrix()
        edge_list = []
        for v, neighbors in self.vertices.items():
            for n in neighbors:
                edge = (v, n)
                edge_inv = (n, v)
                if edge not in edge_list and edge_inv not in edge_list:
                    edge_list.append(edge)
        # for edge in edge_list:
        #     print(edge)
        G.add_edges_from(edge_list)
        nx.draw_networkx(G)
        if title is None:
            plt.title(f"Visualizing 'Graph' - {title}")
        else:
            plt.title("Visualizing 'Graph'")
        plt.show()

    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex):
            self.vertices[vertex.name] = vertex.neighbors
        self.adjacency_matrix = self.update_adjacency_matrix()

    def add_vertices(self, vertices):
        for vertex in vertices:
            if isinstance(vertex, Vertex):
                self.vertices[vertex.name] = vertex.neighbors
        self.adjacency_matrix = self.update_adjacency_matrix()

    def add_edge(self, vertex_from, vertex_to):
        if isinstance(vertex_from, Vertex) and isinstance(vertex_to, Vertex):
            vertex_from.add_neighbor(vertex_to)
            vertex_to.add_neighbor(vertex_from)
            # if isinstance(vertex_from, Vertex) and isinstance(vertex_to, Vertex):
            self.vertices[vertex_from.name] = vertex_from.neighbors
            self.vertices[vertex_to.name] = vertex_to.neighbors
        self.adjacency_matrix = self.update_adjacency_matrix()

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge[0], edge[1])
        self.adjacency_matrix = self.update_adjacency_matrix()

    def update_vertex_names(self):
        self.vertex_names = sorted(self.vertices.keys())

    def update_adjacency_matrix(self):
        if len(self.vertices) >= 1:
            # self.vertex_names = sorted(self.vertices.keys())
            self.update_vertex_names()
            self.vertex_indices = dict(zip(self.vertex_names, range(len(self.vertex_names))))
            adjacency_matrix = np.zeros(shape=(len(self.vertices), len(self.vertices)))
            for i in range(len(self.vertex_names)):
                for j in range(i, len(self.vertices)):
                    for el in self.vertices[self.vertex_names[i]]:
                        j = self.vertex_indices[el]
                        adjacency_matrix[i, j] = 1
            return adjacency_matrix
        else:
            return dict()  # empty graph

    def get_vertex_names(self):
        self.update_vertex_names()
        return deepcopy(self.vertex_names)

    def get_vertices(self):
        return self.vertices.items()

    def get_vertex_neighbors(self, v: str):
        return self.vertices[v]

    def get_adjacency_matrix(self):
        if len(self.vertices) >= 1:
            self.adjacency_matrix = self.update_adjacency_matrix()
            return self.adjacency_matrix
        else:
            return dict()

    def get_adjacency_list(self):
        if len(self.vertices) >= 1:
            return [str(key) + ": " + str(self.vertices[key]) for key in self.vertices.keys()]
        else:
            return dict()

    def get_average_degree(self):
        degrees = []
        for src, neighbors in self.vertices.items():
            degrees.append(len(neighbors))
        return np.mean(degrees)

    # def simple_vertex_cover(self):
    #     num_vertices = len(self.vertices)
    #     visited = [False] * num_vertices
    #
    #     vertex_cover_vertices = []
    #     vertex_cover_edges = []
    #
    #     # Loop once for all vertices
    #     for src in range(num_vertices):
    #         # pick an edge if src not visited
    #         # and a neighbor is not visited
    #         if visited[src]:
    #             continue
    #             # skip the rest of steps if this vertex has
    #             # already been visited
    #
    #         for dst in self.vertices[str(src)]:
    #             dst_int = int(dst)
    #             if not visited[dst_int]:
    #                 vertex_cover_edges.append((str(src), dst))
    #                 vertex_cover_vertices.append(str(src))
    #                 vertex_cover_vertices.append(dst)
    #                 visited[src] = True
    #                 visited[dst_int] = True
    #                 break
    #     return vertex_cover_vertices, vertex_cover_edges

    def generate_edge_rank(self) -> float:
        return self.rng.random()

    def rank_vertices(self):
        self.update_vertex_names()
        if self.vertex_ranks is None:
            ranks = self.rng.random(size=len(self.vertex_names), dtype=np.float32)
            sorted_idx = np.argsort(ranks)
            self.vertex_ranks = []
            for idx in sorted_idx:
                self.vertex_ranks.append((self.vertex_names[idx], ranks[idx]))
        else:
            raise RuntimeWarning("The vertices in this graph already have rankings.")

    def get_vertex_ranks(self):
        if self.vertex_ranks is None:
            self.rank_vertices()
        return self.vertex_ranks  # A list of tuples

class Edge:
    def __init__(self, src: str, dst: str):
        self.source = src
        self.destination = dst

    # def source(self):
    #     return self.source
    #
    # def destination(self):
    #     return self.destination

    def __eq__(self, other):
        if not isinstance(self.source, str) or not isinstance(other.source, str):
            raise RuntimeWarning("In Edge.__eq__, a source vertex is not of type 'str'")
        if not isinstance(self.destination, str) or not isinstance(other.destination, str):
            raise RuntimeWarning("In Edge.__eq__, a destination vertex is not of type 'str'")
        # print(self)
        # print(other)
        other_vertices = other.get_vertices()
        return self.source in other_vertices and self.destination in other_vertices

    def __repr__(self):
        return f"Edge: {self.source} -> {self.destination}"

    def __str__(self):
        return f"{self.source} -> {self.destination}"

    def get_vertices(self):
        return self.source, self.destination

class EdgeList:
    def __init__(self):
        self.G = []
        self.V = 0
        self.E = 0
        self.SELECTED = None
        self.SELECTED2 = None
        # self.DEGREE = None

    def __contains__(self, item: Edge):
        return item in self.G

    def __repr__(self):
        my_graph = ""
        for e in self.G:
            my_graph += str(e) + "\n"
        return my_graph

    def __str__(self):
        my_graph = ""
        for e in self.G:
            my_graph += str(e) + "\n"
        return my_graph

    def __len__(self):
        return len(self.G)

    def copy_edge_list(self, edges, n: int):
        self.V = n
        self.E += len(edges)
        for e in edges:
            v1, v2 = e.get_vertices()
            self.add_edge(v1, v2)
        self.SELECTED = [None for _ in range(len(self.G))]

    def size(self):
        return self.E

    def add_edge(self, src: str, dst: str, increase_size: bool = True):
        new_edge = Edge(src, dst)
        if new_edge not in self.G:
            self.G.append(new_edge)
        # else:
        #     print(f"WARNING: {new_edge} is already in the graph.")

        if increase_size:
            self.E += 1
        # if not self.is_vertex(src):
        #     self.V += 1
        # if not self.is_vertex(dst):
        #     self.V += 1

    def is_vertex(self, v: str):
        for edge in self.G:
            s, d = edge.get_vertices()
            if s == v or d == v:
                return True
        return False

    # def is_edge(self, src: str, dst: str):
    #     check_edge = Edge(src, dst)

    def has_vertex(self, v: int) -> bool:
        for e in self.G:
            src, dst = e.get_vertices()
            if v == src or v == dst:
                return True
        return False

    def visualize(self, title: str = None):  # Inspiration: https://www.geeksforgeeks.org/visualize-graphs-in-python/
        G = nx.Graph()
        for edge in self.G:  # self.G is an edge list
            v1, v2 = edge.get_vertices()
            G.add_edge(v1, v2)

        nx.draw_networkx(G)
        if title is not None:
            plt.title(f"Visualizing 'EdgeList' - {title}")
        plt.show()

    def get_random_edge(self):
        idx = np.random.default_rng().integers(len(self))
        return self.G[idx]

    def adjacent_vertices(self, v: int):  # -> Sequence[int]:
        adj_vertices = []
        for edge in self.G:
            src, dst = edge.get_vertices()
            if v == src:
                adj_vertices.append(dst)
        return adj_vertices

    def adjacent_edges(self, e: Edge, rand: bool = False):
        output = []
        v1, v2 = e.get_vertices()
        for edge in self.G:
            vertices = edge.get_vertices()
            if (v1 in vertices or v2 in vertices) and (v1 not in vertices or v2 not in vertices):
                output.append(edge)
        if rand:
            rng = np.random.default_rng()
            rng.shuffle(output)
        return output

    def add_permutation(self, p):  # : Sequence[int]):
        new_perm_graph = [self.G[idx] for idx in p]
        self.G = new_perm_graph

    def get_edge_index(self, e: Edge):
        for i in range(len(self.G)):
            if type(e) != type(self.G[i]):
                raise RuntimeWarning("In 'get_edge_index()' - Type of the input edge and graph edge are not the same.")
            if self.G[i] == e:
                return i
        return None

# class VertexSet:
#     def __init__(self):
#         self.vertices = []
#         self.rankings = {}
#
#     def add_vertex(self, v: str, rank: float):


# ==============================================================
# --------- VERTEX COVER APPROXIMATION METHODS -----------------
# ==============================================================

def vertex_cover_bucket_method(graph: np.array, epsilon: float, c1: int or float, c2: int or float):
    """

    :param graph: An adjacency matrix representation of a graph
    :param epsilon: A parameter
    :param c1: A large constant
    :param c2: A large constant
    :return:
    """
    # ----- SETUP ------
    n = len(graph)
    log_n = np.log(n)
    threshold = (epsilon**(3/2) * np.sqrt(n)) / (c1 * log_n)
    num_samples = c2 * (n / threshold) * epsilon**(-2) * log_n

    max_degree = (n - 1)  # Can only have an edge to every other vertex at most

    # ----- HOW MANY BUCKETS MIGHT BE NEEDED -----
    num_buckets = 1
    while (1 + epsilon/10.0)**num_buckets < max_degree:
        num_buckets += 1

    # debugging
    degrees_of_vertices = np.sum(graph, axis=1)  # this is in_degree + out_degree
    avg_degree = np.mean(degrees_of_vertices)
    print(f'===== Graph Information =====\n{n} vertices\n{np.sum(graph, axis=None)} edges\nAverage vertex degree: {avg_degree}')
    print(f"=====      Buckets      =====\n{num_buckets} buckets\nneeded for this graph.")
    return None


# ==============================================================
# --------- EDGES IMPLEMENTATION FOR VISUALIZING GRAPHS --------
# ==============================================================
#
# class Edge:
#     def __init__(self, v1, v2):
#         self.v1 = v1
#         self.v2 = v2
