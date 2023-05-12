# Parker Dunn (pgdunn@bu.edu)
# Spring 2023

from src import Graph, Vertex, Edge, EdgeList
from counter import RecursiveCounter

from queue import Queue
from random import choice
from math import ceil

import numpy as np

# ==============================================================
# --------- HELPER METHODS -------------------------------------
# ==============================================================

def print_graph(g: Graph):
    """ Function to print a graph as an adjacency list and adjacency matrix."""
    return str(g.get_adjacency_list()) + "\n\n" + str(g.get_adjacency_matrix())

def print_visited(visited: dict):
    print("-- Key -- | -- Val --")
    for key, val in visited.items():
        print(f"{key:^10s}|{str(val):^10s}")

# def get_neighboring_edges(g: Graph):

# ==============================================================
# --------- OTHER GRAPH ALGOS ----------------------------------
# ==============================================================

def BFS(g: Graph):
    q = Queue()
    vertex_names = g.get_vertex_names()
    vertices = g.get_vertices()

    visited = {v: False for v in vertex_names}
    found = []

    start = choice(vertex_names)
    visited[start] = True
    q.put(start)
    found.append(start)

    while not q.empty():
        vtx_name = q.get()
        for nbr in g.get_vertex_neighbors(vtx_name):
            if not visited[nbr]:
                q.put(nbr)
                visited[nbr] = True
                found.append(nbr)

    return found

# ==============================================================
# --------- HELPER METHODS FOR MAXIMAL MATCHING ----------------
# ==============================================================



# ==============================================================
# --------- MAXIMAL VERTEX COVER FUNCS -------------------------
# ==============================================================

def simple_graph_matching(g: Graph) -> (int, EdgeList):
    # num_vertices = len(g)
    visited = dict()
    for v in g.get_vertex_names():
        visited[v] = False

    matching = EdgeList()
    for src, nbrs in g.get_vertices():
        if visited[src]:
            continue

        for nbr in nbrs:
            if not visited[nbr]:
                matching.add_edge(src, nbr)
                visited[src] = True
                visited[nbr] = True
                break

    return len(matching), matching

def graph_matching_nguyen_v1(g: Graph, d: int, epsilon: float) -> (float, int):
    rng = np.random.default_rng()
    counter = RecursiveCounter()
    edge_ranking = dict()
    # Initialize 'edge_ranking'
    for src, neighbors in g.get_vertices():
        edge_ranking[src] = dict()
        for n in neighbors:
            edge_ranking[src][n] = None

    # matching = EdgeList()
    # for src, neighbors in g.get_vertices():
    #     for n in neighbors:
    #         if edge_ranking[src][n] is None:
    #             r = g.generate_edge_rank()
    #             edge_ranking[src][n] = r
    #             edge_ranking[n][src] = r
    #         if oracle_nguyen_v1(src, n, edge_ranking, g, counter):
    #             matching.add_edge(src, n)
    #
    # return matching, counter.get_count()


    # s = ceil(2**d / epsilon**2)  # Number of vertices to check
    s = ceil(1.0 / epsilon**2)  # Number of vertices to check
    # vertices_to_check = []
    # for v in g.get_vertex_names():
        # for i in range(d):
            # vertices_to_check.append(v)
    # set_edges = rng.choice(vertices_to_check, size=s, replace=True)
    set_of_vertices = rng.choice(g.get_vertex_names(), size=min(s, len(g)), replace=False)
    # set_of_vertices = rng.choice(g.get_vertex_names(), size=)

    total_vertices = 0
    possible_vertices = len(set_of_vertices)
    for src in set_of_vertices:
        # possible_vertices += 1
        for dst in g.get_vertex_neighbors(src):
            if edge_ranking[src][dst] is None:
                r = g.generate_edge_rank()
                edge_ranking[src][dst] = r
                edge_ranking[dst][src] = r
            if oracle_nguyen_v1(src, dst, edge_ranking, g, counter):
                total_vertices += 1
                break

    # estimated_matching_size = total_vertices / 2.0
    fraction_vertices_in_matching = total_vertices / float(possible_vertices)
    estimated_matching_size = (fraction_vertices_in_matching * len(g)) / 2.0
    # I switched to this approach for calculating 'estimated_matching_size'
    # as I was compiling results because I realized something was wrong.
    return estimated_matching_size, counter.get_count()

def oracle_nguyen_v1(v1: str, v2: str, edge_ranking: dict, g: Graph, counter: RecursiveCounter):
    counter.increment()
    neighbor_edges = []
    # COLLECT ALL NEIGHBORING EDGES
    for v in [v1, v2]:
        neighbors = g.get_vertex_neighbors(v)
        for n in neighbors:
            if edge_ranking[v][n] is None:
                r = g.generate_edge_rank()
                edge_ranking[v][n] = r
                edge_ranking[n][v] = r
            neighbor_edges.append((v, n, edge_ranking[v][n]))
    # CHECK IF NEIGHBOR EDGES ARE INCLUDED IN MATCHING (BASED ON RANK)
    r_v12 = edge_ranking[v1][v2]
    for v_a, v_b, r_i in neighbor_edges:
        if r_i < r_v12:
            if oracle_nguyen_v1(v_a, v_b, edge_ranking, g, counter):
                return False
    return True

def graph_matching_nguyen_v2(g: Graph) -> (EdgeList, int, int):
    counter = RecursiveCounter()
    edge_ranking = dict()
    selected = dict()
    # Initialize 'edge_ranking'
    for src, neighbors in g.get_vertices():
        edge_ranking[src], selected[src] = dict(), dict()
        for n in neighbors:
            edge_ranking[src][n] = None
            selected[src][n] = None

    matching = EdgeList()
    for src, neighbors in g.get_vertices():
        for n in neighbors:
            if edge_ranking[src][n] is None:
                r = g.generate_edge_rank()
                edge_ranking[src][n] = r
                edge_ranking[n][src] = r
            if oracle_nguyen_v2(src, n, edge_ranking, g, selected, counter):
                matching.add_edge(src, n)
                selected[src][n] = True
                selected[n][src] = True
            else:
                selected[src][n] = False
                selected[n][src] = False
    return matching, counter.get_count(), len(matching)

def oracle_nguyen_v2(v1: str, v2: str, edge_ranking: dict, g: Graph, selected: dict, counter: RecursiveCounter):
    counter.increment()
    neighbor_edges = []
    rankings = []
    # COLLECT ALL NEIGHBORING EDGES
    r_v12 = edge_ranking[v1][v2]
    for v in [v1, v2]:
        neighbors = g.get_vertex_neighbors(v)
        for n in neighbors:
            if edge_ranking[v][n] is None:
                r = g.generate_edge_rank()
                edge_ranking[v][n] = r
                edge_ranking[n][v] = r
            if selected[v][n]:   # If any possible neighbor is selected, then Edge(v1, v2) is not
                return False
            current_ranking = edge_ranking[v][n]
            if r_v12 > current_ranking:
                neighbor_edges.append((v, n))
                rankings.append(current_ranking)
    sorted_idx = np.argsort(rankings)
    for idx in sorted_idx:
        v_a, v_b = neighbor_edges[idx]
        # r_i = rankings[idx]
        if oracle_nguyen_v2(v_a, v_b, edge_ranking, g, selected, counter):
            return False
    return True

def graph_matching_yoshida(g: Graph, d: int, epsilon: float) -> (float, int):
    # --- SETUP ---
    counter = RecursiveCounter()
    # Number of edges to check!
    s = ceil(d**2 / epsilon**2)
    rng = np.random.default_rng()
    # ========   STEP (1) - CREATE AN EDGE RANKING   ========
    # Creating the data structure to hold rankings
    num_edges = int(g.size_edges())
    edge_ranking = dict()
    for src, neighbors in g.get_vertices():
        # edge_ranking[src], selected[src] = dict(), dict()
        edge_ranking[src] = dict()
        for n in neighbors:
            edge_ranking[src][n] = None

    # This is a random ordering of the edges using integers
    # rankings = rng.permutation(list(range(1, num_edges+1)))
    # rank_iter = iter(rankings)
    # for src, neighbors in g.get_vertices():
    #     for n in neighbors:
    #         if edge_ranking[src][n] is None:
    #             next_rank = next(rank_iter)
    #             edge_ranking[src][n] = next_rank
    #             edge_ranking[n][src] = next_rank

    # debugging
    # show = ['0' for _ in range(num_edges)]
    # for src, neighbors in g.get_vertices():
    #     for n in neighbors:
    #         idx = edge_ranking[src][n]
    #         show[idx-1] = f"{src:>4s}-{n:<4s}a.k.a{n:>4s}-{src:<4s}"
    # print("Here is a ranking of the edges!")
    # for i, edge in enumerate(show):
    #     print(f"({i}) {edge}")

    # ======   STEP (2) - RANDOM SELECT s EDGES TO CHECK   ======
    # This is done without knowledge of what edges actually exist
    all_pairs_p = []
    for v in g.get_vertex_names():
        for i in range(d):
            all_pairs_p.append((v, i))
    # set_P = rng.choice(all_pairs_p, size=min(s, len(all_pairs_p)), replace=False)
    set_P = rng.choice(all_pairs_p, size=s, replace=True)

    # debugging
    # print(f"Value of 's': {s}")
    # print(f"The total number of pairs: {len(all_pairs_p)}")
    # # print(f"The size of 'set_P': {len(set_P)}")
    # print("(goal is to figure out if 's' is ever a relevant value! or if the possible pairs is always a low number")
    # print("Contents of P (the multiset of p):")
    # for pp in set_P:
    #     # print(pp)
    #     print(pp[0], type(pp[0]), "-", pp[1], type(pp[1]))

    # =====  STEP (3) - ASK ORACLE WHICH PAIRS ARE IN THE MATCHING  =====
    t = 0
    for p in set_P:
        src = str(p[0])
        neighbor_to_check = int(p[1])
        neighbors = g.get_vertex_neighbors(src)
        if neighbor_to_check < len(neighbors):
            dst = neighbors[neighbor_to_check]
            if edge_ranking[src][dst] is None:
                r = g.generate_edge_rank()
                edge_ranking[src][dst] = r
                edge_ranking[dst][src] = r
            if oracle_yoshida(src, dst, edge_ranking, g, counter):
                t += 1
    n = len(g)  # number of vertices
    queries = counter.get_count()
    return ((t * d * n) / float(s)) - ((epsilon * n) / 2.0), queries

def oracle_yoshida(v1: str, v2: str, edge_ranking: dict, g: Graph, counter: RecursiveCounter):
    counter.increment()
    neighbor_edges = []
    rankings = []
    # COLLECT ALL NEIGHBORING EDGES WITH LOWER RANKINGS
    r_v12 = edge_ranking[v1][v2]
    for v in [v1, v2]:
        neighbors = g.get_vertex_neighbors(v)
        for n in neighbors:
            if edge_ranking[v][n] is None:
                r = g.generate_edge_rank()
                edge_ranking[v][n] = r
                edge_ranking[n][v] = r
            current_ranking = edge_ranking[v][n]
            if r_v12 > current_ranking:
                neighbor_edges.append((v, n))
                rankings.append(current_ranking)
    sorted_idx = np.argsort(rankings)
    for idx in sorted_idx:
        v_a, v_b = neighbor_edges[idx]
        if oracle_yoshida(v_a, v_b, edge_ranking, g, counter):
            return False
    return True


"""My other attempt at implementing "graph_matching_yoshida"""
# def graph_matching_yoshida(g: Graph, d: int, epsilon: float):
#     # --- SETUP ---
#     counter = RecursiveCounter()
#     # Number of edges to check!
#     s = ceil(d**2 / epsilon**2)
#     rng = np.random.default_rng()
#     # STEPS NEEDED FOR ORACLE
#     # ========   STEP (1) - CREATE VERTEX RANKING RANKING   ========
#     vertex_ranking = dict()
#     # vertex_ranking_vals = []
#     # vertex_ind_set = dict()
#     for v in g.get_vertex_names():
#         r = g.generate_edge_rank()  # method just returns a value in (0, 1)
#         vertex_ranking[v] = r
#         # vertex_ind_set[v] = False
#         # vertex_ranking_vals.append(r)
#
#     # sorted_idx = np.argsort(vertex_ranking_vals)
#     # vertices_sorted_by_rank = [g.get_vertex_names()[i] for i in sorted_idx]
#
#     # debugging
#     # print(f"{'Vertex':10s} | {'Rank':10s}")
#     # for v in vertices_sorted_by_rank:
#     #     print(f"{v:10s} | {vertex_ranking[v]:10.3f}")
#
#     # ========  STEP (2) - CREATE A MAXIMAL INDEPENDENT SET  =========
#     # Greedily add v if no neighbor is in the set
#     # for v in vertices_sorted_by_rank:
#     #     add = True
#     #     for n in g.get_vertex_neighbors(v):
#     #         if vertex_ind_set[n]:
#     #             add = False
#     #             break
#     #     vertex_ind_set[v] = add
#
#     # debugging
#     # print('\nVertex Independent Set')
#     # for v, included in vertex_ind_set.items():
#     #     print(f"{v}:  {included}")
#
#     # ======   STEP (3) - RANDOM SELECT s EDGES TO CHECK   ======
#     # This is done without knowledge of what edges actually exist
#     all_pairs_p = []
#     for v in g.get_vertex_names():
#         for i in range(d):
#             all_pairs_p.append((v, i))
#     set_P = rng.choice(all_pairs_p, size=min(s, len(all_pairs_p)), replace=False)
#
#     # debugging
#     print(f"Value of 's': {s}")
#     print(f"The size of 'set_P': {len(set_P)}")
#     print("(goal is to figure out if 's' is ever a relevant value! or if the possible pairs is always a low number")
#     # print("Contents of P (the multiset of p):")
#     # for pp in set_P:
#     #     # print(pp)
#     #     print(pp[0], type(pp[0]), "-", pp[1], type(pp[1]))




