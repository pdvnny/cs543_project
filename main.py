# Parker Dunn (pgdunn@bu.edu)

import time
from datetime import datetime
from tqdm import tqdm

from src import Graph
from src import EdgeList

from graph_functions import print_graph
from graph_functions import BFS

from graph_functions import simple_graph_matching
from graph_functions import graph_matching_nguyen_v1
from graph_functions import graph_matching_nguyen_v2
from graph_functions import graph_matching_yoshida

from generator import simple_generate_graph
from generator import generate_graph_with_degree
from generator import generate_graph

# ==============================================================
# --------- HELPER FUNCTIONS -----------------------------------
# ==============================================================

def print_graph_to_file(g: Graph):
    date_and_time = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
    out = "logs/recent_graph.txt"
    graph_as_string = print_graph(g)
    with open(out, 'w') as f:
        f.write(graph_as_string)

def print_edge_list_to_file(g: EdgeList):
    date_and_time = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
    out = "logs/recent_edge_list.txt"
    graph_as_string = str(g)
    with open(out, 'w') as f:
        f.write(graph_as_string)

def write_results_to_file(method: str, results: dict):
    if method not in ['nguyen_v1', 'hw2_nguyen_v2', 'yoshida', 'simple']:
        raise Exception(f"Invalid method name provided: {method}")
    file = f"results/{method}_results.csv"

    out = f"{results['d_theory']},{results['vertices']:.0f},{results['edges']:.0f},"
    out += f"{results['epsilon']:.2f},"
    out += f"{results['time']:.4f},"
    out += f"{results['d_true']:.2f},"
    out += f"{results['recursive_calls']:d},"
    out += f"{results['est_matching_size']:.2f},"
    out += f"{results['theory_maximal_matching']:.0f}\n"  # {results['experimental_matching']}\n"

    with open(file, 'a') as f:
        f.write(out)

# """
# For reference:
#
# dict(time=time_elapsed, recursive_calls=queries, vertices=sz,
#                 epsilon=epsilon, d_theory=d, d_true=g.get_average_degree(),
#                 edges=g.size_edges(), est_matching_size=matching_size,
#                 theory_maximal_matching=sz/2
#                 )
# """

# ==============================================================
# --------- TESTING GRAPH GENERATION METHODS -------------------
# ==============================================================

def test_simple_graph_generation():
    d = 2
    num_vertices = 10
    g = simple_generate_graph(d, num_vertices)
    # print(print_graph(g))
    adjacency_lst = g.get_adjacency_list()
    print("------- GENERATED GRAPH -------")
    for vertex_as_string in adjacency_lst:
        print(vertex_as_string)
    g.visualize()
    return g

def test_graph_generation():
    d = 3
    g = generate_graph_with_degree(d)
    adjacency_lst = g.get_adjacency_list()
    print("------- GENERATED GRAPH -------")
    for vertex_as_string in adjacency_lst:
        print(vertex_as_string)
    g.visualize(f"Degree {d}")
    return g

# ==============================================================
# --------- SIMPLE TEST OF GRAPH MATCHING METHODS --------------
# ==============================================================

def test_simple_graph_matching():
    g = test_simple_graph_generation()
    matching = simple_graph_matching(g)
    print("-------- GRAPH MATCHING ---------")
    print(matching)
    matching.visualize("Matching for graph")

def test_oracle_nguyen_v1(d: int = 4, sz: int = 20):
    g = generate_graph(d, sz)
    start = time.time()
    matching, queries = graph_matching_nguyen_v1(g)
    # Returns "EdgeList" object
    time_elapsed = time.time() - start

    print(f"{'='*15}   RESULTS   {'='*15}")
    print(f"Time:\t\t\t{time_elapsed:.3f} seconds")
    print(f"Recursive calls:\t{queries}")
    print()
    sz_graph_edges = g.size_edges()
    print(f"SIZE OF GRAPH: {len(g)} vertices")
    print(f"SIZE OF GRAPH: {sz_graph_edges} edges")
    print()
    print(f"THEORETICAL MAXIMUM MATCHING: {len(g) / 2} edges")
    print(f"SIZE OF MATCHING: {len(matching)} edges")
    print()
    print(f"THEORETICAL AVERAGE DEGREE OF GRAPH: {d}")
    print(f"TRUE AVERAGE DEGREE OF GRAPH: {g.get_average_degree()}")
    print()
    print_graph_to_file(g)
    print_edge_list_to_file(matching)

def test_oracle_nguyen_v2(d: int = 4, sz: int = 20):
    g = generate_graph(d, sz)
    start = time.time()
    matching, queries = graph_matching_nguyen_v2(g)
    # Returns "EdgeList" object
    time_elapsed = time.time() - start

    print(f"{'='*15}   RESULTS   {'='*15}")
    print(f"Time:\t\t\t{time_elapsed:.3f} seconds")
    print(f"Recursive calls:\t{queries}")
    print()
    sz_graph_edges = g.size_edges()
    print(f"SIZE OF GRAPH: {len(g)} vertices")
    print(f"SIZE OF GRAPH: {sz_graph_edges} edges")
    print()
    print(f"THEORETICAL MAXIMUM MATCHING: {len(g) / 2} edges")
    print(f"SIZE OF MATCHING: {len(matching)} edges")
    print()
    print(f"THEORETICAL AVERAGE DEGREE OF GRAPH: {d}")
    print(f"TRUE AVERAGE DEGREE OF GRAPH: {g.get_average_degree()}")
    print()
    print_graph_to_file(g)
    print_edge_list_to_file(matching)

def test_oracle_yoshida(d: int = 4, sz: int = 20):
    g = generate_graph(d, sz)
    # g.visualize()
    start = time.time()
    matching_size, queries = graph_matching_yoshida(g, d, epsilon=0.1)
    # Returns "EdgeList" object
    time_elapsed = time.time() - start
    print(f"{'='*15}   RESULTS   {'='*15}")
    print(f"Time:\t\t\t{time_elapsed:.3f} seconds")
    print(f"Recursive calls:\t{queries}")
    print()
    sz_graph_edges = g.size_edges()
    print(f"SIZE OF GRAPH: {len(g)} vertices")
    print(f"SIZE OF GRAPH: {sz_graph_edges} edges")
    print()
    print(f"THEORETICAL MAXIMUM MATCHING: {len(g) / 2} edges")
    print(f"SIZE OF MATCHING: {matching_size} edges")
    print()
    print(f"THEORETICAL AVERAGE DEGREE OF GRAPH: {d}")
    print(f"TRUE AVERAGE DEGREE OF GRAPH: {g.get_average_degree()}")
    print()
    print_graph_to_file(g)
    # print_edge_list_to_file(matching_size)

# ==============================================================
# --------- EXPERIMENTATION PROCEDURES -------------------------
# ==============================================================

def evaluate_simple_graph_matching(g: Graph, d: int, epsilon: float):
    start = time.time()
    matching_size, _ = simple_graph_matching(g)
    time_elapsed = time.time() - start
    return dict(time=time_elapsed, recursive_calls=0, vertices=len(g),
                epsilon=epsilon, d_theory=d, d_true=g.get_average_degree(),
                edges=g.size_edges(), est_matching_size=matching_size,
                theory_maximal_matching=len(g)/2
                )


def evaluate_nguyen_onak_method(g: Graph, d: int, sz: int, epsilon: float):
    start = time.time()
    matching_size, queries = graph_matching_nguyen_v1(g, d, epsilon)
    time_elapsed = time.time() - start
    return dict(time=time_elapsed, recursive_calls=queries, vertices=sz,
                epsilon=epsilon, d_theory=d, d_true=g.get_average_degree(),
                edges=g.size_edges(), est_matching_size=matching_size,
                theory_maximal_matching=sz/2
                )


def evaluate_yoshida_method(g: Graph, d: int, sz: int, epsilon: float):
    start = time.time()
    matching_size, queries = graph_matching_yoshida(g, d, epsilon)
    time_elapsed = time.time() - start
    return dict(time=time_elapsed, recursive_calls=queries, vertices=sz,
                epsilon=epsilon, d_theory=d, d_true=g.get_average_degree(),
                edges=g.size_edges(), est_matching_size=matching_size,
                theory_maximal_matching=sz/2
                )


"""
This method is configured to naively check if each edge is
in the maximal matching
"""
def evaluate_hw2_method(g: Graph, d: int, sz: int, epsilon: float):
    start = time.time()
    _, queries, matching_size = graph_matching_nguyen_v2(g)
    time_elapsed = time.time() - start
    return dict(time=time_elapsed, recursive_calls=queries, vertices=sz,
                epsilon=epsilon, d_theory=d, d_true=g.get_average_degree(),
                edges=g.size_edges(), est_matching_size=matching_size,
                theory_maximal_matching=sz/2
                )


def experimental_procedure_v1():
    repeats = 10
    ds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]  # 30, 40, 50]
    sizes = [10, 50, 100]  # 500, 1000]  # I want to test 10000 too!
    epsilons = [0.05, 0.1, 0.2]
    for d in ds:
        for sz in sizes:
            for e in epsilons:
                # simple_results = []
                # nguyen_onak_results = []
                # yoshida_results = []
                # hw2_results = []

                for _ in tqdm(range(repeats), desc=f"d={d} | size={sz} | e={e:.2f}"):
                    g = generate_graph(d, sz)

                    out_simple = evaluate_simple_graph_matching(g, d, e)
                    # simple_results.append(out_simple)
                    out_n_o = evaluate_nguyen_onak_method(g, d, sz, e)
                    # nguyen_onak_results.append(out_n_o)
                    out_yoshida = evaluate_yoshida_method(g, d, sz, e)
                    # yoshida_results.append(out_yoshida)
                    out_hw2 = evaluate_hw2_method(g, d, sz, e)
                    # hw2_results.append(out_hw2)

                    write_results_to_file('simple', out_simple)
                    write_results_to_file('nguyen_v1', out_n_o)
                    write_results_to_file('yoshida', out_yoshida)
                    write_results_to_file('hw2_nguyen_v2', out_hw2)
                    # break

                # Average and store results
                # process_results()
                # process_results()
                # process_results()
                # process_results()

                # break
            # break
        # break
    # End of "experimental_procedure_v1"

def experimental_procedure_v2():
    repeats = 10
    # ds = [50, 100]
    # sizes = [100, 1000]
    ds = [10, 50, 100]
    sizes = [1000]
    epsilons = [0.1]
    for d in ds:
        for sz in sizes:
            for e in epsilons:
                for _ in tqdm(range(repeats), desc=f"d={d} | size={sz} | e={e:.2f}"):
                    g = generate_graph(d, sz)
                    out_simple = evaluate_simple_graph_matching(g, d, e)
                    out_n_o = evaluate_nguyen_onak_method(g, d, sz, e)
                    out_yoshida = evaluate_yoshida_method(g, d, sz, e)

                    write_results_to_file('simple', out_simple)
                    write_results_to_file('nguyen_v1', out_n_o)
                    write_results_to_file('yoshida', out_yoshida)
    # End of "experimental_procedure_v2"

def experimental_procedure_v3():
    repeats = 10
    ds = [5]
    sizes = [500, 1000]
    epsilons = [0.1]
    for d in ds:
        for sz in sizes:
            for e in epsilons:
                for _ in tqdm(range(repeats), desc=f"d={d} | size={sz} | e={e:.2f}"):
                    g = generate_graph(d, sz)
                    # out_simple = evaluate_simple_graph_matching(g, d, e)
                    # out_n_o = evaluate_nguyen_onak_method(g, d, sz, e)
                    out_yoshida = evaluate_yoshida_method(g, d, sz, e)

                    # write_results_to_file('simple', out_simple)
                    # write_results_to_file('nguyen_v1', out_n_o)
                    write_results_to_file('yoshida', out_yoshida)
    # End of "experimental_procedure_v3"

def regenerating_nguyen_results_procedure():
    repeats = 10

    # ds = [2, 3, 4, 5, 6, 7]  # 6, 7, 8, 9, 10, 20, 50]  # 30, 40, 50]
    # sizes = [10, 50, 100]  # 500, 1000]  # I want to test 10000 too!
    # epsilons = [0.05, 0.1, 0.2]
    # for d in ds:
    #     for sz in sizes:
    #         for e in epsilons:
    #             for _ in tqdm(range(repeats), desc=f"d={d} | size={sz} | e={e:.2f}"):
    #                 g = generate_graph(d, sz)
    #
    #                 out_n_o = evaluate_nguyen_onak_method(g, d, sz, e)
    #                 write_results_to_file('nguyen_v1', out_n_o)

    # ds = [10] # 20, 30, 40, 50]
    ds = [11, 12, 13, 14, 15]
    # sizes = [500, 1000]
    sizes = [100]
    epsilons = [0.1]
    for d in ds:
        for sz in sizes:
            for e in epsilons:
                for _ in tqdm(range(repeats), desc=f"d={d} | size={sz} | e={e:.2f}"):
                    g = generate_graph(d, sz)

                    out_n_o = evaluate_nguyen_onak_method(g, d, sz, e)
                    write_results_to_file('nguyen_v1', out_n_o)


# ==============================================================
# --------- TESTING "OTHER" GRAPH METHODS ----------------------
# ==============================================================

def test_bfs():
    g = test_graph_generation()
    vertices_reached = BFS(g)
    print("---- VERTICES FOUND VIA BFS ----")
    for v in sorted(vertices_reached):
        print(v, end=" - ")


if __name__ == '__main__':
    # g = test_graph_generation()
    # test_bfs()

    # test_simple_graph_matching()
    # test_oracle_nguyen_v1(d=10, sz=100)
    # test_oracle_nguyen_v2(d=10, sz=100)
    # test_oracle_yoshida(d=10, sz=10000)
    #    Interesting observation
    #    Number of vertices |   s =     |    Number of pairs that could be checked
    #    -------------------|-----------|------- (not number selected)------------
    #        100            |   10000   |      1,000
    #        1000           |   10000   |      10,000
    #        10000          |   10000   |      100,000
    # All degree = 10

    # experimental_procedure_v1()
    # experimental_procedure_v2()
    # experimental_procedure_v3()
    regenerating_nguyen_results_procedure()


# def test_simple_vertex_cover():
#     g = test_graph_generation()
#     vertex_cover_vertices, vertex_cover_as_edge_list = g.simple_vertex_cover()
#     print("\n=================\n  VERTEX COVER  \n=================\n")
#     for edge in vertex_cover_as_edge_list:
#         print(edge[0], "--", edge[1])
