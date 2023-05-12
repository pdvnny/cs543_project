# Parker Dunn (pgdunn@bu.edu)
# Spring 2023

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_accuracy_results(x: np.array, df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
    """
    :param x: The number of vertices for each entry in the dataframes
    :param df1: A dataframe with data from "simple_results.csv"
    :param df2: A dataframe with data from "nguyen_v1_results_attempt1.csv"
    :param df3: A dataframe with data from "yoshida_results.csv"
    :return:
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey='all')
    math_title = r'\epsilon = 0.1'
    fig.suptitle(f"${math_title}$ and ${r'd=5'}$")

    # (0) COMMON FORMATTING FOR EACH PLOT
    axes[0].set_ylabel("Estimated number of edges")
    for ax in axes:
        ax.set_xlabel("Number of vertices, n")

    # (1) PLOT BASELINE ON EACH PLOT (same information on two plots)
    maximal_matching_size = df1['est_matching_size'].to_numpy()
    num_vertices = x
    upper_bound = [2 * maximal_matching_size[ii] + 0.1 * num_vertices[ii] for ii in range(len(num_vertices))]
    lower_bound = [0.5 * maximal_matching_size[ii] - 0.1 * num_vertices[ii] for ii in range(len(num_vertices))]
    for ax in axes:
        ax.plot(x, df1['theory_maximal_matching'],
                marker="", linestyle='-', color='mediumpurple', label="Best Possible Maximum Matching")
        ax.plot(x, df1['est_matching_size'],
                marker="", color='royalblue', label='Linear Algo')
        math_label = r'(2, \epsilon n)-approx'
        ax.fill_between(x, lower_bound, upper_bound,
                        color='cornflowerblue', alpha=0.4, label=f"${math_label}$")

    # (2) PLOT ACCURACY FOR EACH ALGO ON EACH PLOT
    axes[0].plot(x, df2['est_matching_size'], marker='.', linestyle='-', color='red', label='Sublinear Algo')
    axes[1].plot(x, df3['est_matching_size'], marker='.', linestyle='-', color='red', label='Sublinear Algo')

    # (4) SPECIFIC FORMATING FOR EACH PLOT
    # axes[0].set_title("Greedy Linear Time Algo")
    # axes[1].set_title("Nguyen and Onak, 2008")
    axes[0].set_title("Nguyen and Onak, 2008")
    axes[1].set_title("Yoshida, Yamamoto, and Ito, 2012")

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05)
    # plt.legend()
    axes[0].legend()
    axes[1].legend()
    plt.show()

def plot_complexity_by_graph_size(x, df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
    """
    :param x: The number of vertices for each entry in the dataframes
    :param df1: A dataframe with data from "simple_results.csv"
    :param df2: A dataframe with data from "nguyen_v1_results_attempt1.csv"
    :param df3: A dataframe with data from "yoshida_results.csv"
    :return:
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    math_title = r'\epsilon = 0.1'
    fig.suptitle(f"${math_title}$ and ${r'd=5'}$")

    # (0) COMMON FORMATTING FOR EACH PLOT
    axes[0].set_ylabel("Queries to Oracle")
    axes[1].set_ylabel("Time (seconds)")
    for ax in axes:
        ax.set_xlabel("Number of vertices, N")

    # (1) PLOT RUNTIME IN SECONDS
    axes[1].plot(x, df1['time'], color='royalblue', label='Linear Algo')
    axes[1].plot(x, df2['time'], marker='.', color='orange', label='Nguyen and Onak')
    axes[1].plot(x, df3['time'], marker='.', color='red', label='Yoshida, Yamamoto, and Ito')

    # (2) PLOT NUMBER OF QUERIES TO ORACLE
    axes[0].plot(x, df2['recursive_calls'], marker='.', color='orange', label='Nguyen and Onak')
    axes[0].plot(x, df3['recursive_calls'], marker='.', color='red', label='Yoshida, Yamamoto, and Ito')

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05)
    # plt.legend()
    axes[0].legend()
    axes[1].legend()
    plt.show()

def plot_complexity_by_degree(x1: np.array, df1: pd.DataFrame,
                              x2: np.array, df2: pd.DataFrame,
                              x3: np.array, df3: pd.DataFrame):
    """
    :param x: The degree of the graph for each entry in the dataframes
    :param df1: A dataframe with data from "simple_results.csv"
    :param df2: A dataframe with data from "nguyen_v1_results_attempt1.csv"
    :param df3: A dataframe with data from "yoshida_results.csv"
    :return:
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    math_title = r'\epsilon = 0.1'
    fig.suptitle(f"${math_title}$ and ${r'd=5'}$")

    # (0) COMMON FORMATTING FOR EACH PLOT
    axes[0].set_ylabel("Queries to Oracle")
    axes[1].set_ylabel("Time (seconds)")
    for ax in axes:
        ax.set_xlabel("Degree of vertices, d")

    # (1) PLOT RUNTIME IN SECONDS
    axes[1].plot(x1, df1['time'], color='royalblue', label='Linear Algo')
    axes[1].plot(x2, df2['time'], marker='.', color='black', label='Nguyen and Onak')
    axes[1].plot(x3, df3['time'], marker='.', color='red', label='Yoshida, Yamamoto, and Ito')

    # (2) PLOT NUMBER OF QUERIES TO ORACLE
    axes[0].plot(x2, df2['recursive_calls'], marker='.', color='black', label='Nguyen and Onak')
    axes[0].plot(x3, df3['recursive_calls'], marker='.', color='red', label='Yoshida, Yamamoto, and Ito')
    label1 = r'O(\frac{d^{4}}{\epsilon})'
    axes[0].plot([0, 5, 10, 11, 12, 15, 20], [dd**4 / 0.1**2 for dd in [0, 5, 10, 11, 12, 15, 20]], color='crimson', label=f"${label1}$")
    label2 = r'\frac{2^{O(d)}}{\epsilon^{2}}'
    axes[0].plot([0, 5, 10, 11, 12, 15, 20], [2**dd / 0.1**2 for dd in [0, 5, 10, 11, 12, 15, 20]], color='blue', label=f"${label2}$")

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05)
    # plt.legend()
    axes[0].legend()
    axes[1].legend()
    plt.show()
