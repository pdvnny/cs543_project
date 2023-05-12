import numpy as np
import pandas as pd

from visualization import plot_accuracy_results
from visualization import plot_complexity_by_graph_size
from visualization import plot_complexity_by_degree

# =================================================
# ------- HELPER FUNCTIONS -------------
# =================================================

def save_dataframe_to_review(filename: str, df: pd.DataFrame):
    location = f"results/specific_results_to_review/{filename}.csv"
    df.to_csv(location, header=True)

def extract_accuracy_data(df: pd.DataFrame):
    """
    My objective with this method is extract
    data from the df that is relevant for visualizing
    accuracy.
    The features that I need are:
    - vertices (size info)
    - edges (size info)
    - estimated_matching_size
    - theory_maximal_matching
    Conditions:
    - d == 5
    - epsilon == 0.1
    :param df:
    :return:
    """
    df_filter = (df.loc[:, 'd_theory'] == 5) & (df.loc[:, 'epsilon'] == 0.1)
    cols_list = ['vertices', 'edges', 'est_matching_size', 'theory_maximal_matching']
    return df.loc[df_filter, cols_list].reset_index(drop=True)

def aggregate_accuracy_data(df: pd.DataFrame):
    """
    This method combines dataframe rows with the same
    experimental parameters (i.e., because 10 repeats are
    completed)

    :param df:
    :return:
    """
    intermediate_df = df.groupby(by='vertices').agg({'est_matching_size': 'mean',
                                                     'edges': 'mean',
                                                     'theory_maximal_matching': 'mean'})
    # intermediate_df.columns = ['vertices', 'est_matching_size', 'edges', 'theory_maximal_matching']
    return intermediate_df

def get_graph_size_runtime_data(df: pd.DataFrame):
    df_filter = (df.loc[:, 'd_theory'] == 5) & (df.loc[:, 'epsilon'] == 0.1)
    cols_list = ['vertices', 'time', 'recursive_calls']
    inter_df = df.loc[df_filter, cols_list].reset_index(drop=True)
    inter_df2 = inter_df.groupby(by='vertices').agg({'time': 'mean', 'recursive_calls': 'mean'})
    return inter_df2

def get_degree_runtime_data(df: pd.DataFrame):
    df_filter = (df.loc[:, 'vertices'] == 100) & (df.loc[:, 'epsilon'] == 0.1)
    cols_list = ['d_theory', 'time', 'recursive_calls']
    inter_df = df.loc[df_filter, cols_list].reset_index(drop=True)
    inter_df2 = inter_df.groupby(by='d_theory').agg({'time': 'mean', 'recursive_calls': 'mean'})
    return inter_df2

# =================================================
# ------- FUNCTIONS FOR GENERATING RESULTS --------
# =================================================

def generate_accuracy_results():
    # ========================================
    # -------------   ACCURACY ---------------
    # My objective here is to attempt to
    # extract data and visualize how well
    # algorithms approximate the maximal
    # matching
    # Configuration
    # (1) epsilon = 0.1
    # (2) degree = 5
    # ========================================
    data_source_simple = "results/simple_results.csv"
    data_source_nguyen_v1 = "results/nguyen_v1_results.csv"
    data_source_yoshida = "results/yoshida_results.csv"
    df_simple = pd.read_csv(data_source_simple, delimiter=',', header=0)
    df_nguyen = pd.read_csv(data_source_nguyen_v1, delimiter=',', header=0)
    df_yoshida = pd.read_csv(data_source_yoshida, delimiter=',', header=0)
    # print(f"Size of dataframe: {len(df_simple)}")  # 1101 records
    # print(df_simple.head(5))
    # print(df_simple.dtypes)
    df_acc_data = extract_accuracy_data(df_simple)
    df_nguyen_acc = extract_accuracy_data(df_nguyen)
    save_dataframe_to_review("5-12_3am_poor_accuracy_results_for_onak_nguyen_algo", df_nguyen_acc)
    df_yoshida_acc = extract_accuracy_data(df_yoshida)
    # print(f"Size of dataframe: {len(df_acc_data)}")
    # print(df_acc_data.head(10))
    df_acc_data_2 = aggregate_accuracy_data(df_acc_data)
    df_nguyen_acc2 = aggregate_accuracy_data(df_nguyen_acc)
    df_yoshida_acc2 = aggregate_accuracy_data(df_yoshida_acc)
    # print(f"Size of dataframe: {len(df_acc_data_2)}")
    # print(df_acc_data_2.head(10))
    # print(df_acc_data_2.columns)
    # print(df_acc_data_2.index.to_numpy())
    plot_accuracy_results(df_acc_data_2.index.to_numpy(),
                          df_acc_data_2, df_nguyen_acc2, df_yoshida_acc2)

def generate_graph_size_runtime_results():
    # ===========================================
    # ----   GRAPH SIZE RUNTIME DEPENDENCE ----
    # My objective here is to attempt to
    # extract data and visualize the time and
    # query complexity of each sublinear algo
    # as a function of graph size
    # Configuration
    # (1) epsilon = 0.1
    # (2) degree = 5
    # ===========================================
    data_source_simple = "results/simple_results.csv"
    data_source_nguyen_v1 = "results/nguyen_v1_results_attempt2.csv"
    data_source_yoshida = "results/yoshida_results.csv"
    df_simple = pd.read_csv(data_source_simple, delimiter=',', header=0)
    df_nguyen = pd.read_csv(data_source_nguyen_v1, delimiter=',', header=0)
    df_yoshida = pd.read_csv(data_source_yoshida, delimiter=',', header=0)
    df_simple_agg = get_graph_size_runtime_data(df_simple)
    df_nguyen_agg = get_graph_size_runtime_data(df_nguyen)
    df_yoshida_agg = get_graph_size_runtime_data(df_yoshida)
    plot_complexity_by_graph_size(df_simple_agg.index.to_numpy(), df_simple_agg,
                                  df_nguyen_agg, df_yoshida_agg)

def generate_degree_runtime_results():
    # ===========================================
    # ----   GRAPH DEGREE RUNTIME DEPENDENCE ----
    # My objective here is to attempt to
    # extract data and visualize the time and
    # query complexity of each sublinear algo
    # as a function of the degree of vertices
    # in a graph
    # Configuration
    # (1) epsilon = 0.1
    # (2) size = 100
    # ===========================================
    data_source_simple = "results/simple_results.csv"
    data_source_nguyen_v1 = "results/nguyen_v1_results_attempt2.csv"
    data_source_yoshida = "results/yoshida_results.csv"
    df_simple = pd.read_csv(data_source_simple, delimiter=',', header=0)
    df_nguyen = pd.read_csv(data_source_nguyen_v1, delimiter=',', header=0)
    df_yoshida = pd.read_csv(data_source_yoshida, delimiter=',', header=0)
    df_simple_agg = get_degree_runtime_data(df_simple)
    df_nguyen_agg = get_degree_runtime_data(df_nguyen)
    df_yoshida_agg = get_degree_runtime_data(df_yoshida)
    print(df_nguyen_agg)
    plot_complexity_by_degree(df_simple_agg.index.to_numpy(), df_simple_agg,
                              df_nguyen_agg.index.to_numpy(), df_nguyen_agg,
                              df_yoshida_agg.index.to_numpy(), df_yoshida_agg)


# =================================================
# ----------------    MAIN    ---------------------
# =================================================


if __name__ == '__main__':
    # generate_accuracy_results()
    # generate_graph_size_runtime_results()
    generate_degree_runtime_results()
