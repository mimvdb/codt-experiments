import numpy as np
import pandas as pd
from src.plot.perf import ecdf_expansions, ecdf_time
from src.plot.plot_dataset_info import plot_dataset_info
from src.plot.ablation import anytime_table_expansions, anytime_table_time, graph_anytime_expansions, graph_anytime_time, tto_table
from src.plot.debug import all_strats_equal, time_expansion_ratio_analysis
from src.plot.generalisation import oos_table

PLOT_FUNCS = {
    "all_strats_equal": all_strats_equal,
    "oos_table": oos_table,
    "graph_anytime_expansions": graph_anytime_expansions,
    "graph_anytime_time": graph_anytime_time,
    "anytime_table_expansions": anytime_table_expansions,
    "anytime_table_time": anytime_table_time,
    "tto_table": tto_table,
    "time_expansion_ratio": time_expansion_ratio_analysis,
    "dataset_info": plot_dataset_info,
    "ecdf_time": ecdf_time,
    "ecdf_expansions": ecdf_expansions,
}

def split_by_attr(attribute, include_all = True):
    def attr_filter(df):
        dfs = []
        attr = df[attribute]
        attrs = attr.unique()
        for t in attrs:
            dfs.append((t, df[attr == t]))
        if include_all:
            dfs.append(("all", df))
        return dfs

    return attr_filter

def ablation_split(df):
    df_s = df[np.logical_and(df["p.terminal_solver"] == "left-right", df["p.branch_relaxation"] == "lowerbound")]
    df_t = df[np.logical_and(df["p.strategy"] == "bfs-balance-small-lb", df["p.branch_relaxation"] == "lowerbound")]
    df_b = df[np.logical_and(df["p.strategy"] == "bfs-balance-small-lb", df["p.terminal_solver"] == "left-right")]
    dfs = [("strategies", df_s), ("terminal_solvers", df_t), ("branch_relaxations", df_b)]
    return dfs

def filter_best(df):
    not_codt = ~(df["p.method"] == "codt")
    all_best = np.logical_and(df["p.terminal_solver"] == "left-right", np.logical_and(df["p.strategy"] == "bfs-balance-small-lb", df["p.branch_relaxation"] == "lowerbound"))
    df = df[np.logical_or(not_codt, np.logical_and(all_best, ~df["p.tune"]))]
    assert not df.duplicated(subset=["p.max_depth", "p.dataset", "p.method"]).any()
    return [("best", df)]

def split_by_strategy_category(df: pd.DataFrame):
    # NOTE: This is not valid for plots that measure relative performance like the objective/gap integral
    # since the best upper/lower bound needs to be in the df to compare against.
    df_dfs = df[df["p.strategy"].str.startswith("dfs")]
    df_other = df[np.logical_or(df["p.strategy"] == "and-or", df["p.strategy"] == "bfs-lds")]
    df_bfs = df[np.logical_and(df["p.strategy"] != "bfs-lds", df["p.strategy"].str.startswith("bfs"))]
    dfs = [("dfs", df_dfs), ("other", df_other), ("bfs", df_bfs)]
    return dfs

FILTER_FUNCS = {
    "split_tasks": split_by_attr("p.task"),
    "split_tasks_no_all": split_by_attr("p.task", False),
    "split_depths": split_by_attr("p.max_depth"),
    "split_depths_no_all": split_by_attr("p.max_depth", False),
    "split_strategy_type": split_by_strategy_category,
    "ablation_split": ablation_split,
    "filter_best": filter_best,
}