import numpy as np
from src.plot.ablation import anytime_table_expansions, anytime_table_time, graph_anytime_expansions, graph_anytime_time
from src.plot.debug import all_strats_equal
from src.plot.generalisation import oos_table

PLOT_FUNCS = {
    "all_strats_equal": all_strats_equal,
    "oos_table": oos_table,
    "graph_anytime_expansions": graph_anytime_expansions,
    "graph_anytime_time": graph_anytime_time,
    "anytime_table_expansions": anytime_table_expansions,
    "anytime_table_time": anytime_table_time,
}

def get_task_filter(include_all = True):
    def task_filter(df):
        dfs = []
        task = df["p.task"]
        tasks = task.unique()
        for t in tasks:
            dfs.append((t, df[task == t]))
        if include_all:
            dfs.append(("all", df))
        return dfs

    return task_filter

def ablation_split(df):
    df_s = df[np.logical_and(df["p.terminal_solver"] == "left-right", df["p.branch_relaxation"] == "lowerbound")]
    df_t = df[np.logical_and(df["p.strategy"] == "bfs-gosdt", df["p.branch_relaxation"] == "lowerbound")]
    df_b = df[np.logical_and(df["p.strategy"] == "bfs-gosdt", df["p.terminal_solver"] == "left-right")]
    dfs = [("strategies", df_s), ("terminal_solvers", df_t), ("branch_relaxations", df_b)]
    return dfs

FILTER_FUNCS = {
    "split_tasks": get_task_filter(),
    "split_tasks_no_all": get_task_filter(False),
    "ablation_split": ablation_split,
}