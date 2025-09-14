from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
import itertools
from pathlib import Path
from autorank import autorank, latex_report, plot_stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.methods import get_method
from src.methods.base import RunParams

def set_style():
    sns.set_context('paper')
    plt.rc('font', size=10, family='serif')
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('axes', labelsize='small', grid=True)
    plt.rc('legend', fontsize='x-small')
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)
    plt.rc('text', usetex = True)
    sns.set_palette("colorblind")

def get_baseline_ub(df):
    dataset = df["p.dataset"].unique().squeeze()[()]
    depth = df["p.max_depth"].unique().squeeze()[()]
    task = df["p.task"].unique().squeeze()[()]
    cp = df["p.cp"].unique().squeeze()[()]

    cart = get_method("cart", task)
    def train_objective(y_true, y_pred):
        if task == "regression":
            x = np.std(y_true) * np.std(y_true)
            sse = np.sum((y_true - y_pred)**2)
            return sse, x
        else:
            x = len(y_true)
            misses = np.count_nonzero(y_true - y_pred)
            return misses, x


    cart_result = cart.run(RunParams("cart", task, -1, dataset, depth, test_set="", cp=cp), scorer=train_objective)
    baseline_score, x = cart_result.train_score
    return baseline_score + x * cp * cart_result.leaves

def get_facet_dim(df):
    s = len(df["p.strategy"].unique())
    t = len(df["p.terminal_solver"].unique())
    b = len(df["p.branch_relaxation"].unique())
    if s != 1:
        assert t == 1 and b == 1
        return "p.strategy", "Strategy"
    elif t != 1:
        assert s == 1 and b == 1
        return "p.terminal_solver", "Terminal solver"
    else:
        assert s == 1 and t == 1
        return "p.branch_relaxation", "Branch relaxation"


def graph_anytime_expansions(df: pd.DataFrame, output_dir: Path):
    graph_anytime(df, output_dir, "expansions", "Graph expansions")

def graph_anytime_time(df: pd.DataFrame, output_dir: Path):
    graph_anytime(df, output_dir, "time", "Time (s)")

def graph_anytime(df: pd.DataFrame, output_dir: Path, x_key: str, x_label: str):
    set_style()
    # Clear file for appending later
    with open(output_dir / "fig.tex", "w") as f:
        f.write("")
    
    datasets = df["p.dataset"].unique()
    datasetXdepth = []
    for dataset in datasets:
        depths = df[df["p.dataset"] == dataset]["p.max_depth"].unique()
        datasetXdepth.extend([(dataset, d) for d in depths])
    datasetXdepth.sort(key=lambda x: (x[1], x[0]))
    facet_dim, facet_name = get_facet_dim(df)
    facets = df[facet_dim].unique()
    df = df_with_integral(df, "o.time", 2, facet_dim, datasetXdepth, facets)
    df = df[df["ss"] != "notinthisgraph"]

    for dataset, depth in datasetXdepth:
        this_df = df[np.logical_and(df["p.dataset"] == dataset, df["p.max_depth"] == depth)]
        baseline_ub = get_baseline_ub(this_df)

        final_ubs = []
        for facet in facets:
            single = this_df[this_df[facet_dim] == facet]
            ubs = single["o.intermediate_ubs"].squeeze()
            final_ubs.append(ubs[-1][0])
        baseline_lb = min(final_ubs)

        combined = None
        for facet in facets:
            single = this_df[this_df[facet_dim] == facet]
            ss = single["ss"].squeeze()
            if ss not in ["DFS", "DFS-Random", "$h_{SmalltLB}$", "LDS", "DFS-ConTree", "AND/OR"]:
                continue
            ubs = single["o.intermediate_ubs"].squeeze()
            lbs = single["o.intermediate_lbs"].squeeze()
            df_u = pd.DataFrame(ubs, columns=["score", "expansions", "time"])
            df_u["type"] = "Upper bound"
            df_l = pd.DataFrame(lbs, columns=["score", "expansions", "time"])
            df_l["type"] = "Lower bound"
            both = pd.concat([df_u, df_l], ignore_index=True)
            both[facet_name] = facet
            both["ss"] = ss
            if combined is not None:
                combined = pd.concat([combined, both], ignore_index=True)
            else:
                combined = both

        max_x = combined[x_key].max()

        rel = sns.FacetGrid(combined, hue="type", col="ss", col_wrap=3, height=1.8, aspect=1.1, xlim=(0, max_x * 1.05), ylim=(0,baseline_ub * 1.1))
        rel.map(sns.lineplot, x_key, "score", drawstyle="steps-post")
        rel.refline(y=baseline_ub, linestyle="--")
        rel.refline(y=baseline_lb, linestyle="--")

        def objective_integral(data=None, color=None, label=None, **kwargs):
            ax = plt.gca()
            lines = ax.get_lines()
            x, y = lines[0].get_xdata(), lines[0].get_ydata()

            # Don't count score if higher than CART
            x = list(x)
            y = list(map(lambda y: min(y, baseline_ub), y))

            # Extend to beyond graph
            x.append(max_x * 1.1)
            y.append(y[-1])
            return ax.fill_between(x, y, baseline_lb, step="post", color="C0", alpha=0.3, **kwargs)
        rel.map(objective_integral)
        # rel.set(xscale="log")
        rel.set_xlabels(x_label)
        rel.set_ylabels("Objective")
        rel.set_titles(template="{col_name}")
        filename = f"fig-anytime-{x_key}-{dataset}-d{depth}.pdf"
        plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches = 0.03)
        plt.close()
        caption = f"Lower and upper bound over {x_key} for {dataset} dataset (d={depth})"
        label = f"fig:anytime_{x_key}_{dataset}_d{depth}"
        with open(output_dir / "fig.tex", "a") as f:
            print("""\\begin{figure}
    \\centering
    \\includegraphics[width=\\textwidth]{figures/""" + filename + """}
    \\caption{""" + caption + """}
    \\label{""" + label + """}
\\end{figure}""", file=f)

def anytime_table_expansions(df: pd.DataFrame, output_dir: Path):
    anytime_table(df, output_dir, "o.expansions", 1, "Graph expansions")

def anytime_table_time(df: pd.DataFrame, output_dir: Path):
    anytime_table(df, output_dir, "o.time", 2, "Time (s)")

def df_with_integral(df: pd.DataFrame, x_max_key: str, x_key: int, facet_dim, datasetXdepth, facets):
    df = df.copy()
    integrals = {}

    for dataset, depth in datasetXdepth:
        this_df = df[np.logical_and(df["p.dataset"] == dataset, df["p.max_depth"] == depth)]
        baseline_ub = get_baseline_ub(this_df)

        final_ubs = []
        for facet in facets:
            single = this_df[this_df[facet_dim] == facet]
            ubs = single["o.intermediate_ubs"].squeeze()
            final_ubs.append(ubs[-1][0])
        baseline_lb = min(final_ubs)

        max_x = this_df[x_max_key].max()

        for facet in facets:
            single = this_df[this_df[facet_dim] == facet]
            ubs = list(single["o.intermediate_ubs"].squeeze()) # NOTE: Make copy, as contents of df may be reused
            lbs = list(single["o.intermediate_lbs"].squeeze())

            # Extend to the end of the graph.
            if ubs[-1][x_key] < max_x:
                ubs.append((ubs[-1][0], max_x, max_x))
            if lbs[-1][x_key] < max_x:
                lbs.append((lbs[-1][0], max_x, max_x))

            def map_key(it, key, at_most=np.inf):
                return np.array(list(map(lambda x: min(x[key], at_most), it)))
            
            def stepwise_integrate(y, x):
                dx = np.divide(np.subtract(x[1:], x[:-1], dtype=np.float64), max_x, dtype=np.float64)
                return np.sum(dx * y[:-1], dtype=np.float64)

            objective_integral = stepwise_integrate(map_key(ubs, 0, baseline_ub), map_key(ubs, x_key))
            gap_integral = objective_integral - stepwise_integrate(map_key(lbs, 0), map_key(lbs, x_key))

            # Scale the area under the curve so a constant CART solution would be one.
            # Subtract the highest lower bound found, so that the best possible method could achieve 0.
            objective_integral -= baseline_lb
            objective_integral /= baseline_ub - baseline_lb

            # Scale the area under the curve so a constant CART solution would be one.
            gap_integral /= baseline_ub

            integrals[(dataset, depth, facet)] = (objective_integral, gap_integral)
    
    def to_saved(i):
        return lambda row: integrals[(row["p.dataset"], row["p.max_depth"], row[facet_dim])][i]
    
    df["objective_integral"] = df.apply(to_saved(0), axis=1)
    df["gap_integral"] = df.apply(to_saved(1), axis=1)
    df["invalid"] = np.logical_or(df["o.memory_usage_bytes"] >= df["p.memory_limit"], df["o.time"] >= df["p.timeout"])
    df["is_trivial"] = df.groupby(["p.dataset", "p.max_depth"])["o.time"].transform("max") < 1
    df["is_easy"] = df.groupby(["p.dataset", "p.max_depth"])["o.time"].transform("max") < 10
    df["is_unsolved"] = df.groupby(["p.dataset", "p.max_depth"])["invalid"].transform("all")
    df["is_solved"] = ~df.groupby(["p.dataset", "p.max_depth"])["invalid"].transform("any")
    df["ttop"] = df.apply(lambda row: row["p.timeout"] if row["invalid"] else row["o.time"], axis=1)
    df["ttos"] = df.apply(lambda row: row["p.timeout"] if row["invalid"] else row["o.intermediate_ubs"][-1][2], axis=1)

    # method_to_label = {
    #     "dfs": "DFS-ConTree",
    #     "dfs-random": "DFS-Random",
    #     "dfs-prio": "DFS",
    # }

    method_to_label = {
        "dfs": "DFS-ConTree",
        "dfs-random": "DFS-Random",
        "dfs-prio": "DFS",
        "bfs-lds": "LDS",
        "bfs-curiosity": "$h_{Curiosity}$",
        "bfs-big": "$h_{Big}$",
        "bfs-small": "$h_{Small}$",
        "bfs-random": "$h_{Random}$",
        "bfs-lb-tiebreak-big": "$h_{LBtBig}$",
        "bfs-lb-tiebreak-small": "$h_{LBtSmall}$",
        "bfs-lb": "$h_{LB}$",
        "bfs-small-tiebreak-lb": "$h_{SmalltLB}$",
        "bfs-big-tiebreak-lb": "$h_{BigtLB}$",
        "and-or": "AND/OR",
        "bfs-balance-big-lb": "$h_{GOSDT}$",
        "bfs-balance-small-lb": "$h_{LB\&Small}$",
    }
    method_to_label = defaultdict(lambda: "notinthisgraph", method_to_label)
    df["ss"] = df[facet_dim].map(method_to_label)
    return df

def anytime_table(df: pd.DataFrame, output_dir: Path, x_max_key: str, x_key: int, x_label: str):
    set_style()
    datasets = df["p.dataset"].unique()
    datasetXdepth = []
    for dataset in datasets:
        depths = df[df["p.dataset"] == dataset]["p.max_depth"].unique()
        datasetXdepth.extend([(dataset, d) for d in depths])
    datasetXdepth.sort(key=lambda x: (x[1], x[0]))
    facet_dim, facet_name = get_facet_dim(df)
    facets = df[facet_dim].unique()
    df = df_with_integral(df, x_max_key, x_key, facet_dim, datasetXdepth, facets)
    df = df[df["ss"] != "notinthisgraph"]
    df = df[~df["is_easy"]]

    do_autorank(pd.pivot(df, columns="ss", index=["p.dataset", "p.max_depth"], values="objective_integral"), output_dir / "autorank", width=6)

    x_name = x_max_key[2:]
    with open(output_dir / "objective_integral_table.tex", "w") as f:
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", file=f)
        print(f"% Objective integral table for x={x_label}", file=f)
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", file=f)
        for dataset, depth in datasetXdepth:
            this_df = df[np.logical_and(df["p.dataset"] == dataset, df["p.max_depth"] == depth)]
            if len(this_df) == 0:
                print(f"% Skipping {dataset} d={depth}, all SS searched less than ten seconds (including OoM)", file=f)
                continue

            print(f"{dataset}-d{depth} &", file=f)
            for facet in facets:
                single = this_df[this_df[facet_dim] == facet]
                end = "&" if facet != facets[-1] else "\\\\"
                integral = single["objective_integral"].squeeze()
                print(f"{integral:.2f} {end} % ({dataset} d={depth}) {facet}", file=f)

    df = df[df["p.max_depth"] >= 3]
    df = df[df["ss"].isin(["DFS", "DFS-Random", "$h_{SmalltLB}$"])]

    row_order = df["p.task"].unique()
    row_order.sort()
    height = 2 if len(row_order) == 1 else 1.8
    aspect = 0.8 if len(row_order) == 1 else 1.4
    rel = sns.FacetGrid(df, hue="ss", row="p.task", row_order=["classification", "regression"], col="p.max_depth", sharex="col", sharey="row", height=height, aspect=aspect, legend_out=False)

    # Use a custom function to extend the line to the right edge
    def extended_ecdf(x, **kwargs):
        ax = plt.gca()
        sns.ecdfplot(x=x, stat="count", complementary=False, **kwargs)
        # Get the current x-axis limits
        xlim = ax.get_xlim()
        # For each line, extend it to the right edge
        for line in ax.lines:
            xdata, ydata = line.get_data()
            if len(xdata) > 0:
                # Add a point at the right edge with the same y-value as the last point
                extended_x = np.append(xdata, xlim[1])
                extended_y = np.append(ydata, ydata[-1])
                line.set_data(extended_x, extended_y)
    
    rel.map(extended_ecdf, "objective_integral")

    rel.set(xscale="log")
    rel.set_xlabels("Objective integral")
    rel.set_ylabels("Datasets")

    # Extend y-axis by 0.5 on the top
    for ax in rel.axes.flat:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] + 0.5)

    # Only integer tickmarks on y-axis
    for ax in rel.axes.flat:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Legend should only show ss in the data
    methods = sorted(df["ss"].unique())
    rel.add_legend(title="Method", label_order=methods)

    # Move the legend to the last ax of the first row of the FacetGrid
    handles, labels = rel.axes.flat[0].get_legend_handles_labels()
    rel.legend.remove()
    leg = rel.axes[1][-1].legend(handles, labels, title=None, loc="upper left", fontsize="6")
    # leg.set_title("Method", prop={"size": "6"})

    rel.set_titles(template="{row_name} $d={col_name}$")
    filename = f"fig-oi-ecdf-{x_max_key[2:]}.pdf"
    plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches = 0.03)
    plt.close()

def do_autorank(df, path, width=4.3):
    path.mkdir(parents=True, exist_ok=True)
    stdout_io = StringIO()
    with redirect_stdout(stdout_io):
        result = autorank(df, order="ascending", approach="frequentist", force_mode="nonparametric")
        latex_report(result, figure_path=str(path), complete_document=False)
        plt.close()
        set_style()
        plot_stats(result, width=width, allow_insignificant=True)
        plt.savefig(path / "fig-significance.pdf", bbox_inches="tight", pad_inches = 0.03)
        plt.close()
        
    stdout_str = stdout_io.getvalue()
    with open(path / "autorank.tex", "w") as f:
        f.write(stdout_str)


def tto_table(df: pd.DataFrame, output_dir: Path):
    set_style()
    datasets = df["p.dataset"].unique()
    datasetXdepth = []
    for dataset in datasets:
        depths = df[df["p.dataset"] == dataset]["p.max_depth"].unique()
        datasetXdepth.extend([(dataset, d) for d in depths])
    datasetXdepth.sort(key=lambda x: (x[1], x[0]))

    facet_dim, facet_name = get_facet_dim(df)
    facets = df[facet_dim].unique()
    df = df_with_integral(df, "o.time", 2, facet_dim, datasetXdepth, facets)
    # df = df[~df["is_easy"]]

    ttop_df = pd.pivot(df, columns="ss", index=["p.dataset", "p.max_depth"], values="ttop")
    ttos_df = pd.pivot(df, columns="ss", index=["p.dataset", "p.max_depth"], values="ttos")

    do_autorank(ttop_df, output_dir / "autorank_ttop", 6)
    do_autorank(ttos_df, output_dir / "autorank_ttos", 6)
    
    non_trivial_nor_unsolved = df[np.logical_and(~df["is_unsolved"], ~df["is_trivial"])]
    # non_trivial_nor_unsolved = df[df["is_trivial"]]
    with open(output_dir / "tto_table.tex", "w") as f:
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", file=f)
        print(f"% TTOP/TTOS table", file=f)
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", file=f)
        for dataset, depth in datasetXdepth:
            this_df = non_trivial_nor_unsolved[np.logical_and(non_trivial_nor_unsolved["p.dataset"] == dataset, non_trivial_nor_unsolved["p.max_depth"] == depth)]

            if len(this_df) == 0:
                print(f"% Skipping {dataset} d={depth}, all SS trivial or unsolved", file=f)
                continue

            best_ttop = this_df["ttop"].min()

            print(f"{dataset}-d{depth} &", file=f)
            for facet in facets:
                single = this_df[this_df[facet_dim] == facet]
                end = "&" if facet != facets[-1] else "\\\\"
                ttop = single["ttop"].squeeze()
                ttos = single["ttos"].squeeze()
                if single["o.memory_usage_bytes"].squeeze() >= single["p.memory_limit"].squeeze():
                    result = "OoM"
                elif single["invalid"].squeeze():
                    result = "-"
                else:
                    result = f"{ttop:.0f}"
                    if best_ttop == ttop:
                        result = "\\textbf{" + result + "}"
                print(f"{result} {end} % TTOS: {ttos:.0f} ({dataset} d={depth}) {facet}", file=f)
    
    df = df[~df["invalid"]]
    df = df[df["ss"].isin(["DFS-Random", "$h_{SmalltLB}$", "AND/OR"])]
    # df = df[df["ss"].isin(["DFS", "DFS-Random", "DFS-ConTree"])]
    # df = df[df["ss"].isin(["DFS", "DFS-Random", "DFS-ConTree", "AND/OR", "$h_{SmalltLB}$"])]
    max_x = df["o.time"].max()

    row_order = df["p.task"].unique()
    row_order.sort()
    height = 2 if len(row_order) == 1 else 1.8
    aspect = 0.8 if len(row_order) == 1 else 0.9
    rel = sns.FacetGrid(df, hue="ss", row="p.task", row_order=["classification", "regression"], col="p.max_depth", sharey="row", xlim=(1, max_x*1.05), height=height, aspect=aspect, legend_out=False)

    # Use a custom function to extend the line to the right edge
    def extended_ecdf(x, **kwargs):
        ax = plt.gca()
        sns.ecdfplot(x=x, stat="count", **kwargs)
        # Get the current x-axis limits
        xlim = ax.get_xlim()
        # For each line, extend it to the right edge
        for line in ax.lines:
            xdata, ydata = line.get_data()
            if len(xdata) > 0:
                # Add a point at the right edge with the same y-value as the last point
                extended_x = np.append(xdata, xlim[1])
                extended_y = np.append(ydata, ydata[-1])
                line.set_data(extended_x, extended_y)
    
    rel.map(extended_ecdf, "o.time")

    rel.set(xscale="log")
    rel.set_xlabels("Time (s)")
    rel.set_ylabels("Datasets")

    # Extend y-axis by 0.5 on the top
    for ax in rel.axes.flat:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] + 0.5)

    # Only integer tickmarks on y-axis
    for ax in rel.axes.flat:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Legend should only show ss in the data
    methods = sorted(df["ss"].unique())
    rel.add_legend(title="Method", label_order=methods)

    # Move the legend to the last ax of the first row of the FacetGrid
    handles, labels = rel.axes.flat[0].get_legend_handles_labels()
    rel.legend.remove()
    leg = rel.axes[0][-1].legend(handles, labels, title=None, loc="upper left", fontsize="6")
    # leg.set_title("Method", prop={"size": "6"})

    rel.set_titles(template="{row_name} $d={col_name}$")
    filename = f"fig-ss-ecdf-time.pdf"
    plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches = 0.03)
    plt.close()


def speedup_d2(df: pd.DataFrame, output_dir: Path):
    set_style()
    method_to_label = {
        "leaf": "None",
        "d1": "Depth-one",
        "left-right": "Left/right",
    }
    method_order = ["Left/right", "Depth-one", "None"]
    df["solver"] = df["p.terminal_solver"].map(method_to_label)

    focus_solvers = np.logical_and(df["p.strategy"] == "bfs-balance-small-lb", df["p.branch_relaxation"] == "lowerbound")
    df = df[np.logical_and(focus_solvers, ~df["p.tune"])]
    depths = [2, 3, 4]

    datasets = sorted(df["p.dataset"].unique())
    speedups_d1 = []
    speedups_left_right = []
    for depth, dataset in itertools.product(depths, datasets):
        this_one = df[df["p.dataset"] == dataset]
        this_one = this_one[this_one["p.max_depth"] == depth]

        if len(this_one[this_one["o.time"] < this_one["p.timeout"] - 1]) != 3:
            print(f"Dataset {dataset} at depth {depth} not considered, one or more timeouts")
            continue
        leaf_y = this_one[this_one["p.terminal_solver"] == "leaf"]["o.time"].squeeze()
        d1_y = this_one[this_one["p.terminal_solver"] == "d1"]["o.time"].squeeze()
        left_right_y = this_one[this_one["p.terminal_solver"] == "left-right"]["o.time"].squeeze()
        speedups_d1.append(leaf_y/d1_y)
        speedups_left_right.append(leaf_y/left_right_y)

    geomean_d1 = np.exp(np.log(speedups_d1).mean())
    geomean_lr = np.exp(np.log(speedups_left_right).mean())
    print(f"Geometric mean speedup of d1 is {geomean_d1:.2f}")
    print(f"Geometric mean speedup of left/right is {geomean_lr:.2f}")

    df = df[df["o.time"] < df["p.timeout"] - 1]
    max_x = df["o.time"].max()

    rel = sns.FacetGrid(df, hue="solver", hue_order=method_order, col="p.max_depth", sharey="row", xlim=(1, max_x*1.05), height=2, aspect=0.8, legend_out=False)

    # Use a custom function to extend the line to the right edge
    def extended_ecdf(x, **kwargs):
        ax = plt.gca()
        sns.ecdfplot(x=x, stat="count", **kwargs)
        # Get the current x-axis limits
        xlim = ax.get_xlim()
        # For each line, extend it to the right edge
        for line in ax.lines:
            xdata, ydata = line.get_data()
            if len(xdata) > 0:
                # Add a point at the right edge with the same y-value as the last point
                extended_x = np.append(xdata, xlim[1])
                extended_y = np.append(ydata, ydata[-1])
                line.set_data(extended_x, extended_y)
    
    rel.map(extended_ecdf, "o.time")

    rel.set(xscale="log")
    rel.set_xlabels("Time (s)")
    rel.set_ylabels("Datasets")

    # Extend y-axis by 0.5 on the top
    for ax in rel.axes.flat:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] + 0.5)

    # Only integer tickmarks on y-axis
    for ax in rel.axes.flat:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Legend should only show method in the data
    rel.add_legend(title="Shallow solver")
    rel.set_titles(template="$d={col_name}$")

    # Move the legend to the last ax of the first row of the FacetGrid
    handles, labels = rel.axes.flat[0].get_legend_handles_labels()
    rel.legend.remove()
    leg = rel.axes[0][-1].legend(handles, labels, title=None, loc="upper left", fontsize="6")
    # leg.set_title("Method", prop={"size": "6"})

    filename = f"fig-solver-ecdf.pdf"
    plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches = 0.03)
    plt.close()
