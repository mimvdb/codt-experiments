from pathlib import Path
from typing import Dict
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


def graph_anytime_expansions(results: Dict, output_dir: Path):
    graph_anytime(results, output_dir, "expansions", "Graph expansions")

def graph_anytime_time(results: Dict, output_dir: Path):
    graph_anytime(results, output_dir, "time", "Time (s)")

def graph_anytime(results: Dict, output_dir: Path, x_key: str, x_label: str):
    df = pd.json_normalize(results, max_level=1)
    set_style()

    datasets = df["p.dataset"].unique()
    datasetXdepth = []
    for dataset in datasets:
        depths = df[df["p.dataset"] == dataset]["p.max_depth"].unique()
        datasetXdepth.extend([(dataset, d) for d in depths])
    strategies = df["p.strategy"].unique()

    for dataset, depth in datasetXdepth:
        this_df = df[np.logical_and(df["p.dataset"] == dataset, df["p.max_depth"] == depth)]
        baseline_ub = get_baseline_ub(this_df)

        final_lbs = []
        for strategy in strategies:
            single = this_df[this_df["p.strategy"] == strategy]
            lbs = single["o.intermediate_lbs"].squeeze()
            final_lbs.append(lbs[-1][0])
        baseline_lb = max(final_lbs)

        combined = None
        for strategy in strategies:
            single = this_df[this_df["p.strategy"] == strategy]
            ubs = single["o.intermediate_ubs"].squeeze()
            lbs = single["o.intermediate_lbs"].squeeze()
            df_u = pd.DataFrame(ubs, columns=["score", "expansions", "time"])
            df_u["type"] = "Upper bound"
            df_l = pd.DataFrame(lbs, columns=["score", "expansions", "time"])
            df_l["type"] = "Lower bound"
            both = pd.concat([df_u, df_l], ignore_index=True)
            both["strategy"] = strategy
            if combined is not None:
                combined = pd.concat([combined, both], ignore_index=True)
            else:
                combined = both

        rel = sns.FacetGrid(combined, hue="type", col="strategy", col_wrap=3)
        rel.map(sns.lineplot, x_key, "score", drawstyle="steps-post")
        rel.refline(y=baseline_ub, linestyle="--")
        rel.refline(y=baseline_lb, linestyle="--")
        def objective_integral(data=None, color=None, label=None, **kwargs):
            ax = plt.gca()
            lines = ax.get_lines()
            x, y = lines[0].get_xdata(), lines[0].get_ydata()
            return ax.fill_between(x, y, baseline_lb, step="post", color="C0", alpha=0.3, **kwargs)
        rel.map(objective_integral)
        # rel.set(xscale="log")
        rel.set_xlabels(x_label)
        rel.set_ylabels("Objective")
        filename = f"fig-anytime-{x_key}-{dataset}-d{depth}.pdf"
        plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches = 0.03)
        caption = f"Lower and upper bound over {x_key} for {dataset} dataset (d={depth})"
        label = f"fig:anytime_{x_key}_{dataset}_d{depth}"
        print("""\\begin{figure}
    \\centering
    \\includegraphics[width=\\textwidth]{figures/""" + filename + """}
    \\caption{""" + caption + """}
    \\label{""" + label + """}
\\end{figure}""")

def anytime_table_expansions(results: Dict, output_dir: Path):
    anytime_table(results, output_dir, "o.expansions", 1, "Graph expansions")

def anytime_table_time(results: Dict, output_dir: Path):
    anytime_table(results, output_dir, "o.time", 2, "Time (s)")

def anytime_table(results: Dict, output_dir: Path, x_max_key: str, x_key: int, x_label: str):
    df = pd.json_normalize(results, max_level=1)

    datasets = df["p.dataset"].unique()
    datasetXdepth = []
    for dataset in datasets:
        depths = df[df["p.dataset"] == dataset]["p.max_depth"].unique()
        datasetXdepth.extend([(dataset, d) for d in depths])
    datasetXdepth.sort(key=lambda x: (x[1], x[0]))
    strategies = df["p.strategy"].unique()

    integrals = []

    for dataset, depth in datasetXdepth:
        this_df = df[np.logical_and(df["p.dataset"] == dataset, df["p.max_depth"] == depth)]
        baseline_ub = get_baseline_ub(this_df)

        final_lbs = []
        for strategy in strategies:
            single = this_df[this_df["p.strategy"] == strategy]
            lbs = single["o.intermediate_lbs"].squeeze()
            final_lbs.append(lbs[-1][0])

        baseline_lb = max(final_lbs)
        max_x = this_df[x_max_key].max()

        for strategy in strategies:
            single = this_df[this_df["p.strategy"] == strategy]
            ubs = single["o.intermediate_ubs"].squeeze()
            lbs = single["o.intermediate_lbs"].squeeze()

            # Extend to the end of the graph.
            if ubs[-1][x_key] < max_x:
                ubs.append((ubs[-1][0], max_x, max_x))
            if lbs[-1][x_key] < max_x:
                lbs.append((lbs[-1][0], max_x, max_x))

            def map_key(it, key):
                return np.array(list(map(lambda x: x[key], it)))
            
            def stepwise_integrate(y, x):
                dx = x[1:] - x[:-1]
                return np.sum(dx * y[:-1])

            objective_integral = stepwise_integrate(map_key(ubs, 0), map_key(ubs, x_key))
            gap_integral = objective_integral - stepwise_integrate(map_key(lbs, 0), map_key(lbs, x_key))
            objective_integral /= max_x
            objective_integral -= baseline_lb
            objective_integral /= baseline_ub - baseline_lb
            gap_integral /= max_x
            gap_integral /= baseline_ub

            integrals.append((dataset, depth, strategy, objective_integral, gap_integral))

    result_df = pd.DataFrame(integrals, columns=["dataset", "depth", "strategy", "objective_integral", "gap_integral"])

    print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"% Objective integral table for x={x_label}")
    print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    for dataset, depth in datasetXdepth:
        print(f"{dataset}-d{depth} &")
        this_df = result_df[np.logical_and(result_df["dataset"] == dataset, result_df["depth"] == depth)]

        for strategy in strategies:
            single = this_df[this_df["strategy"] == strategy]
            end = "&" if strategy != strategies[-1] else "\\\\"
            integral = single["objective_integral"].squeeze()
            print(f"{integral:.2f} {end} % ({dataset} d={depth}) {strategy}")
    
    print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"% Gap integral table for x={x_label}")
    print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    for dataset, depth in datasetXdepth:
        print(f"{dataset}-d{depth} &")
        this_df = result_df[np.logical_and(result_df["dataset"] == dataset, result_df["depth"] == depth)]

        for strategy in strategies:
            single = this_df[this_df["strategy"] == strategy]
            end = "&" if strategy != strategies[-1] else "\\\\"
            integral = single["gap_integral"].squeeze()
            print(f"{integral:.2f} {end} % ({dataset} d={depth}) {strategy}")

    set_style()
    rel = sns.jointplot(data=result_df, x="objective_integral", y="gap_integral", hue="strategy", xlim=(0.0,1.0), ylim=(0.0,1.0), marginal_kws={"bw_adjust": .2})
    rel.set_axis_labels("Objective integral", "Gap integral")
    filename = f"fig-anytime-{x_key}.pdf"
    plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches = 0.03)
    caption = f"Anytime performance. The x axis is the integral over {x_key} of the found objective, the y axis is the integral over {x_key} of the gap between lower and upper bound. Values greater than one are excluded from this graph."
    label = f"fig:anytime_{x_key}"
    print("""\\begin{figure}
    \\centering
    \\includegraphics[width=\\textwidth]{figures/""" + filename + """}
    \\caption{""" + caption + """}
    \\label{""" + label + """}
\\end{figure}""")
    
    plt.close()
    
    # Remove showfliers = false for now as they are not informative.
    rel = sns.boxplot(data=result_df, x="objective_integral", y="strategy", showfliers=False)
    rel.set_xlabel("Objective integral")
    rel.set_ylabel("Strategy")
    filename = f"fig-anytime-objective-integral-box-{x_key}.pdf"
    plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches = 0.03)
    caption = f"Integral of the found objective over {x_key} per strategy."
    label = f"fig:anytime_objective_integral_box_{x_key}"
    print("""\\begin{figure}
    \\centering
    \\includegraphics[width=\\textwidth]{figures/""" + filename + """}
    \\caption{""" + caption + """}
    \\label{""" + label + """}
\\end{figure}""")
    
    plt.close()
    
    rel = sns.boxplot(data=result_df, x="gap_integral", y="strategy")
    rel.set_xlabel("Gap integral")
    rel.set_ylabel("Strategy")
    filename = f"fig-anytime-gap-integral-box-{x_key}.pdf"
    plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches = 0.03)
    caption = f"Integral of the gap between upper and lower bounds over {x_key} per strategy."
    label = f"fig:anytime_gap_integral_box_{x_key}"
    print("""\\begin{figure}
    \\centering
    \\includegraphics[width=\\textwidth]{figures/""" + filename + """}
    \\caption{""" + caption + """}
    \\label{""" + label + """}
\\end{figure}""")