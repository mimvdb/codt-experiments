from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from autorank import autorank, latex_report
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

    facet_dim, facet_name = get_facet_dim(df)
    facets = df[facet_dim].unique()

    for dataset, depth in datasetXdepth:
        this_df = df[np.logical_and(df["p.dataset"] == dataset, df["p.max_depth"] == depth)]
        baseline_ub = get_baseline_ub(this_df)

        final_lbs = []
        for facet in facets:
            single = this_df[this_df[facet_dim] == facet]
            lbs = single["o.intermediate_lbs"].squeeze()
            final_lbs.append(lbs[-1][0])
        baseline_lb = max(final_lbs)

        combined = None
        for facet in facets:
            single = this_df[this_df[facet_dim] == facet]
            ubs = single["o.intermediate_ubs"].squeeze()
            lbs = single["o.intermediate_lbs"].squeeze()
            df_u = pd.DataFrame(ubs, columns=["score", "expansions", "time"])
            df_u["type"] = "Upper bound"
            df_l = pd.DataFrame(lbs, columns=["score", "expansions", "time"])
            df_l["type"] = "Lower bound"
            both = pd.concat([df_u, df_l], ignore_index=True)
            both[facet_name] = facet
            if combined is not None:
                combined = pd.concat([combined, both], ignore_index=True)
            else:
                combined = both

        max_x = combined[x_key].max()

        rel = sns.FacetGrid(combined, hue="type", col=facet_name, col_wrap=4, height=2, aspect=0.8, xlim=(0, max_x * 1.05), ylim=(0,baseline_ub * 1.1))
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

def anytime_table(df: pd.DataFrame, output_dir: Path, x_max_key: str, x_key: int, x_label: str):
    datasets = df["p.dataset"].unique()
    datasetXdepth = []
    for dataset in datasets:
        depths = df[df["p.dataset"] == dataset]["p.max_depth"].unique()
        datasetXdepth.extend([(dataset, d) for d in depths])
    datasetXdepth.sort(key=lambda x: (x[1], x[0]))

    facet_dim, facet_name = get_facet_dim(df)
    facets = df[facet_dim].unique()

    integrals = []

    for dataset, depth in datasetXdepth:
        this_df = df[np.logical_and(df["p.dataset"] == dataset, df["p.max_depth"] == depth)]
        baseline_ub = get_baseline_ub(this_df)

        final_lbs = []
        for facet in facets:
            single = this_df[this_df[facet_dim] == facet]
            lbs = single["o.intermediate_lbs"].squeeze()
            final_lbs.append(lbs[-1][0])

        baseline_lb = max(final_lbs)
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

            integrals.append((dataset, depth, facet, objective_integral, gap_integral, single["p.task"].squeeze()))

    result_df = pd.DataFrame(integrals, columns=["dataset", "depth", facet_name, "objective_integral", "gap_integral", "task"])

    x_name = x_max_key[2:]
    with open(output_dir / "objective_integral_table.tex", "w") as f:
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", file=f)
        print(f"% Objective integral table for x={x_label}", file=f)
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", file=f)
        for dataset, depth in datasetXdepth:
            print(f"{dataset}-d{depth} &", file=f)
            this_df = result_df[np.logical_and(result_df["dataset"] == dataset, result_df["depth"] == depth)]

            for facet in facets:
                single = this_df[this_df[facet_name] == facet]
                end = "&" if facet != facets[-1] else "\\\\"
                integral = single["objective_integral"].squeeze()
                print(f"{integral:.2f} {end} % ({dataset} d={depth}) {facet}", file=f)
    
    with open(output_dir / "gap_integral_table.tex", "w") as f:
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", file=f)
        print(f"% Gap integral table for x={x_label}", file=f)
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", file=f)
        for dataset, depth in datasetXdepth:
            print(f"{dataset}-d{depth} &", file=f)
            this_df = result_df[np.logical_and(result_df["dataset"] == dataset, result_df["depth"] == depth)]

            for facet in facets:
                single = this_df[this_df[facet_name] == facet]
                end = "&" if facet != facets[-1] else "\\\\"
                integral = single["gap_integral"].squeeze()
                print(f"{integral:.2f} {end} % ({dataset} d={depth}) {facet}", file=f)

    with open(output_dir / "fig.tex", "w") as f:
        set_style()
        rel = sns.FacetGrid(data=result_df, col="task", row="depth", height=0.8 + 0.25*len(facets), aspect=np.sqrt(8)/np.sqrt(len(facets)))
        rel.map(sns.boxplot, "objective_integral", facet_name, order=facets)
        rel.set_xlabels("Objective integral")
        rel.set_ylabels(facet_name)
        filename = f"fig-anytime-objective-integral-box-{x_name}.pdf"
        plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches = 0.03)
        plt.close()
        caption = f"Integral of the found objective over {x_name} per {facet_name.lower()}."
        label = f"fig:anytime_objective_integral_box_{x_name}"
        print("""\\begin{figure}
    \\centering
    \\includegraphics[width=\\textwidth]{figures/""" + filename + """}
    \\caption{""" + caption + """}
    \\label{""" + label + """}
\\end{figure}""", file=f)
        

        rel = sns.FacetGrid(data=result_df, col="task", row="depth", height=0.8 + 0.25*len(facets), aspect=np.sqrt(8)/np.sqrt(len(facets)))
        rel.map(sns.boxplot, "gap_integral", facet_name, order=facets)
        rel.set_xlabels("Gap integral")
        rel.set_ylabels(facet_name)
        filename = f"fig-anytime-gap-integral-box-{x_name}.pdf"
        plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches = 0.03)
        plt.close()
        caption = f"Integral of the gap between upper and lower bounds over {x_name} per {facet_name.lower()}."
        label = f"fig:anytime_gap_integral_box_{x_name}"
        print("""\\begin{figure}
    \\centering
    \\includegraphics[width=\\textwidth]{figures/""" + filename + """}
    \\caption{""" + caption + """}
    \\label{""" + label + """}
\\end{figure}""", file=f)
        

def tto_table(df: pd.DataFrame, output_dir: Path):
    datasets = df["p.dataset"].unique()
    datasetXdepth = []
    for dataset in datasets:
        depths = df[df["p.dataset"] == dataset]["p.max_depth"].unique()
        datasetXdepth.extend([(dataset, d) for d in depths])
    datasetXdepth.sort(key=lambda x: (x[1], x[0]))

    facet_dim, facet_name = get_facet_dim(df)
    facets = df[facet_dim].unique()

    names = []
    ttop_datasets = []
    ttos_datasets = []

    for dataset, depth in datasetXdepth:
        this_df = df[np.logical_and(df["p.dataset"] == dataset, df["p.max_depth"] == depth)]

        ttop_facets = []
        ttos_facets = []
        for facet in facets:
            single = this_df[this_df[facet_dim] == facet]
            timeout = single["p.timeout"].squeeze()
            mem_limit = single["p.memory_limit"].squeeze()
            mem_used = single["o.memory_usage_bytes"].squeeze()
            ttop = single["o.time"].squeeze()
            invalid = ttop >= timeout or mem_used >= mem_limit
            if invalid:
                ttop = timeout
            last_ub = single["o.intermediate_ubs"].squeeze()[-1]
            last_ub_t = last_ub[2]
            ttos = last_ub_t if not invalid else timeout
            ttop_facets.append(ttop)
            ttos_facets.append(ttos)
        names.append(f"{dataset}")
        ttop_datasets.append(ttop_facets)
        ttos_datasets.append(ttos_facets)

    ttop_df = pd.DataFrame(ttop_datasets, columns=facets)
    ttop_path = output_dir / "ttop"
    ttop_path.mkdir(parents=True, exist_ok=True)
    stdout_io = StringIO()
    with redirect_stdout(stdout_io):
        result = autorank(ttop_df, order="ascending", approach="frequentist", force_mode="nonparametric")
        latex_report(result, figure_path=str(ttop_path), complete_document=False)
        
    stdout_str = stdout_io.getvalue()
    with open(output_dir / "ttop_report.tex", "w") as f:
        f.write(stdout_str)

    ttos_df = pd.DataFrame(ttos_datasets, columns=facets)
    ttos_path = output_dir / "ttos"
    ttos_path.mkdir(parents=True, exist_ok=True)
    stdout_io = StringIO()
    with redirect_stdout(stdout_io):
        result = autorank(ttos_df, order="ascending", approach="frequentist", force_mode="nonparametric")
        latex_report(result, figure_path=str(ttos_path), complete_document=False)
        
    stdout_str = stdout_io.getvalue()
    with open(output_dir / "ttos_report.tex", "w") as f:
        f.write(stdout_str)

    with open(output_dir / "table.tex", "w") as f:
        for i in range(len(names)):
            print(f"{names[i]}&", file=f)
            for j in range(len(ttop_datasets[i])):
                end = "&" if j != len(ttop_datasets[i]) - 1 else "\\\\"
                print(f"{ttop_datasets[i][j]:.0f} {end}% TTOS: ({ttos_datasets[i][j]:.0f}), {facets[j]}", file=f)