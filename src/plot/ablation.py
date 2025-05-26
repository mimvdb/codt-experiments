from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
        combined = None
        i = 0
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
            i+=1

        rel = sns.FacetGrid(combined, hue="type", col="strategy", col_wrap=3)
        rel.map(sns.lineplot, x_key, "score", drawstyle="steps-post")
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