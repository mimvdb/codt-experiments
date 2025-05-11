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


def graph_anytime(results: Dict, output_dir: Path):
    df = pd.json_normalize(results, max_level=1)
    set_style()

    for dataset in df["p.dataset"].unique():
        combined = None
        i = 0
        for strategy, ubs, lbs in df[df["p.dataset"] == dataset][["p.strategy", "o.intermediate_ubs", "o.intermediate_lbs"]].itertuples(index=False, name=None):
            print(strategy)
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
        rel.map(sns.lineplot, "expansions", "score", drawstyle="steps-post")
        rel.set_xlabels("Graph expansions")
        rel.set_ylabels("Objective")
        plt.savefig(output_dir / f"fig-anytime-{dataset}.pdf", bbox_inches="tight", pad_inches = 0.03)