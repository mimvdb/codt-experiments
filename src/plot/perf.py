from contextlib import redirect_stdout
from io import StringIO
import itertools
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


def ecdf_expansions(df: pd.DataFrame, output_dir: Path):
    df = df[df["p.method"] != "quantbnb"]
    df = df[df["p.task"] == "classification"]
    do_ecdf(df, output_dir, "o.expansions", "Graph expansions")

def ecdf_time(df: pd.DataFrame, output_dir: Path):
    do_ecdf(df, output_dir, "o.time", "Time (s)")

def do_ecdf(df: pd.DataFrame, output_dir: Path, x_key: str, x_label: str):
    set_style()

    method_to_label = {
        "cart": "CART",
        "codt": "CODTree (Ours)",
        "quantbnb": "Quant-BnB",
        "contree": "ConTree"
    }
    method_order = ["CODTree (Ours)", "ConTree", "Quant-BnB"]
    df["method"] = df["p.method"].map(method_to_label)
    df = df[df["o.time"] < df["p.timeout"] - 1]

    rel = sns.FacetGrid(df, hue="method", hue_order=method_order, row="p.task", col="p.max_depth", sharey="row", height=2, aspect=0.8)
    rel.map(sns.ecdfplot, x_key, stat="count")
    rel.set(xscale="log")
    rel.set_xlabels(x_label)
    rel.set_ylabels("Datasets")

    # Only integer tickmarks on y-axis
    for ax in rel.axes.flat:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Legend should only show method in the data
    methods = df["method"].unique()
    label_order = [method for method in method_order if method in methods]
    rel.add_legend(title="Method", label_order=label_order)
    rel.set_titles(template="{row_name} $d={col_name}$")
    filename = f"fig-methods-ecdf-{x_key[2:]}.pdf"
    plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches = 0.03)
    plt.close()

