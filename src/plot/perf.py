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

    max_x = df[x_key].max()

    rel = sns.FacetGrid(df, hue="method", hue_order=method_order, row="p.task", row_order=["classification", "regression"], col="p.max_depth", sharey="row", xlim=(1, max_x*1.05), height=2, aspect=0.8)

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
    
    rel.map(extended_ecdf, x_key)

    rel.set(xscale="log")
    rel.set_xlabels(x_label)
    rel.set_ylabels("Datasets")

    # Extend y-axis by 0.5 on the top
    for ax in rel.axes.flat:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] + 0.5)

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
