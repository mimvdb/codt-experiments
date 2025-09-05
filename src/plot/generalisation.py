from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from autorank import autorank, latex_report, plot_stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

method_to_label = {
    "cart": "CART",
    "codt": "CODTree",
    "quantbnb": "Quant-BnB",
    "contree": "ConTree"
}

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

def oos_table(df: pd.DataFrame, output_dir: Path):
    df = df[df["p.test_set"] != ""]
    cells = df.groupby(["p.dataset", "p.method", "p.max_depth"])
    means = cells["o.test_score"].mean()

    best_scores_per_dataset = means.unstack(["p.dataset", "p.max_depth"]).max(axis=0)

    # methodXdepths = [
    #     ("cart", 2), ("quantbnb", 2), ("codt", 2), ("cart", 3), ("quantbnb", 3), ("codt", 3), ("codt", 4), ("cart", "Unlimited")
    # ]
    methodXdepths = [
        # ("cart", 2), ("codt", 2),
        ("cart", 3), ("codt", 3),
        # ("cart", 4), ("codt", 4),
        ("cart", 20)
    ]
    datasets = sorted(df["p.dataset"].unique())

    significance_rows = []
    significance_labels = map(lambda x: f"{method_to_label[x[0]]} (d={x[1]})", methodXdepths[:-1])

    with open(output_dir / "oos_table.tex", "w") as f:
        for dataset in datasets:
            print(f"{dataset} &", file=f)
            significance_columns = []
            for method, max_depth in methodXdepths:
                cell = cells.get_group((dataset, method, max_depth))
                score = cell["o.test_score"]
                count = score.count()
                mean = score.mean()
                sem = score.sem()
                max_time = cell["o.time"].max()
                max_branches = cell["o.leaves"].max() - 1
                timeout = int(cell["p.timeout"].max() / 11) - 5 if method == "codt" else cell["p.timeout"].max() - 5

                is_best = max_depth != "Unlimited" and round(mean, 2) >= round(best_scores_per_dataset[(dataset, max_depth)], 2)
                result_str = f"{mean:.2f}"
                if is_best:
                    result_str = "\\textbf{" + result_str + "}"
                if max_time > timeout:
                    result_str += "*"

                end = "&" if (method, max_depth) != methodXdepths[-1] else f"& {max_branches}\\\\"
                print(f"{result_str} {end} % ({method} d={max_depth}) Mean over {count} test sets SEM=({sem:.2f})", file=f)
                if (method, max_depth) != methodXdepths[-1]:
                    significance_columns.append(mean)
            significance_rows.append(significance_columns)

    oos_df = pd.DataFrame(significance_rows, columns=significance_labels)
    oos_path = output_dir / "autorank"
    oos_path.mkdir(parents=True, exist_ok=True)
    stdout_io = StringIO()
    with redirect_stdout(stdout_io):
        result = autorank(oos_df, order="descending", approach="frequentist", force_mode="nonparametric")
        latex_report(result, figure_path=str(oos_path), complete_document=False)
        plt.close()
        set_style()
        plot_stats(result, width=4.3)
        plt.savefig(output_dir / "fig-oos-significance.pdf", bbox_inches="tight", pad_inches = 0.03)
        plt.close()
        plt.figure(figsize=(3, 2))
        rank_df = oos_df.rank(axis="columns", ascending=False).mean().sort_values() #.sort_values('meanrank')  # Sort by mean rank for better visualization
        sns.barplot(x=rank_df, y=rank_df.index, orient='h')
        plt.xlabel('Mean Rank')
        plt.ylabel('Method')
        
        plt.tight_layout()
        plt.savefig(output_dir / "fig-oos-mean-ranks.pdf", bbox_inches='tight')
        
    stdout_str = stdout_io.getvalue()
    with open(output_dir / "autorank.tex", "w") as f:
        f.write(stdout_str)