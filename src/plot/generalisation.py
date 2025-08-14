from pathlib import Path
import pandas as pd

def oos_table(df: pd.DataFrame, output_dir: Path):
    df = df[df["p.test_set"] != ""]
    df["p.max_depth"] = df["p.max_depth"].fillna("Unlimited")
    cells = df.groupby(["p.dataset", "p.method", "p.max_depth"])
    means = cells["o.test_score"].mean()

    best_scores_per_dataset = means.unstack(["p.dataset", "p.max_depth"]).max(axis=0)

    methodXdepths = [
        ("cart", 2), ("quantbnb", 2), ("codt", 2), ("cart", 3), ("quantbnb", 3), ("codt", 3), ("codt", 4), ("cart", "Unlimited")
    ]
    datasets = sorted(df["p.dataset"].unique())

    with open(output_dir / "oos_table.tex", "w") as f:
        for dataset in datasets:
            print(f"{dataset} &", file=f)
            for method, max_depth in methodXdepths:
                cell = cells.get_group((dataset, method, max_depth))
                score = cell["o.test_score"]
                count = score.count()
                mean = score.mean()
                sem = score.sem()

                is_best = max_depth != "Unlimited" and round(mean, 2) >= round(best_scores_per_dataset[(dataset, max_depth)], 2)
                result_str = f"{mean:.2f}"
                if is_best:
                    result_str = "\\textbf{" + result_str + "}"

                end = "&" if (method, max_depth) != methodXdepths[-1] else "\\\\"
                print(f"{result_str} {end} % ({method} d={max_depth}) Mean over {count} test sets SEM=({sem:.2f})", file=f)

