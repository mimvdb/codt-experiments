from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

def all_strats_equal(results: Dict, output_dir: Path):
    df = pd.json_normalize(results, max_level=1)

    datasets = df["p.dataset"].unique()
    datasetXdepth = []
    for dataset in datasets:
        df_with_dataset = df[df["p.dataset"] == dataset]
        depths = df_with_dataset["p.max_depth"].unique()
        for depth in depths:
            test_sets = df_with_dataset[df_with_dataset["p.max_depth"] == depth]["p.test_set"].unique()
            datasetXdepth.extend([(dataset, depth, t) for t in test_sets])
    strategies = df["p.strategy"].unique()
    methods = df["p.method"].unique()

    for dataset, depth, test_set in datasetXdepth:
        this_df = df[np.logical_and(np.logical_and(df["p.dataset"] == dataset, df["p.max_depth"] == depth), df["p.test_set"] == test_set)]
        best_score = this_df["o.train_score"].max().round(decimals=6)
        for method in methods:
            single = this_df[this_df["p.method"] == method]
            if method == "codt":
                for strategy in strategies:
                    single_s = single[single["p.strategy"] == strategy]
                    score = single_s["o.train_score"].squeeze().round(decimals=6)
                    time = single_s["o.time"].squeeze()
                    print(f"{score == best_score}: {dataset}-{test_set} d{depth} {method}-{strategy} Time: {time:.2f}")
            elif method in ["cart"]:
                # Skip non-optimal
                continue
            else:
                score = single["o.train_score"].squeeze().round(decimals=6)
                time = single["o.time"].squeeze()
                print(f"{score == best_score}: {dataset}-{test_set} d{depth} {method} Time: {time:.2f}")