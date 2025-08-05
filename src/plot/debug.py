from pathlib import Path
import numpy as np
import pandas as pd

def all_strats_equal(df: pd.DataFrame, output_dir: Path):
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
                    mem = single_s["o.memory_usage_bytes"].squeeze() / 1024 / 1024
                    print(f"{score == best_score}: {dataset}-{test_set} d{depth} {method}-{strategy} Time: {time:.2f} Mem: {mem:.2f}MB")
            elif method in ["cart"]:
                # Skip non-optimal
                continue
            else:
                score = single["o.train_score"].squeeze().round(decimals=6)
                time = single["o.time"].squeeze()
                print(f"{score == best_score}: {dataset}-{test_set} d{depth} {method} Time: {time:.2f}")

def time_expansion_ratio_analysis(df: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df = df[df["p.method"] == "codt"]
    df = df[df["p.terminal_solver"] == "left-right"]
    
    # Only include runs that took longer than 1 second
    df = df[df["o.time"] > 1.0]
    
    # Calculate ratios for all rows at once
    df = df.copy()
    df["ratio"] = df["o.time"] / df["o.expansions"]
    
    # Filter out rows where expansions is 0 to avoid division by zero
    df = df[df["o.expansions"] > 0]
    
    # Group by dataset and depth, ignoring test_set and strategy
    grouped = df.groupby(["p.dataset", "p.max_depth"])
    
    print("=== RATIO ANALYSIS PER DATASET-DEPTH ===\n")
    
    all_percentage_devs = []
    high_deviation_runs = []  # Store runs with >100% deviation
    
    for (dataset, depth), group in grouped:
        key = f"{dataset}-d{depth}"
        ratios = group["ratio"].tolist()
        
        # Calculate statistics for this group
        avg_ratio = np.mean(ratios)
        
        if len(ratios) > 1:
            std_dev = np.std(ratios)
            print(f"{key}:")
            print(f"  Average ratio: {avg_ratio:.6f}")
            print(f"  Standard deviation: {std_dev:.6f}")
            print(f"  Min ratio: {min(ratios):.6f}, Max ratio: {max(ratios):.6f}")
            print(f"  Individual runs:")
        else:
            print(f"{key}: Only one data point, ratio: {ratios[0]:.6f}")
            print(f"  Individual runs:")
        
        # Print individual ratios with percentage deviation under the statistics
        for _, row in group.iterrows():
            percentage_dev = (row["ratio"] - avg_ratio) / avg_ratio * 100
            print(f"    {row['p.strategy']} {row['p.branch_relaxation']} Time: {row['o.time']:.4f} Expansions: {row['o.expansions']} Ratio: {row['ratio']:.6f} Deviation: {percentage_dev:.2f}%")
            all_percentage_devs.append(percentage_dev)
            
            # Store runs with >100% deviation
            if abs(percentage_dev) > 100:
                high_deviation_runs.append({
                    'dataset': dataset,
                    'depth': depth,
                    'strategy': row['p.strategy'],
                    'relaxation': row['p.branch_relaxation'],
                    'deviation': percentage_dev
                })
        
        print()  # Add blank line between groups
    
    # Create single box plot of all percentage deviations
    plt.figure(figsize=(8, 6))
    plt.boxplot(all_percentage_devs)
    plt.title('Distribution of Percentage Deviations from Average Time/Expansion Ratio')
    plt.ylabel('Percentage Deviation (%)')
    plt.xticks([1], ['All Runs'])
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(output_dir / "time_expansion_ratio_deviations.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print overall statistics
    print(f"\n=== OVERALL DEVIATION STATISTICS ===")
    print(f"Mean percentage deviation: {np.mean(all_percentage_devs):.2f}%")
    print(f"Median percentage deviation: {np.median(all_percentage_devs):.2f}%")
    print(f"Standard deviation of percentage deviations: {np.std(all_percentage_devs):.2f}%")
    print(f"Max percentage deviation: {np.max(all_percentage_devs):.2f}%")
    print(f"Number of runs: {len(all_percentage_devs)}")
    
    # Print summary of high deviation runs
    print(f"\n=== RUNS WITH >100% DEVIATION ===")
    if high_deviation_runs:
        print(f"Found {len(high_deviation_runs)} runs with >100% deviation:")
        for run in sorted(high_deviation_runs, key=lambda x: x['deviation'], reverse=True):
            print(f"{run['dataset']}-d{run['depth']} {run['strategy']} {run['relaxation']} Deviation: {run['deviation']:.2f}%")
    else:
        print("No runs with >100% deviation found.")