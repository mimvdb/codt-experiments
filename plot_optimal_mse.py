from codt_py import OptimalDecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from src.data import get_dataset

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

def compute_optimal_mse_splits(X_col, y, X_full):
    """
    For each split index, fit OptimalDecisionTreeRegressor on left and right partitions,
    and return the sum of their MSEs. If left or right side is empty or has only 1 sample, set its cost to 0.
    """
    order = np.argsort(X_col)
    y_sorted = y[order]
    X_sorted = X_full[order]
    mses = []
    unique_indices = np.unique(X_col[order], return_index=True)[1]
    for i in range(1, len(y)):
        if i != 1 and i not in unique_indices:
            continue
        n_left = i
        n_right = len(y) - i
        # Left partition
        if n_left > 1:
            reg_left = OptimalDecisionTreeRegressor(max_depth=1)
            reg_left.fit(X_sorted[:n_left], y_sorted[:n_left])
            pred_left = reg_left.predict(X_sorted[:n_left])
            mse_left = np.mean((y_sorted[:n_left] - pred_left) ** 2)
        else:
            mse_left = 0
        # Right partition
        if n_right > 1:
            reg_right = OptimalDecisionTreeRegressor(max_depth=1)
            reg_right.fit(X_sorted[n_left:], y_sorted[n_left:])
            pred_right = reg_right.predict(X_sorted[n_left:])
            mse_right = np.mean((y_sorted[n_left:] - pred_right) ** 2)
        else:
            mse_right = 0
        # Weighted sum by partition sizes
        weighted_mse = (n_left * mse_left + n_right * mse_right) / (n_left + n_right)
        mses.append((i, weighted_mse))
    return np.array(mses)

def plot_optimal_mse(X, y, title, filename):
    set_style()
    plt.rc('figure', figsize=(6, 1.8)) # Slightly bigger than non-optimal, for the x-axis labels.
    plt.figure()
    n_features = X.shape[1]
    rows = []
    
    # Track minimum values
    min_mse = float('inf')
    min_idx = -1
    min_feat = -1
    
    for i in range(n_features):
        mses = compute_optimal_mse_splits(X[:, i], y, X)
        for idx, mse in mses:
            rows.append({"Feature": f"Feature {i}", "Split index": idx, "MSE": mse})
            # Track the minimum MSE
            if mse < min_mse:
                min_mse = mse
                min_idx = idx
                min_feat = i
    
    df = pd.DataFrame(rows)
    sns.lineplot(data=df, x="Split index", y="MSE", hue="Feature", marker="o", markersize="3", markeredgewidth=0, legend=False)
    
    # Add red dot at minimum MSE
    plt.scatter(min_idx, min_mse, color='red', label=f"Min Feature {min_feat} (idx={min_idx})", zorder=20)
    
    plt.xlabel("Split index")
    plt.ylabel("MSE")
    plt.title(title)

    # Set exactly 2 y tick marks (to be consistent with the other one)
    plt.locator_params(axis='y', nbins=2)

    # plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.03)
    # plt.show()

def main():
    X, y = get_dataset("concrete", "regression")

    plot_optimal_mse(
        X,
        y,
        "Depth-2 MSE",
        "fig-optimal-mse.pdf"
    )

if __name__ == "__main__":
    main()