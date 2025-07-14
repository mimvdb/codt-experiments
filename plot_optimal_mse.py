from codt_py import OptimalDecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

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
    for i in range(1, len(y)):
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
        mses.append(weighted_mse)
    return np.array(mses)

def plot_optimal_mse(mses_per_feature, X, title, filename):
    plt.figure()
    set_style()
    n_features = X.shape[1]
    for i in range(n_features):
        mses = mses_per_feature[i]
        sns.lineplot(x=range(1, len(mses)+1), y=mses, label=f"Feature {i}")
    plt.xlabel("Split index")
    plt.ylabel("Sum of optimal MSEs after split")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.03)
    plt.show()

def main():
    X, y = get_dataset("concrete", "regression")
    n_features = X.shape[1]

    # Compute actual optimal MSE for each split using OptimalDecisionTreeRegressor
    optimal_mses_per_feature = []
    for i in range(n_features):
        mses = compute_optimal_mse_splits(X[:, i], y, X)
        optimal_mses_per_feature.append(mses)

    plot_optimal_mse(
        optimal_mses_per_feature,
        X,
        "Sum of Optimal MSEs after Split for All Features",
        "fig-optimal-mse.pdf"
    )

if __name__ == "__main__":
    main()