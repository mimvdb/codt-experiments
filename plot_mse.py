from src.data import get_dataset
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


def regular(enabled_ll, enabled_lr, enabled_rl, enabled_rr):
    # Compute MSE for left and right splits
    mse_ll = np.mean((enabled_ll - enabled_ll.mean()) ** 2) if len(enabled_ll) > 0 else 0
    mse_lr = np.mean((enabled_lr - enabled_lr.mean()) ** 2) if len(enabled_lr) > 0 else 0
    mse_rl = np.mean((enabled_rl - enabled_rl.mean()) ** 2) if len(enabled_rl) > 0 else 0
    mse_rr = np.mean((enabled_rr - enabled_rr.mean()) ** 2) if len(enabled_rr) > 0 else 0

    # Combine MSEs for left and right splits
    mse_l = (len(enabled_ll) * mse_ll + len(enabled_lr) * mse_lr) / max(len(enabled_ll) + len(enabled_lr), 1)
    mse_r = (len(enabled_rl) * mse_rl + len(enabled_rr) * mse_rr) / max(len(enabled_rl) + len(enabled_rr), 1)
    return mse_l, mse_r


def split_sides_left_only(enabled_ll, enabled_lr, enabled_rl, enabled_rr):
    # Compute SSE for left and right splits
    sse_ll = np.sum((enabled_ll - enabled_ll.mean()) ** 2) if len(enabled_ll) > 0 else 0
    sse_lr = np.sum((enabled_lr - enabled_lr.mean()) ** 2) if len(enabled_lr) > 0 else 0
    return sse_ll, sse_lr


def compute_mse_splits(X_col, y, left_mask, func):
    """
    Computes MSE values for all splits of a feature column.

    Args:
        X_col: Feature column.
        y: Target values.
        left_mask: Boolean mask for the left split.

    Returns:
        Tuple of arrays (mses_l, mses_r) for left and right splits.
    """
    # Sort by feature value
    order = np.argsort(X_col)
    y_sorted = y[order]

    enabled_l = left_mask[order]
    enabled_r = ~enabled_l

    mses_l = []
    mses_r = []

    for i in range(len(y)):
        # Split the data into left and right parts
        y_left = y_sorted[:i]
        y_right = y_sorted[i:]

        # Filter enabled points for left and right splits
        enabled_ll = y_left[enabled_l[:i]]
        enabled_lr = y_right[enabled_l[i:]]
        enabled_rl = y_left[enabled_r[:i]]
        enabled_rr = y_right[enabled_r[i:]]

        mse_l, mse_r = func(enabled_ll, enabled_lr, enabled_rl, enabled_rr)

        mses_l.append(mse_l)
        mses_r.append(mse_r)

    return np.array(mses_l), np.array(mses_r)


def plot_mse(mses_per_feature, X, title, filename, highlight_min=None):
    """
    Plots MSE values for all features with markers at unique indices.

    Args:
        mses_per_feature: List of MSE arrays for each feature.
        X: Feature matrix.
        title: Title of the plot.
        filename: Name of the file to save the plot.
        highlight_min: Tuple (min_idx, min_val, min_feat) to highlight the minimum MSE, or None.
    """
    plt.figure()
    set_style()
    n_features = X.shape[1]

    rows = []
    for i in range(n_features):
        mses = mses_per_feature[i]
        # Get the x values at which we can actually split, ignore others
        unique_indices = np.unique(np.sort(X[:, i]), return_index=True)[1]
        for idx in unique_indices:
            rows.append({"Feature": f"Feature {i}", "Split index": idx, "MSE after split": mses[idx]})
    df = pd.DataFrame(rows)
    sns.lineplot(data=df, x="Split index", y="MSE after split", hue="Feature", marker="o", markersize="3", markeredgewidth=0)

    # Highlight the minimum MSE if provided
    if highlight_min:
        min_idx, min_val, min_feat = highlight_min
        plt.scatter(min_idx, min_val, color='red', label=f"Min Feature {min_feat} (idx={min_idx})", zorder=20)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.03)
    plt.show()


def main():
    X, y = get_dataset("concrete", "regression")
    n_features = X.shape[1]

    # Compute MSE values for all features and find the minimum index and feature
    mses_per_feature = []
    min_idx = -1
    min_feat = -1
    min_val = float('inf')
    for i in range(n_features):
        unique_indices = np.unique(np.sort(X[:, i]), return_index=True)[1]
        left_mask = np.zeros(len(X), dtype=bool)  # Set left_mask to all False for the first plot
        _, mses = compute_mse_splits(X[:, i], y, left_mask, regular)  # Use right MSEs
        mses_per_feature.append(mses)

        # Find the minimum MSE and corresponding feature
        split_mses = mses[unique_indices]
        new_min_idx = np.argmin(split_mses)
        if split_mses[new_min_idx] < min_val:
            min_idx = unique_indices[new_min_idx]
            min_feat = i
            min_val = split_mses[new_min_idx]

    # Plot the first figure
    plot_mse(
        mses_per_feature,
        X,
        "MSE vs Split Index for All Features",
        "fig-gini.pdf",
        highlight_min=(min_idx, min_val, min_feat)
    )

    # Compute MSE for left and right splits seperately
    mses_per_feature_l = []
    mses_per_feature_r = []
    for i in range(n_features):
        unique_indices = np.unique(np.sort(X[:, i]), return_index=True)[1]
        left_mask = np.ones(len(X), dtype=bool)  # Set left_mask to all True so all go left.
        mses_l, mses_r = compute_mse_splits(X[:, i], y, left_mask, split_sides_left_only)  # Use right MSEs
        mses_per_feature_l.append(mses_l)
        mses_per_feature_r.append(mses_r)

    # Plot the figures
    plot_mse(
        mses_per_feature_l,
        X,
        "MSE vs Split Index for All Features",
        "fig-gini-left-side.pdf",
        highlight_min=(min_idx, min_val, min_feat)
    )
    plot_mse(
        mses_per_feature_r,
        X,
        "MSE vs Split Index for All Features",
        "fig-gini-right-side.pdf",
        highlight_min=(min_idx, min_val, min_feat)
    )

    # Compute MSE splits after the best split
    mses_per_feature_l = []
    mses_per_feature_r = []
    sorted_indices = np.argsort(X[:, min_feat])  # Get the sorted indices of the feature
    mask = np.zeros(len(X), dtype=bool)  # Initialize a boolean mask
    mask[sorted_indices[:min_idx + 1]] = True  # Set the mask for the left split based on min_idx

    for i in range(n_features):
        mses_l, mses_r = compute_mse_splits(X[:, i], y, mask, regular)
        mses_per_feature_l.append(mses_l)
        mses_per_feature_r.append(mses_r)

    # Plot MSE after the best split (left and right)
    plot_mse(
        mses_per_feature_l,
        X,
        "MSE vs Split Index for All Features, after the best split point left",
        "fig-gini-after-split-l.pdf"
    )
    plot_mse(
        mses_per_feature_r,
        X,
        "MSE vs Split Index for All Features, after the best split point right",
        "fig-gini-after-split-r.pdf"
    )


if __name__ == "__main__":
    main()
