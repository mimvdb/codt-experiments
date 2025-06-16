from src.data import get_dataset
import numpy as np
import matplotlib.pyplot as plt
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


def compute_mse_splits(X_col, y):
    # Sort by feature value
    order = np.argsort(X_col)
    X_sorted = X_col[order]
    y_sorted = y[order]
    thresholds = (X_sorted[:-1] + X_sorted[1:]) / 2

    mse_total = []

    for i in range(1, len(y)):
        y_left = y_sorted[:i]
        y_right = y_sorted[i:]
        if len(y_left) == 0 or len(y_right) == 0:
            continue
        mse_l = np.mean((y_left - y_left.mean()) ** 2)
        mse_r = np.mean((y_right - y_right.mean()) ** 2)
        mse = (len(y_left) * mse_l + len(y_right) * mse_r) / len(y)
        mse_total.append(mse)

    return thresholds, mse_total


def main():
    X, y = get_dataset("concrete", "regression")
    n_features = X.shape[1]

    plt.figure()
    set_style()
    for i in range(n_features):
        thresholds, mses = compute_mse_splits(X[:, i], y)
        sns.lineplot(x=range(len(mses)), y=mses, label=f"Feature {i}")
    plt.xlabel("Split index")
    plt.ylabel("MSE after split")
    plt.title("MSE vs Split Index for All Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig-gini.pdf", bbox_inches="tight", pad_inches = 0.03)
    plt.show()


if __name__ == "__main__":
    main()
